"""LoRA fine-tuning of Whisper on accented / domain-specific English speech.

A deliberately explicit PyTorch training loop (no ``Trainer``) so every moving part is
visible: PEFT LoRA on the attention ``q_proj``/``v_proj`` projections, gradient
accumulation, mixed precision (``fp16`` with a ``GradScaler`` or ``bf16``), linear
warm-up/decay, periodic WER evaluation and adapter-only checkpoints.

Typical runs::

    # Free Colab T4: EdAcc (accented English), fp16, effective batch 16
    python -m src.train_lora --dataset edinburghcstr/edacc --precision fp16 \\
        --batch-size 8 --grad-accum-steps 2 --max-steps 1000

    # Your own recordings: a JSONL/CSV manifest with "audio" and "text" columns
    python -m src.train_lora --manifest data/train.jsonl --eval-manifest data/dev.jsonl

    # 20-second wiring check on any machine (tiny model, synthetic audio)
    python -m src.train_lora --smoke-test

The resulting directory holds only the adapter (a few MB) and is loaded for inference
with ``--lora <dir>`` or ``asr.lora_checkpoint`` in the config.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):  # allow ``python src/train_lora.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import (
    SAMPLE_RATE,
    DeviceManager,
    PathLike,
    get_logger,
    load_audio,
    load_config,
    normalize_text,
    parse_overrides,
    resample,
)

logger = get_logger(__name__)

Example = Tuple[np.ndarray, str]

# --------------------------------------------------------------------------- data


class SpeechRecords(Dataset):
    """Map-style dataset yielding ``(waveform_16k, transcript)`` pairs.

    Records are either local files (``{"audio": path, "text": ...}``) or rows of a
    HuggingFace dataset whose audio column was cast with ``decode=False`` so the raw
    bytes are decoded here with ``soundfile`` (no torchcodec / ffmpeg requirement).
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        audio_key: str = "audio",
        text_key: str = "text",
        max_audio_sec: float = 30.0,
    ) -> None:
        self.records = records
        self.audio_key = audio_key
        self.text_key = text_key
        self.max_samples = int(max_audio_sec * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def decode_audio(source: Any) -> np.ndarray:
        """Decode a path, raw bytes, a datasets ``{"bytes","path"}`` dict or an array to 16 kHz mono."""
        if isinstance(source, np.ndarray):
            return source.astype(np.float32, copy=False)
        if isinstance(source, Mapping):
            if source.get("array") is not None:
                array = np.asarray(source["array"], dtype=np.float32)
                return resample(array, int(source.get("sampling_rate", SAMPLE_RATE)), SAMPLE_RATE)
            if source.get("bytes"):
                import soundfile as sf

                data, sr = sf.read(io.BytesIO(source["bytes"]), dtype="float32", always_2d=True)
                return resample(data.mean(axis=1), sr, SAMPLE_RATE)
            source = source.get("path")
        if source is None:
            raise ValueError("Record has no decodable audio")
        return load_audio(str(source), SAMPLE_RATE)

    def __getitem__(self, index: int) -> Example:
        record = self.records[int(index)]
        waveform = self.decode_audio(record[self.audio_key])[: self.max_samples]
        return waveform, str(record[self.text_key]).strip()


def read_manifest(path: PathLike) -> List[Dict[str, str]]:
    """Read a JSONL / CSV / TSV manifest with ``audio`` and ``text`` columns.

    Relative audio paths are resolved against the manifest's directory.
    """
    path = Path(path)
    rows: List[Dict[str, str]] = []
    if path.suffix.lower() in {".jsonl", ".json"}:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter=delimiter))
    for row in rows:
        audio = row.get("audio") or row.get("audio_path") or row.get("path")
        if audio is None or "text" not in row:
            raise ValueError(f"Manifest rows need 'audio' and 'text' fields: {path}")
        audio_path = Path(audio)
        row["audio"] = str(audio_path if audio_path.is_absolute() else (path.parent / audio_path))
    return rows


def load_records(
    manifest: Optional[PathLike] = None,
    dataset: Optional[str] = None,
    split: str = "validation",
    audio_column: str = "audio",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    max_audio_sec: float = 30.0,
    seed: int = 42,
    dataset_config: Optional[str] = None,
) -> SpeechRecords:
    """Build a :class:`SpeechRecords` from a local manifest or a HuggingFace dataset.

    Args:
        manifest: JSONL/CSV manifest; takes precedence over ``dataset``.
        dataset: Hub dataset id, e.g. ``edinburghcstr/edacc``.
        split: Dataset split to load.
        audio_column / text_column: Column names in the Hub dataset.
        max_samples: Randomly subsample to this many utterances.
        max_audio_sec: Clips are truncated to this length (Whisper sees at most 30 s).
        seed: Seed of the subsampling shuffle.
        dataset_config: Optional Hub dataset configuration name.
    """
    if manifest is not None:
        rows = read_manifest(manifest)
        if max_samples is not None and len(rows) > max_samples:
            random.Random(seed).shuffle(rows)
            rows = rows[:max_samples]
        logger.info("Loaded %d utterances from %s", len(rows), manifest)
        return SpeechRecords(rows, "audio", "text", max_audio_sec)
    if dataset is None:
        raise ValueError("Provide either a manifest or a dataset name")

    from datasets import Audio, load_dataset

    logger.info("Loading %s [%s] (the first call downloads the dataset)", dataset, split)
    hub = load_dataset(dataset, dataset_config, split=split) if dataset_config else load_dataset(dataset, split=split)
    hub = hub.cast_column(audio_column, Audio(decode=False))
    if max_samples is not None and len(hub) > max_samples:
        hub = hub.shuffle(seed=seed).select(range(max_samples))
    logger.info("Loaded %d utterances from %s [%s]", len(hub), dataset, split)
    return SpeechRecords(hub, audio_column, text_column, max_audio_sec)


def synthetic_records(n: int = 8, seed: int = 0) -> SpeechRecords:
    """Tiny in-memory dataset (noise + CS sentences) used by ``--smoke-test``."""
    rng = np.random.default_rng(seed)
    sentences = [
        "gradient descent minimizes the loss function",
        "backpropagation applies the chain rule",
        "the eigenvalue of the covariance matrix",
        "a homography maps one plane to another",
    ]
    rows = [
        {"audio": (0.01 * rng.standard_normal(SAMPLE_RATE * 2)).astype(np.float32), "text": sentences[i % 4]}
        for i in range(n)
    ]
    return SpeechRecords(rows)


@dataclass
class WhisperCollator:
    """Turn ``(waveform, text)`` pairs into padded ``input_features`` and ``labels``.

    Padding positions in ``labels`` are set to ``-100`` so they are ignored by the loss.
    The leading ``<|startoftranscript|>`` is removed because the model prepends
    ``decoder_start_token_id`` itself when it shifts the labels right.
    """

    processor: Any
    max_label_tokens: int = 448

    def __call__(self, batch: Sequence[Example]) -> Dict[str, Any]:
        waveforms = [wave for wave, _ in batch]
        texts = [text for _, text in batch]
        features = self.processor.feature_extractor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        tokens = self.processor.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_label_tokens, return_tensors="pt"
        )
        labels = tokens.input_ids.masked_fill(tokens.attention_mask.ne(1), -100)
        if bool((labels[:, 0] == self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all()):
            labels = labels[:, 1:]
        return {"input_features": features.input_features, "labels": labels, "texts": texts}


# --------------------------------------------------------------------------- precision


@dataclass
class PrecisionPlan:
    """How the forward pass is cast for a given device.

    Attributes:
        name: ``fp32``, ``fp16`` or ``bf16``.
        autocast_dtype: dtype passed to ``torch.autocast`` or ``None`` for full precision.
        use_scaler: Whether a ``GradScaler`` is required (fp16 on CUDA only).
    """

    name: str
    autocast_dtype: Optional[torch.dtype]
    use_scaler: bool


def resolve_precision(requested: str, device: torch.device) -> PrecisionPlan:
    """Map ``auto``/``fp16``/``bf16``/``fp32`` to something the device can actually train with.

    * CUDA: ``auto`` -> ``fp16`` (the fastest option on a Colab T4); ``bf16`` when
      requested and supported, without a gradient scaler.
    * MPS / CPU: full ``fp32``. Half-precision *training* on Metal lacks a gradient
      scaler and under/overflows easily, so a request for it is downgraded with a warning.
    """
    requested = (requested or "auto").lower()
    if device.type == "cuda":
        if requested == "bf16":
            if torch.cuda.is_bf16_supported():
                return PrecisionPlan("bf16", torch.bfloat16, False)
            logger.warning("bf16 is not supported on this GPU; using fp16.")
            return PrecisionPlan("fp16", torch.float16, True)
        if requested in {"auto", "fp16"}:
            return PrecisionPlan("fp16", torch.float16, True)
        return PrecisionPlan("fp32", None, False)
    if requested in {"fp16", "bf16"}:
        logger.warning("%s training is only enabled on CUDA; using fp32 on %s.", requested, device.type)
    return PrecisionPlan("fp32", None, False)


# --------------------------------------------------------------------------- model


def build_lora_model(cfg: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Load Whisper, freeze it and attach LoRA adapters to the configured projections.

    Returns:
        ``(peft_model, processor)``.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = str(cfg["base_model"])
    processor = WhisperProcessor.from_pretrained(base)
    processor.tokenizer.set_prefix_tokens(language=str(cfg.get("language", "en")), task=str(cfg.get("task", "transcribe")))
    model = WhisperForConditionalGeneration.from_pretrained(base)
    # Labels already contain the language/task prefix, so nothing may be forced on top.
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA r=%d on %s: %.2fM trainable / %.1fM total parameters (%.2f%%)",
        lora_config.r, lora_config.target_modules, trainable / 1e6, total / 1e6, 100.0 * trainable / total,
    )  # fmt: skip
    return peft_model, processor


def linear_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up to the base LR followed by linear decay to zero."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def cycle(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    """Iterate over a DataLoader forever (reshuffles at every pass)."""
    while True:
        yield from loader


# --------------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(
    model: Any,
    processor: Any,
    loader: DataLoader,
    device_manager: DeviceManager,
    plan: PrecisionPlan,
    language: str = "en",
    task: str = "transcribe",
    max_new_tokens: int = 128,
) -> Dict[str, float]:
    """Compute teacher-forced loss and greedy-decoding WER on ``loader``."""
    import jiwer

    device = device_manager.device
    model.eval()
    losses: List[float] = []
    references: List[str] = []
    hypotheses: List[str] = []
    for batch in loader:
        features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device.type, dtype=plan.autocast_dtype, enabled=plan.autocast_dtype is not None):
            losses.append(float(model(input_features=features, labels=labels).loss))
            generated = model.generate(
                input_features=features, language=language, task=task, max_new_tokens=max_new_tokens
            )
        hypotheses.extend(processor.batch_decode(generated, skip_special_tokens=True))
        references.extend(batch["texts"])
    model.train()
    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(references, hypotheses) if normalize_text(r)]
    wer = float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])) if pairs else float("nan")
    return {"eval_loss": float(np.mean(losses)) if losses else float("nan"), "eval_wer": wer}


# --------------------------------------------------------------------------- training


def train(cfg: Mapping[str, Any], train_records: SpeechRecords, eval_records: Optional[SpeechRecords],
          device_manager: Optional[DeviceManager] = None, num_workers: int = 0) -> Dict[str, Any]:  # fmt: skip
    """Run LoRA fine-tuning.

    Args:
        cfg: The ``lora`` section of the configuration (already merged with CLI flags).
        train_records: Training utterances.
        eval_records: Optional held-out utterances for periodic loss/WER.
        device_manager: Device selection; auto-detected when omitted.
        num_workers: DataLoader workers for audio decoding.

    Returns:
        Summary dict with the final metrics and the adapter directory.
    """
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_manager = device_manager or DeviceManager()
    device = device_manager.device
    plan = resolve_precision(str(cfg.get("precision", "auto")), device)
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = build_lora_model(cfg)
    model.to(device)
    model.train()

    collator = WhisperCollator(processor, int(cfg.get("max_label_tokens", 448)))
    batch_size = int(cfg.get("batch_size", 8))
    train_loader = DataLoader(
        train_records, batch_size=batch_size, shuffle=True, collate_fn=collator,
        num_workers=num_workers, drop_last=len(train_records) >= batch_size,
    )  # fmt: skip
    eval_loader = (
        DataLoader(eval_records, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)
        if eval_records is not None and len(eval_records) > 0
        else None
    )

    max_steps = int(cfg.get("max_steps", 1000))
    grad_accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    logging_steps = max(1, int(cfg.get("logging_steps", 10)))
    eval_steps = int(cfg.get("eval_steps", 0))
    save_steps = int(cfg.get("save_steps", 0))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(cfg.get("learning_rate", 1e-4)), weight_decay=float(cfg.get("weight_decay", 0.0))
    )
    scheduler = linear_schedule(optimizer, int(cfg.get("warmup_steps", 0)), max_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=plan.use_scaler)

    logger.info(
        "Training on %s | precision=%s | batch=%d x accum=%d (effective %d) | steps=%d | %d utterances",
        device_manager.describe(), plan.name, batch_size, grad_accum, batch_size * grad_accum,
        max_steps, len(train_records),
    )  # fmt: skip

    batches = cycle(train_loader)
    history: List[Dict[str, float]] = []
    running_loss = 0.0
    device_manager.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(grad_accum):
            batch = next(batches)
            features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device.type, dtype=plan.autocast_dtype, enabled=plan.autocast_dtype is not None):
                loss = model(input_features=features, labels=labels).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}; lower the learning rate or use --precision fp32."
                )
            # Scale so the accumulated gradient equals the mean over the effective batch.
            scaler.scale(loss / grad_accum).backward()
            step_loss += float(loss.detach()) / grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += step_loss

        if step % logging_steps == 0 or step == max_steps:
            device_manager.synchronize()
            elapsed = time.perf_counter() - started
            window = logging_steps if step % logging_steps == 0 else (step % logging_steps)
            logger.info(
                "step %5d/%d | loss %.4f | lr %.2e | %.2f s/step",
                step, max_steps, running_loss / window, scheduler.get_last_lr()[0], elapsed / step,
            )  # fmt: skip
            history.append({"step": step, "train_loss": running_loss / window})
            running_loss = 0.0

        if eval_loader is not None and eval_steps > 0 and (step % eval_steps == 0 or step == max_steps):
            metrics = evaluate(model, processor, eval_loader, device_manager, plan,
                               str(cfg.get("language", "en")), str(cfg.get("task", "transcribe")))  # fmt: skip
            logger.info("step %5d | eval_loss %.4f | eval_wer %.2f%%", step, metrics["eval_loss"], 100 * metrics["eval_wer"])
            history.append({"step": step, **metrics})

        if save_steps > 0 and step % save_steps == 0 and step != max_steps:
            model.save_pretrained(str(output_dir / f"step-{step}"))

    model.save_pretrained(str(output_dir))
    device_manager.synchronize()
    summary: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "base_model": str(cfg["base_model"]),
        "device": device_manager.describe(),
        "precision": plan.name,
        "steps": max_steps,
        "effective_batch_size": batch_size * grad_accum,
        "train_seconds": round(time.perf_counter() - started, 1),
        "history": history,
        "config": {k: v for k, v in cfg.items()},
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Saved LoRA adapter to %s", output_dir)
    return summary


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper with LoRA (q_proj/v_proj) on accented or domain-specific speech."
    )
    parser.add_argument("--config", default=None, help="YAML config overriding configs/default.yaml")
    parser.add_argument("--base-model", default=None, help="Whisper checkpoint (default: lora.base_model)")
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset id (default: lora.dataset)")
    parser.add_argument("--dataset-config", default=None, help="Optional dataset configuration name")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--manifest", default=None, help="Local JSONL/CSV with 'audio' and 'text' columns")
    parser.add_argument("--eval-manifest", default=None, help="Local evaluation manifest")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cuda | mps | cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 steps on synthetic audio with whisper-tiny to verify the setup")  # fmt: skip
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config value, e.g. --set lora.warmup_steps=100")  # fmt: skip
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config, parse_overrides(args.overrides))
    cfg: Dict[str, Any] = dict(config["lora"])
    flag_to_key = {
        "base_model": "base_model", "dataset": "dataset", "train_split": "train_split",
        "eval_split": "eval_split", "output_dir": "output_dir", "max_steps": "max_steps",
        "batch_size": "batch_size", "grad_accum_steps": "grad_accum_steps",
        "learning_rate": "learning_rate", "precision": "precision", "lora_r": "r", "lora_alpha": "alpha",
    }  # fmt: skip
    for flag, key in flag_to_key.items():
        value = getattr(args, flag)
        if value is not None:
            cfg[key] = value

    device_manager = DeviceManager(args.device or str(config.get("runtime", {}).get("device", "auto")))

    if args.smoke_test:
        cfg.update(
            base_model="openai/whisper-tiny", max_steps=2, batch_size=2, grad_accum_steps=2, warmup_steps=1,
            logging_steps=1, eval_steps=2, save_steps=0, r=4, alpha=8,
            output_dir=str(Path(cfg["output_dir"]).parent / "smoke-test-lora"),
        )  # fmt: skip
        train_records, eval_records = synthetic_records(8, seed=0), synthetic_records(2, seed=1)
    else:
        max_audio = float(cfg.get("max_audio_sec", 30.0))
        seed = int(cfg.get("seed", 42))
        train_records = load_records(
            manifest=args.manifest, dataset=cfg.get("dataset"), split=str(cfg.get("train_split", "train")),
            audio_column=str(cfg.get("audio_column", "audio")), text_column=str(cfg.get("text_column", "text")),
            max_samples=args.max_train_samples, max_audio_sec=max_audio, seed=seed,
            dataset_config=args.dataset_config,
        )  # fmt: skip
        eval_records = None
        eval_samples = int(cfg.get("eval_samples", 64))
        if args.eval_manifest or (args.manifest is None and cfg.get("eval_split")):
            eval_records = load_records(
                manifest=args.eval_manifest, dataset=cfg.get("dataset"), split=str(cfg.get("eval_split", "test")),
                audio_column=str(cfg.get("audio_column", "audio")), text_column=str(cfg.get("text_column", "text")),
                max_samples=eval_samples, max_audio_sec=max_audio, seed=seed, dataset_config=args.dataset_config,
            )  # fmt: skip

    summary = train(cfg, train_records, eval_records, device_manager, num_workers=args.num_workers)
    final = [h for h in summary["history"] if "eval_wer" in h]
    if final and not math.isnan(final[-1]["eval_wer"]):
        print(f"Final eval WER: {100 * final[-1]['eval_wer']:.2f}%")
    print(f"Adapter saved to: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
