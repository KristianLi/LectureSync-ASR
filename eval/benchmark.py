"""Offline evaluation: recognition accuracy and on-device inference cost.

Two sub-commands:

``accuracy``
    Word error rate (``jiwer``) and domain technical-term recall for the base Whisper
    model and, when ``--lora`` is given, the LoRA fine-tuned model. ``--with-prompt``
    adds glossary-prompted variants so the effect of terminology injection is measured
    separately from the effect of fine-tuning.

``latency``
    Per-clip latency (``avg_ms``, ``p50_ms``, ``p95_ms``), real-time factor (``RTF``) and
    peak resident memory (``psutil``) of the full ASR path on the current device.
    MPS and CUDA run asynchronously, so the device is synchronised
    (``torch.mps.synchronize()`` / ``torch.cuda.synchronize()``) immediately before the
    clock is started and again before it is stopped; without that the timer would only
    measure how long it takes to *enqueue* the work.

Examples::

    python eval/benchmark.py accuracy --dataset edinburghcstr/edacc --split test \\
        --max-samples 200 --lora checkpoints/whisper-small-lora-edacc --with-prompt
    python eval/benchmark.py accuracy --manifest data/dev.jsonl --terms data/terms.txt
    python eval/benchmark.py latency --audio samples/lecture_clip.wav --runs 20
    python eval/benchmark.py latency            # synthetic clip, fixed decode length
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from src.asr import WhisperASR  # noqa: E402
from src.train_lora import load_records  # noqa: E402
from src.utils import (  # noqa: E402
    SAMPLE_RATE,
    DeviceManager,
    cpu_brand,
    get_logger,
    load_audio,
    load_config,
    normalize_text,
    parse_overrides,
    peak_rss_mb,
)
from src.vad import SpeechSegment  # noqa: E402
from src.vocab import TermExtractor, load_glossary_file, term_recall  # noqa: E402

logger = get_logger("benchmark")

# --------------------------------------------------------------------------- helpers


class PeakMemoryMonitor:
    """Sample this process's RSS in a background thread and remember the maximum.

    ``resource.getrusage`` only reports the lifetime peak; sampling lets the benchmark
    attribute a peak to the measured section alone.
    """

    def __init__(self, interval_sec: float = 0.02) -> None:
        self.interval_sec = interval_sec
        self.peak_bytes = 0
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "PeakMemoryMonitor":
        self.peak_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024.0 * 1024.0)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in ``[0, 100]``)."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else float("nan")


def machine_info(device_manager: DeviceManager) -> Dict[str, Any]:
    """Describe the host so results from different machines can be compared."""
    return {
        "machine": platform.machine(),
        "cpu": cpu_brand(),
        "os": f"{platform.system()} {platform.release()}",
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device_manager.device.type,
        "dtype": str(device_manager.dtype).replace("torch.", ""),
    }


def save_results(payload: Dict[str, Any], results_dir: Path, stem: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def fmt(value: Optional[float], pattern: str = "{:.2f}") -> str:
    """Format a number for a Markdown cell, with ``n/a`` for missing values."""
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else pattern.format(value)


# --------------------------------------------------------------------------- accuracy


def evaluate_system(
    name: str,
    asr: WhisperASR,
    waveforms: Sequence[np.ndarray],
    references: Sequence[str],
    terms: Sequence[str],
    prompt_terms: Optional[Sequence[str]],
    device_manager: DeviceManager,
) -> Dict[str, Any]:
    """Transcribe ``waveforms`` with one system and score it against ``references``."""
    import jiwer

    asr.set_glossary(prompt_terms)
    device_manager.synchronize()
    started = time.perf_counter()
    hypotheses = asr.transcribe_waveforms(waveforms)
    device_manager.synchronize()
    elapsed = time.perf_counter() - started

    refs_norm = [normalize_text(r) for r in references]
    hyps_norm = [normalize_text(h) for h in hypotheses]
    keep = [i for i, ref in enumerate(refs_norm) if ref]
    wer = float(jiwer.wer([refs_norm[i] for i in keep], [hyps_norm[i] for i in keep])) if keep else float("nan")
    recall = term_recall(references, hypotheses, terms)
    audio_sec = sum(len(w) for w in waveforms) / SAMPLE_RATE
    logger.info(
        "%-22s WER %.2f%% | term recall %s (%d/%d) | RTF %.3f",
        name, 100 * wer, fmt(100 * recall["recall"], "{:.1f}%"), recall["matched"], recall["total"],
        elapsed / max(audio_sec, 1e-9),
    )  # fmt: skip
    return {
        "system": name,
        "wer": wer,
        "term_recall": recall["recall"],
        "term_matched": recall["matched"],
        "term_total": recall["total"],
        "utterances": len(keep),
        "audio_sec": audio_sec,
        "rtf": elapsed / max(audio_sec, 1e-9),
        "hypotheses": hypotheses,
    }


def run_accuracy(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """WER and technical-term recall: base Whisper vs. LoRA (and prompted variants)."""
    device_manager = DeviceManager(args.device or str(config["runtime"].get("device", "auto")),
                                   str(config["runtime"].get("dtype", "auto")))  # fmt: skip
    lora_cfg = config.get("lora", {})
    records = load_records(
        manifest=args.manifest,
        dataset=args.dataset or (None if args.manifest else lora_cfg.get("dataset")),
        split=args.split or str(lora_cfg.get("eval_split", "test")),
        audio_column=str(lora_cfg.get("audio_column", "audio")),
        text_column=str(lora_cfg.get("text_column", "text")),
        max_samples=args.max_samples,
        seed=int(lora_cfg.get("seed", 42)),
    )
    logger.info("Decoding %d utterances", len(records))
    pairs = [records[i] for i in range(len(records))]
    waveforms = [wave for wave, _ in pairs]
    references = [text for _, text in pairs]

    custom_terms = load_glossary_file(args.terms)
    terms = custom_terms or TermExtractor().benchmark_terms()
    # Only prompt with terms that actually occur in this evaluation set, as slides would.
    present = term_recall(references, references, terms)["per_term"]
    prompt_terms = sorted(present, key=lambda t: -present[t][1])

    asr_cfg = dict(config["asr"], sub_segment_timestamps=False)
    if args.model:
        asr_cfg["model_name"] = args.model
    if args.batch_size:
        asr_cfg["batch_size"] = args.batch_size

    systems: List[Dict[str, Any]] = []
    variants: List[Any] = [("base", None)] + ([("lora", args.lora)] if args.lora else [])
    for label, checkpoint in variants:
        asr = WhisperASR(asr_cfg, device_manager, lora_checkpoint=checkpoint or "").load()
        name = "Base Whisper" if label == "base" else "LoRA Whisper"
        systems.append(evaluate_system(name, asr, waveforms, references, terms, None, device_manager))
        if args.with_prompt and prompt_terms:
            systems.append(
                evaluate_system(f"{name} + glossary", asr, waveforms, references, terms, prompt_terms, device_manager)
            )
        del asr
        device_manager.empty_cache()

    source = args.manifest or f"{args.dataset or lora_cfg.get('dataset')}[{args.split or lora_cfg.get('eval_split')}]"
    print(f"\n### Accuracy on {source} ({len(waveforms)} utterances, model {asr_cfg['model_name']})\n")
    print("| System | WER (%) | Term recall (%) | Terms matched | RTF |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in systems:
        print(
            f"| {row['system']} | {fmt(100 * row['wer'])} | {fmt(100 * row['term_recall'], '{:.1f}')} "
            f"| {row['term_matched']}/{row['term_total']} | {fmt(row['rtf'], '{:.3f}')} |"
        )
    payload = {
        "kind": "accuracy",
        "source": str(source),
        "model": asr_cfg["model_name"],
        "lora": args.lora,
        "machine": machine_info(device_manager),
        "systems": systems,
    }
    path = save_results(payload, Path(config["benchmark"]["results_dir"]), f"accuracy_{Path(str(source)).stem}")
    print(f"\nSaved: {path}")
    return payload


# --------------------------------------------------------------------------- latency


def synthetic_clip(seconds: float, seed: int = 0) -> np.ndarray:
    """Speech-shaped test signal: harmonic stack with syllable-rate amplitude modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 120.0 + 20.0 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    signal = 0.2 * voiced * envelope + 0.005 * rng.standard_normal(t.shape)
    return signal.astype(np.float32)


def run_latency(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Latency, real-time factor and peak memory of single-clip ASR on this device."""
    bench_cfg = config["benchmark"]
    warmup = args.warmup if args.warmup is not None else int(bench_cfg.get("warmup_runs", 3))
    runs = args.runs if args.runs is not None else int(bench_cfg.get("measured_runs", 20))
    clip_sec = float(args.clip_sec or bench_cfg.get("clip_sec", 30.0))

    rss_before_load = psutil.Process().memory_info().rss / 1024**2
    device_manager = DeviceManager(args.device or str(config["runtime"].get("device", "auto")),
                                   args.dtype or str(config["runtime"].get("dtype", "auto")))  # fmt: skip
    asr_cfg = dict(config["asr"], batch_size=1)
    if args.model:
        asr_cfg["model_name"] = args.model

    device_manager.synchronize()
    tick = time.perf_counter()
    asr = WhisperASR(asr_cfg, device_manager, lora_checkpoint=args.lora or "").load()
    device_manager.synchronize()
    load_sec = time.perf_counter() - tick

    if args.audio:
        audio = load_audio(args.audio)[: int(clip_sec * SAMPLE_RATE)]
        workload = f"real audio ({Path(args.audio).name})"
    else:
        audio = synthetic_clip(clip_sec)
        tokens = int(args.decode_tokens or bench_cfg.get("decode_tokens", 64))
        # A synthetic clip has no words, so pin the decode length to make runs comparable.
        asr.generation_overrides = {"min_new_tokens": tokens, "max_new_tokens": tokens}
        workload = f"synthetic clip, fixed {tokens}-token decode"
    audio_sec = len(audio) / SAMPLE_RATE
    segment = SpeechSegment(0.0, audio_sec, torch.from_numpy(audio))

    logger.info("Benchmarking %s on %s: %s, %d warm-up + %d measured runs",
                asr.model_name, device_manager.describe(), workload, warmup, runs)  # fmt: skip
    logging.getLogger("lecturesync.src.asr").setLevel(logging.WARNING)  # keep the timed loop quiet
    for _ in range(warmup):
        asr.transcribe([segment])
    device_manager.synchronize()

    # Encoder-only latency: the fixed cost every 30 s window pays regardless of text length.
    features, _ = asr.extract_features([audio])
    encoder = asr.model.get_encoder()
    encoder_ms: List[float] = []
    with torch.no_grad():
        for _ in range(max(3, runs // 2)):
            device_manager.synchronize()
            tick = time.perf_counter()
            encoder(features)
            device_manager.synchronize()
            encoder_ms.append(1000.0 * (time.perf_counter() - tick))

    latencies_ms: List[float] = []
    n_tokens = 0
    with PeakMemoryMonitor() as monitor:
        for _ in range(runs):
            device_manager.synchronize()  # nothing queued from the previous run
            tick = time.perf_counter()
            output = asr.transcribe([segment])
            device_manager.synchronize()  # wait for the GPU before reading the clock
            latencies_ms.append(1000.0 * (time.perf_counter() - tick))
            n_tokens = sum(len(asr.processor.tokenizer(seg["text"], add_special_tokens=False).input_ids) for seg in output)

    avg_ms = float(np.mean(latencies_ms))
    result = {
        "kind": "latency",
        "model": asr.model_name,
        "lora": args.lora,
        "workload": workload,
        "audio_sec": audio_sec,
        "runs": runs,
        "warmup_runs": warmup,
        "load_sec": load_sec,
        "avg_ms": avg_ms,
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "encoder_avg_ms": float(np.mean(encoder_ms)),
        "rtf": (avg_ms / 1000.0) / audio_sec,
        "decoded_tokens": n_tokens,
        "peak_rss_mb": monitor.peak_mb,
        "lifetime_peak_rss_mb": peak_rss_mb(),
        "rss_before_load_mb": rss_before_load,
        "accelerator_memory": device_manager.memory_stats(),
        "machine": machine_info(device_manager),
        "latencies_ms": latencies_ms,
    }

    info = result["machine"]
    print(f"\n### Latency: {asr.model_name} on {info['cpu']} ({info['device']}, {info['dtype']})\n")
    print(f"Workload: {workload}, {audio_sec:.1f} s clip, {runs} runs after {warmup} warm-up runs\n")
    print("| Device | Backend | dtype | avg_ms | p50_ms | p95_ms | RTF | Encoder ms | Peak RSS (MB) | Load (s) |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {info['cpu']} ({info['ram_gb']:.0f} GB) | {info['device']} | {info['dtype']} "
        f"| {avg_ms:.1f} | {result['p50_ms']:.1f} | {result['p95_ms']:.1f} | {result['rtf']:.4f} "
        f"| {result['encoder_avg_ms']:.1f} | {result['peak_rss_mb']:.0f} | {load_sec:.1f} |"
    )
    for key, value in result["accelerator_memory"].items():
        print(f"\n{key}: {value:.0f} MB", end="")
    kind = "real" if args.audio else "synthetic"
    stem = f"latency_{asr.model_name.split('/')[-1]}_{info['device']}_{info['dtype']}_{kind}"
    path = save_results(result, Path(bench_cfg["results_dir"]), stem)
    print(f"\n\nSaved: {path}")
    return result


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LectureSync-ASR accuracy and latency benchmarks.")
    parser.add_argument("--config", default=None, help="YAML config overriding configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    sub = parser.add_subparsers(dest="command", required=True)

    acc = sub.add_parser("accuracy", help="WER and technical-term recall (base vs. LoRA)")
    acc.add_argument("--manifest", default=None, help="JSONL/CSV with 'audio' and 'text' columns")
    acc.add_argument("--dataset", default=None, help="HuggingFace dataset id (default: lora.dataset)")
    acc.add_argument("--split", default=None, help="Dataset split (default: lora.eval_split)")
    acc.add_argument("--max-samples", type=int, default=200)
    acc.add_argument("--lora", default=None, help="LoRA adapter directory to compare against the base model")
    acc.add_argument("--terms", default=None, help="Glossary file for term recall (default: built-in CS/AI lexicon)")
    acc.add_argument("--with-prompt", action="store_true", help="Also evaluate glossary-prompted decoding")
    acc.add_argument("--model", default=None, help="Base Whisper checkpoint (default: asr.model_name)")
    acc.add_argument("--batch-size", type=int, default=None)
    acc.add_argument("--device", default=None)

    lat = sub.add_parser("latency", help="avg/p95 latency, RTF and peak RSS on this device")
    lat.add_argument("--audio", default=None, help="Real clip to decode (default: synthetic, fixed decode length)")
    lat.add_argument("--clip-sec", type=float, default=None)
    lat.add_argument("--runs", type=int, default=None)
    lat.add_argument("--warmup", type=int, default=None)
    lat.add_argument("--decode-tokens", type=int, default=None)
    lat.add_argument("--model", default=None)
    lat.add_argument("--lora", default=None)
    lat.add_argument("--device", default=None)
    lat.add_argument("--dtype", default=None, help="auto | float32 | float16")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config, parse_overrides(args.overrides))
    if args.command == "accuracy":
        run_accuracy(args, config)
    else:
        run_latency(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
