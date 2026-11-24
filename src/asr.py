"""Whisper speech recognition with optional LoRA adapters and glossary prompting.

Pipeline for one batch of VAD segments:

1. log-Mel features are computed on the CPU and moved to the model device/dtype,
2. the encoder runs once; its output is shared by decoding and confidence scoring,
3. ``generate`` decodes with the language/task prefix (the modern replacement for
   ``forced_decoder_ids``) and, when a glossary is set, with ``prompt_ids`` so slide
   terminology is in the decoder's left context,
4. timestamp tokens split each chunk into phrase-level segments,
5. a single teacher-forced decoder pass yields per-token log-probabilities, which are
   averaged into a ``confidence`` in ``(0, 1]`` per segment.

Step 5 deliberately relies only on the plain ``forward`` API instead of the
``output_scores`` plumbing of ``generate``, whose return format differs between
transformers releases and between timestamp / non-timestamp decoding.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch

from .utils import SAMPLE_RATE, DeviceManager, PathLike, get_logger
from .vad import SpeechSegment

logger = get_logger(__name__)

_TIMESTAMP_STEP_SEC = 0.02  # Whisper timestamp tokens advance in 20 ms increments
_WHITESPACE_RE = re.compile(r"\s+")


class TranscriptSegment(TypedDict):
    """Structured transcript unit returned by :meth:`WhisperASR.transcribe`."""

    id: int
    start: float
    end: float
    text: str
    confidence: float


ProgressCallback = Callable[[int, int], None]


class WhisperASR:
    """HuggingFace Whisper wrapper tuned for lecture audio.

    Args:
        config: The ``asr`` section of the YAML configuration.
        device_manager: Shared :class:`DeviceManager`; created on demand when omitted.
        model_name: Overrides ``config["model_name"]``.
        lora_checkpoint: Overrides ``config["lora_checkpoint"]``; a directory written by
            ``src/train_lora.py`` (contains ``adapter_config.json``).

    Example:
        >>> asr = WhisperASR(config["asr"], device_manager).load()
        >>> asr.set_glossary(["backpropagation", "homography", "eigenvalue"])
        >>> asr.transcribe(vad_segments)[0]
        {'id': 0, 'start': 0.0, 'end': 6.8, 'text': '...', 'confidence': 0.93}
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        device_manager: Optional[DeviceManager] = None,
        model_name: Optional[str] = None,
        lora_checkpoint: Optional[PathLike] = None,
    ) -> None:
        cfg = dict(config or {})
        self.model_name: str = str(model_name or cfg.get("model_name", "openai/whisper-small"))
        checkpoint = lora_checkpoint if lora_checkpoint is not None else cfg.get("lora_checkpoint")
        self.lora_checkpoint: Optional[Path] = Path(checkpoint) if checkpoint else None
        self.merge_lora: bool = bool(cfg.get("merge_lora", True))
        self.language: str = str(cfg.get("language", "en"))
        self.task: str = str(cfg.get("task", "transcribe"))
        self.batch_size: int = int(cfg.get("batch_size", 8))
        self.num_beams: int = int(cfg.get("num_beams", 1))
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 224))
        self.sub_segment_timestamps: bool = bool(cfg.get("sub_segment_timestamps", True))
        self.max_subsegment_sec: float = float(cfg.get("max_subsegment_sec", 12.0))
        self.min_confidence: float = float(cfg.get("min_confidence", 0.0))
        self.prompt_prefix: str = str(cfg.get("prompt_prefix", "Key terms:"))
        self.prompt_max_terms: int = int(cfg.get("prompt_max_terms", 40))
        self.prompt_max_tokens: int = int(cfg.get("prompt_max_tokens", 120))

        self.device_manager: DeviceManager = device_manager or DeviceManager()
        self.processor: Any = None
        self.model: Any = None
        self.glossary: List[str] = []
        self.prompt_text: str = ""
        self._prompt_ids: Optional[torch.Tensor] = None
        self._prefix_ids: List[int] = []
        self._timestamp_begin: int = 0
        self._eos_id: int = 0
        self._sot_id: int = 0
        self._multilingual: bool = True

    # ------------------------------------------------------------------ loading

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> torch.device:
        return self.device_manager.device

    @property
    def dtype(self) -> torch.dtype:
        return self.device_manager.dtype

    def load(self) -> "WhisperASR":
        """Load processor, base weights and (optionally) the LoRA adapter. Idempotent."""
        if self.is_loaded:
            return self
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading %s on %s", self.model_name, self.device_manager.describe())
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        if self.lora_checkpoint is not None:
            model = self._attach_lora(model, self.lora_checkpoint)

        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self._configure_special_tokens()
        return self

    def _attach_lora(self, model: torch.nn.Module, checkpoint: Path) -> torch.nn.Module:
        """Wrap ``model`` with a PEFT LoRA adapter and optionally merge it into the weights."""
        from peft import PeftModel

        if not (checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No adapter_config.json in LoRA checkpoint: {checkpoint}")
        adapter_cfg = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        trained_on = adapter_cfg.get("base_model_name_or_path")
        if trained_on and trained_on != self.model_name:
            logger.warning(
                "LoRA adapter was trained on %s but is being applied to %s.", trained_on, self.model_name
            )
        peft_model = PeftModel.from_pretrained(model, str(checkpoint))
        if self.merge_lora:
            logger.info("Merging LoRA adapter from %s into the base weights", checkpoint)
            return peft_model.merge_and_unload()
        logger.info("Loaded LoRA adapter from %s (kept as a separate adapter)", checkpoint)
        # The LoRA layers are injected in place, so the unwrapped model is adapter-aware.
        return peft_model.get_base_model()

    def _configure_special_tokens(self) -> None:
        """Cache the token ids needed to build decoder prefixes and parse timestamps."""
        tokenizer = self.processor.tokenizer
        generation_config = self.model.generation_config
        self._multilingual = bool(getattr(generation_config, "is_multilingual", True))
        self._sot_id = int(tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
        self._eos_id = int(tokenizer.eos_token_id)
        self._no_timestamps_id = int(tokenizer.convert_tokens_to_ids("<|notimestamps|>"))
        self._timestamp_begin = self._no_timestamps_id + 1
        prefix = [self._sot_id]
        if self._multilingual:
            prefix.append(int(tokenizer.convert_tokens_to_ids(f"<|{self.language}|>")))
            prefix.append(int(tokenizer.convert_tokens_to_ids(f"<|{self.task}|>")))
        if not self.sub_segment_timestamps:
            prefix.append(self._no_timestamps_id)
        self._prefix_ids = prefix

    # ------------------------------------------------------------------ glossary prompt

    def set_glossary(self, terms: Optional[Sequence[str]]) -> str:
        """Bias decoding towards domain terminology via Whisper ``prompt_ids``.

        The terms are written into a short natural-language sentence that Whisper sees as
        "previous context" (``<|startofprev|> ...``). Seeing ``homography`` or
        ``eigenvalue`` spelled out there makes the decoder far more likely to emit the
        same spelling instead of a phonetically similar common word.

        Args:
            terms: Terms ordered by importance; ``None`` or empty clears the prompt.

        Returns:
            The prompt text actually used (truncated to the configured token budget).
        """
        self.load()
        unique: List[str] = []
        seen = set()
        for term in terms or []:
            cleaned = _WHITESPACE_RE.sub(" ", str(term)).strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                unique.append(cleaned)
        if not unique:
            self.glossary, self.prompt_text, self._prompt_ids = [], "", None
            return ""

        tokenizer = self.processor.tokenizer
        chosen: List[str] = []
        text = ""
        for term in unique[: self.prompt_max_terms]:
            candidate = f"{self.prompt_prefix} {', '.join(chosen + [term])}."
            n_tokens = len(tokenizer(" " + candidate, add_special_tokens=False).input_ids)
            if n_tokens > self.prompt_max_tokens:
                break
            chosen.append(term)
            text = candidate
        if not chosen:
            self.glossary, self.prompt_text, self._prompt_ids = [], "", None
            return ""

        self.glossary = chosen
        self.prompt_text = text
        self._prompt_ids = self.processor.get_prompt_ids(text, return_tensors="pt").to(self.device)
        logger.info("Glossary prompt set: %d terms, %d tokens", len(chosen), int(self._prompt_ids.numel()))
        return text

    # ------------------------------------------------------------------ features

    def extract_features(self, waveforms: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute padded log-Mel features on the model's device and dtype.

        Returns:
            ``(input_features, attention_mask)``; both tensors live on ``self.device``.
        """
        self.load()
        batch = self.processor.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = batch.input_features.to(device=self.device, dtype=self.dtype)
        mask = batch.attention_mask.to(device=self.device)
        return features, mask

    # ------------------------------------------------------------------ decoding

    def _generation_kwargs(self) -> Dict[str, Any]:
        prompt_len = int(self._prompt_ids.numel()) if self._prompt_ids is not None else 0
        max_positions = int(self.model.config.max_target_positions)
        budget = max_positions - prompt_len - len(self._prefix_ids) - 4
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max(16, min(self.max_new_tokens, budget)),
            "num_beams": self.num_beams,
            "return_timestamps": self.sub_segment_timestamps,
        }
        if self._multilingual:
            kwargs["language"] = self.language
            kwargs["task"] = self.task
        if self._prompt_ids is not None:
            kwargs["prompt_ids"] = self._prompt_ids
        return kwargs

    def _content_tokens(self, row: Sequence[int]) -> List[int]:
        """Strip prompt, decoder prefix, padding and EOS from one generated row.

        Depending on the transformers version ``generate`` returns either the full
        decoder sequence (prompt + prefix + text) or only the newly generated tokens;
        both layouts are handled.
        """
        tokens = list(row)
        if self._sot_id in tokens:
            tokens = tokens[len(tokens) - 1 - tokens[::-1].index(self._sot_id) + 1 :]
        start = 0
        while (
            start < len(tokens)
            and self._eos_id < tokens[start] < self._timestamp_begin  # language / task / notimestamps
        ):
            start += 1
        tokens = tokens[start:]
        if self._eos_id in tokens:
            tokens = tokens[: tokens.index(self._eos_id)]
        return tokens

    @torch.no_grad()
    def _score_tokens(self, encoder_outputs: Any, rows: Sequence[Sequence[int]]) -> List[List[float]]:
        """Teacher-forced log-probability of every generated token.

        Args:
            encoder_outputs: Encoder states for the batch (reused from decoding).
            rows: Content tokens per batch item, as returned by :meth:`_content_tokens`.

        Returns:
            One list of natural-log probabilities per batch item, aligned with ``rows``.
        """
        prompt = self._prompt_ids.tolist() if self._prompt_ids is not None else []
        prefix = prompt + self._prefix_ids
        n_prefix = len(prefix)
        longest = max((len(r) for r in rows), default=0)
        if longest == 0:
            return [[] for _ in rows]

        full = torch.full((len(rows), n_prefix + longest), self._eos_id, dtype=torch.long)
        for index, row in enumerate(rows):
            full[index, :n_prefix] = torch.tensor(prefix, dtype=torch.long)
            if row:
                full[index, n_prefix : n_prefix + len(row)] = torch.tensor(list(row), dtype=torch.long)
        full = full.to(self.device)

        logits = self.model(encoder_outputs=encoder_outputs, decoder_input_ids=full[:, :-1]).logits
        logits = logits[:, n_prefix - 1 :, :].float()  # position j predicts content token j
        targets = full[:, n_prefix:]
        chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_probs = (chosen - torch.logsumexp(logits, dim=-1)).cpu()
        return [log_probs[index, : len(row)].tolist() for index, row in enumerate(rows)]

    def _split_on_timestamps(
        self, tokens: Sequence[int], log_probs: Sequence[float], duration: float
    ) -> List[Tuple[float, float, List[int], List[float]]]:
        """Group text tokens between Whisper timestamp tokens.

        Returns:
            Tuples ``(start, end, token_ids, token_log_probs)`` with times relative to the
            chunk start. Text without a closing timestamp ends at ``duration``.
        """
        pieces: List[Tuple[float, float, List[int], List[float]]] = []
        current_tokens: List[int] = []
        current_lps: List[float] = []
        current_start = 0.0
        for token, log_prob in zip(tokens, log_probs):
            if token >= self._timestamp_begin:
                stamp = min((token - self._timestamp_begin) * _TIMESTAMP_STEP_SEC, duration)
                if current_tokens:
                    end = max(stamp, current_start)
                    pieces.append((current_start, end, current_tokens, current_lps))
                    current_tokens, current_lps = [], []
                current_start = stamp
            elif token < self._eos_id:
                current_tokens.append(token)
                current_lps.append(log_prob)
        if current_tokens:
            pieces.append((min(current_start, duration), duration, current_tokens, current_lps))
        # Guard against non-monotonic or zero-length stamps from a confused decoder.
        cleaned: List[Tuple[float, float, List[int], List[float]]] = []
        cursor = 0.0
        for start, end, toks, lps in pieces:
            start = max(start, cursor)
            end = min(max(end, start), duration)
            if end - start < 0.05:
                end = min(duration, start + 0.05)
            cleaned.append((start, end, toks, lps))
            cursor = end
        return cleaned

    def _split_long_piece(
        self, piece: Tuple[float, float, List[int], List[float]]
    ) -> List[Tuple[float, float, List[int], List[float]]]:
        """Break an over-long phrase at sentence ends.

        Whisper sometimes emits a single pair of timestamps around 20+ seconds of speech,
        which is too coarse for replaying one sentence. Such a piece is cut after tokens
        ending in ``.``, ``?`` or ``!``; the cut times are interpolated from the character
        position, which is accurate to about a second at a steady speaking rate.
        """
        start, end, tokens, log_probs = piece
        if self.max_subsegment_sec <= 0 or end - start <= self.max_subsegment_sec or len(tokens) < 2:
            return [piece]
        tokenizer = self.processor.tokenizer
        token_texts = [tokenizer.decode([token]) for token in tokens]
        total_chars = max(1, sum(len(text) for text in token_texts))
        target = 0.6 * self.max_subsegment_sec
        pieces: List[Tuple[float, float, List[int], List[float]]] = []
        first, piece_start, chars_seen = 0, start, 0
        for index, text in enumerate(token_texts[:-1]):
            chars_seen += len(text)
            now = start + (end - start) * chars_seen / total_chars
            if text.rstrip().endswith((".", "?", "!")) and now - piece_start >= target:
                pieces.append((piece_start, now, tokens[first : index + 1], log_probs[first : index + 1]))
                first, piece_start = index + 1, now
        pieces.append((piece_start, end, tokens[first:], log_probs[first:]))
        return pieces

    @staticmethod
    def _confidence(log_probs: Sequence[float]) -> float:
        """Geometric-mean token probability, clipped to ``[0, 1]``."""
        if not log_probs:
            return 0.0
        return float(min(1.0, max(0.0, math.exp(sum(log_probs) / len(log_probs)))))

    @torch.no_grad()
    def _decode_batch(self, segments: Sequence[SpeechSegment]) -> List[List[Dict[str, Any]]]:
        """Decode one batch; returns the list of sub-segments for every input segment."""
        waveforms = [seg.audio_tensor.detach().cpu().numpy() for seg in segments]
        features, mask = self.extract_features(waveforms)
        encoder_outputs = self.model.get_encoder()(features)
        kwargs = self._generation_kwargs()
        try:
            generated = self.model.generate(
                features, attention_mask=mask, encoder_outputs=encoder_outputs, **kwargs
            )
        except (TypeError, ValueError) as exc:
            # Older transformers releases reject precomputed encoder states in Whisper.generate.
            logger.debug("generate(encoder_outputs=...) unsupported (%s); re-encoding.", exc)
            generated = self.model.generate(features, attention_mask=mask, **kwargs)
        sequences = generated["sequences"] if isinstance(generated, Mapping) else generated
        rows = [self._content_tokens(row) for row in sequences.cpu().tolist()]
        log_probs = self._score_tokens(encoder_outputs, rows)

        tokenizer = self.processor.tokenizer
        results: List[List[Dict[str, Any]]] = []
        for seg, tokens, lps in zip(segments, rows, log_probs):
            duration = float(seg.end_sec - seg.start_sec)
            if self.sub_segment_timestamps:
                pieces = [
                    part
                    for piece in self._split_on_timestamps(tokens, lps, duration)
                    for part in self._split_long_piece(piece)
                ]
            else:
                text_pairs = [(t, lp) for t, lp in zip(tokens, lps) if t < self._eos_id]
                pieces = (
                    [(0.0, duration, [t for t, _ in text_pairs], [lp for _, lp in text_pairs])]
                    if text_pairs
                    else []
                )
            items: List[Dict[str, Any]] = []
            for start, end, toks, tok_lps in pieces:
                text = _WHITESPACE_RE.sub(" ", tokenizer.decode(toks, skip_special_tokens=True)).strip()
                if not text:
                    continue
                confidence = self._confidence(tok_lps)
                if confidence < self.min_confidence:
                    continue
                items.append(
                    {
                        "start": round(seg.start_sec + start, 3),
                        "end": round(seg.start_sec + end, 3),
                        "text": text,
                        "confidence": round(confidence, 4),
                    }
                )
            results.append(items)
        return results

    def _decode_batch_safe(self, segments: Sequence[SpeechSegment]) -> List[List[Dict[str, Any]]]:
        """Run :meth:`_decode_batch`, retrying in float32 if half precision fails on MPS."""
        try:
            return self._decode_batch(segments)
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            if self.dtype != torch.float16 or not self.device_manager.is_mps:
                raise
            logger.warning("float16 decoding failed on MPS (%s); retrying in float32.", exc)
            self.device_manager.fallback_to_float32()
            self.model = self.model.to(dtype=torch.float32)
            return self._decode_batch(segments)

    # ------------------------------------------------------------------ public API

    def transcribe(
        self,
        segments: Sequence[SpeechSegment],
        glossary: Optional[Sequence[str]] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe VAD segments into time-stamped transcript segments.

        Args:
            segments: Output of :meth:`src.vad.VoiceActivityDetector.segment`; each item
                must be at most 30 s long.
            glossary: Optional terms; when given, replaces the current glossary prompt.
            progress: Optional ``callback(done, total)`` invoked after every batch.

        Returns:
            ``[{"id", "start", "end", "text", "confidence"}, ...]`` in chronological order.
            ``start``/``end`` are seconds in the original recording.
        """
        self.load()
        if glossary is not None:
            self.set_glossary(glossary)
        transcript: List[TranscriptSegment] = []
        total = len(segments)
        for offset in range(0, total, self.batch_size):
            batch = list(segments[offset : offset + self.batch_size])
            for items in self._decode_batch_safe(batch):
                for item in items:
                    transcript.append(
                        TranscriptSegment(
                            id=len(transcript),
                            start=item["start"],
                            end=item["end"],
                            text=item["text"],
                            confidence=item["confidence"],
                        )
                    )
            if progress is not None:
                progress(min(offset + self.batch_size, total), total)
        logger.info("ASR produced %d transcript segments from %d speech chunks", len(transcript), total)
        return transcript

    def transcribe_waveforms(self, waveforms: Sequence[np.ndarray]) -> List[str]:
        """Transcribe independent clips (each <= 30 s) and return one string per clip.

        Used by the benchmark, where every dataset utterance is scored on its own.
        """
        self.load()
        texts: List[str] = []
        for offset in range(0, len(waveforms), self.batch_size):
            chunk = waveforms[offset : offset + self.batch_size]
            segments = [
                SpeechSegment(0.0, len(w) / SAMPLE_RATE, torch.as_tensor(np.asarray(w), dtype=torch.float32))
                for w in chunk
            ]
            for items in self._decode_batch_safe(segments):
                texts.append(" ".join(item["text"] for item in items).strip())
        return texts
