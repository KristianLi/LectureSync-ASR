"""Voice activity detection: cut a long recording into Whisper-sized speech segments.

The detector wraps the Silero VAD TorchScript model loaded through ``torch.hub``. When
the hub is unreachable it degrades gracefully:

1. a previously downloaded copy in the local ``torch.hub`` cache,
2. ``torch.hub.load("snakers4/silero-vad", "silero_vad")`` over the network,
3. the ``silero-vad`` pip wheel, which bundles the same weights and needs no network,
4. a dependency-free short-time-energy detector.

Whatever backend produces the per-frame speech probabilities, the same post-processing
turns them into segments: hysteresis thresholding, merging of pauses shorter than
``min_silence_sec`` (0.8 s), and splitting of regions longer than ``max_segment_sec``
(30 s) at their quietest frame.

Silero is a ~2 MB recurrent model that runs faster than real time on one CPU core, so it
is deliberately kept on the CPU. Segment tensors are therefore always CPU tensors; the
ASR stage moves features to MPS/CUDA itself, which rules out cross-device tensor mixes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .utils import SAMPLE_RATE, PathLike, get_logger, load_audio

logger = get_logger(__name__)

_FRAME_SAMPLES = 512  # Silero v5+ consumes exactly 512 samples (32 ms) per step at 16 kHz
_SILERO_CONTEXT_OK_RATES = (16_000,)


@dataclass
class SpeechSegment:
    """One contiguous stretch of speech.

    Attributes:
        start_sec: Segment start in seconds from the beginning of the recording.
        end_sec: Segment end in seconds.
        audio_tensor: 1-D ``float32`` CPU tensor holding the 16 kHz samples.
    """

    start_sec: float
    end_sec: float
    audio_tensor: torch.Tensor

    @property
    def duration(self) -> float:
        """Segment length in seconds."""
        return self.end_sec - self.start_sec


class VoiceActivityDetector:
    """Silero-VAD-based segmenter with offline fallbacks.

    Args:
        config: The ``vad`` section of the YAML configuration.
        sample_rate: Sampling rate of the waveforms passed to :meth:`segment`.

    Example:
        >>> vad = VoiceActivityDetector(config["vad"])
        >>> segments = vad.segment_file("lecture.mp4")
        >>> segments[0].start_sec, segments[0].end_sec, segments[0].audio_tensor.shape
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, sample_rate: int = SAMPLE_RATE) -> None:
        cfg = dict(config or {})
        if sample_rate not in _SILERO_CONTEXT_OK_RATES:
            raise ValueError(f"VoiceActivityDetector expects 16 kHz audio, got {sample_rate} Hz")
        self.sample_rate = sample_rate
        self.requested_backend: str = str(cfg.get("backend", "auto"))
        self.hub_repo: str = str(cfg.get("hub_repo", "snakers4/silero-vad"))
        self.threshold: float = float(cfg.get("threshold", 0.5))
        self.neg_threshold: float = float(cfg.get("neg_threshold", max(self.threshold - 0.15, 0.01)))
        self.min_speech_sec: float = float(cfg.get("min_speech_sec", 0.25))
        self.min_silence_sec: float = float(cfg.get("min_silence_sec", 0.8))
        self.max_segment_sec: float = float(cfg.get("max_segment_sec", 30.0))
        self.speech_pad_sec: float = float(cfg.get("speech_pad_sec", 0.1))
        self.frame_sec: float = _FRAME_SAMPLES / float(sample_rate)
        self.backend: str = "unloaded"
        self._model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------ model loading

    def load(self) -> "VoiceActivityDetector":
        """Resolve the backend (idempotent). Called lazily by :meth:`segment`."""
        if self.backend != "unloaded":
            return self
        order: Sequence[str]
        if self.requested_backend == "auto":
            order = ("silero_hub_cache", "silero_hub", "silero_pip", "energy")
        elif self.requested_backend == "silero_hub":
            order = ("silero_hub_cache", "silero_hub", "silero_pip", "energy")
        elif self.requested_backend == "silero_pip":
            order = ("silero_pip", "energy")
        else:
            order = ("energy",)

        loaders = {
            "silero_hub_cache": self._load_from_hub_cache,
            "silero_hub": self._load_from_hub,
            "silero_pip": self._load_from_pip,
        }
        for name in order:
            if name == "energy":
                break
            try:
                model = loaders[name]()
            except Exception as exc:  # noqa: BLE001 - offline / missing package: try the next source
                logger.debug("VAD backend %s unavailable: %s", name, exc)
                continue
            if model is not None:
                self._model = model.to("cpu").eval()
                self.backend = name
                logger.info("Silero VAD loaded via %s", name)
                return self
        if self.requested_backend not in {"energy"}:
            logger.warning("Silero VAD could not be loaded; using the energy-based fallback detector.")
        self.backend = "energy"
        return self

    def _load_from_hub_cache(self) -> Optional[torch.nn.Module]:
        """Use a repository already present in the torch.hub cache (no network access)."""
        owner_repo = self.hub_repo.replace("/", "_")
        hub_dir = Path(torch.hub.get_dir())
        candidates = sorted(hub_dir.glob(f"{owner_repo}_*")) if hub_dir.is_dir() else []
        for repo_dir in candidates:
            if (repo_dir / "hubconf.py").is_file():
                loaded = torch.hub.load(str(repo_dir), "silero_vad", source="local", trust_repo=True)
                return loaded[0] if isinstance(loaded, tuple) else loaded
        return None

    def _load_from_hub(self) -> Optional[torch.nn.Module]:
        """Download the repository through ``torch.hub`` (first run, needs network).

        ``torch.hub`` downloads with ``urllib``, which on python.org macOS builds has no
        CA bundle until "Install Certificates.command" is run. Pointing OpenSSL at the
        ``certifi`` bundle (already installed as a dependency) keeps TLS verification on
        and makes the download work out of the box.
        """
        if "SSL_CERT_FILE" not in os.environ:
            try:
                import certifi

                os.environ["SSL_CERT_FILE"] = certifi.where()
            except ImportError:  # pragma: no cover - certifi ships with requests/httpx
                pass
        loaded = torch.hub.load(self.hub_repo, "silero_vad", trust_repo=True, verbose=False)
        return loaded[0] if isinstance(loaded, tuple) else loaded

    @staticmethod
    def _load_from_pip() -> Optional[torch.nn.Module]:
        """Load the TorchScript weights bundled in the ``silero-vad`` wheel."""
        from silero_vad import load_silero_vad

        return load_silero_vad()

    # ------------------------------------------------------------------ probabilities

    @torch.no_grad()
    def speech_probabilities(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Return one speech probability per 32 ms frame.

        Args:
            audio: Mono waveform at ``self.sample_rate``.
        """
        self.load()
        wave = torch.as_tensor(audio, dtype=torch.float32).detach().flatten().cpu()
        if wave.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        remainder = wave.numel() % _FRAME_SAMPLES
        if remainder:
            wave = torch.nn.functional.pad(wave, (0, _FRAME_SAMPLES - remainder))
        frames = wave.view(-1, _FRAME_SAMPLES)

        if self.backend == "energy" or self._model is None:
            return self._energy_probabilities(frames)

        model = self._model
        if hasattr(model, "reset_states"):
            model.reset_states()
        probs = np.empty(frames.shape[0], dtype=np.float32)
        for index in range(frames.shape[0]):
            probs[index] = float(model(frames[index : index + 1], self.sample_rate).item())
        return probs

    @staticmethod
    def _energy_probabilities(frames: torch.Tensor) -> np.ndarray:
        """Map frame energy to a pseudo-probability around an adaptive noise floor.

        A frame 12 dB above the 10th-percentile energy maps to 0.5, so the same
        thresholds used for Silero remain meaningful.
        """
        rms = frames.pow(2).mean(dim=1).sqrt().clamp_min(1e-7)
        db = 20.0 * torch.log10(rms)
        floor = torch.quantile(db, 0.10).clamp_min(-70.0)
        margin = 12.0
        return torch.clamp((db - floor) / (2.0 * margin), 0.0, 1.0).numpy().astype(np.float32)

    # ------------------------------------------------------------------ post-processing

    def _regions_from_probabilities(self, probs: np.ndarray) -> List[Tuple[int, int]]:
        """Hysteresis thresholding -> merge short pauses -> drop blips -> pad -> split long."""
        n_frames = int(probs.shape[0])
        raw: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for index in range(n_frames):
            p = probs[index]
            if start is None and p >= self.threshold:
                start = index
            elif start is not None and p < self.neg_threshold:
                raw.append((start, index))
                start = None
        if start is not None:
            raw.append((start, n_frames))
        if not raw:
            return []

        # Merge regions separated by a pause shorter than ``min_silence_sec``.
        max_gap = int(round(self.min_silence_sec / self.frame_sec))
        merged: List[List[int]] = [list(raw[0])]
        for seg_start, seg_end in raw[1:]:
            if seg_start - merged[-1][1] < max_gap:
                merged[-1][1] = seg_end
            else:
                merged.append([seg_start, seg_end])

        min_frames = max(1, int(round(self.min_speech_sec / self.frame_sec)))
        kept = [(s, e) for s, e in merged if e - s >= min_frames]

        # Pad with context, never crossing the midpoint to a neighbouring region.
        pad = int(round(self.speech_pad_sec / self.frame_sec))
        padded: List[Tuple[int, int]] = []
        for index, (seg_start, seg_end) in enumerate(kept):
            lower = 0 if index == 0 else (kept[index - 1][1] + seg_start) // 2
            upper = n_frames if index == len(kept) - 1 else (seg_end + kept[index + 1][0] + 1) // 2
            padded.append((max(lower, seg_start - pad), min(upper, seg_end + pad)))

        max_frames = max(2, int(self.max_segment_sec / self.frame_sec))
        final: List[Tuple[int, int]] = []
        for seg_start, seg_end in padded:
            final.extend(self._split_long_region(seg_start, seg_end, probs, max_frames))
        return final

    @staticmethod
    def _split_long_region(
        start: int, end: int, probs: np.ndarray, max_frames: int
    ) -> List[Tuple[int, int]]:
        """Cut ``[start, end)`` into pieces of at most ``max_frames`` frames.

        Each cut is placed at the lowest-probability frame inside the second half of the
        allowed window, i.e. at the most pause-like moment before the limit is reached.
        """
        pieces: List[Tuple[int, int]] = []
        while end - start > max_frames:
            lo = start + max_frames // 2
            hi = start + max_frames
            cut = lo + int(np.argmin(probs[lo:hi]))
            cut = min(max(cut, start + 1), hi)
            pieces.append((start, cut))
            start = cut
        pieces.append((start, end))
        return pieces

    # ------------------------------------------------------------------ public API

    def segment(self, audio: Union[np.ndarray, torch.Tensor]) -> List[SpeechSegment]:
        """Split a mono 16 kHz waveform into speech segments.

        Returns:
            Chronologically ordered :class:`SpeechSegment` objects, each at most
            ``max_segment_sec`` long and holding its own CPU audio tensor.
        """
        wave = torch.as_tensor(audio, dtype=torch.float32).detach().flatten().cpu()
        probs = self.speech_probabilities(wave)
        regions = self._regions_from_probabilities(probs)
        total = wave.numel()
        segments: List[SpeechSegment] = []
        for start_frame, end_frame in regions:
            start_sample = min(start_frame * _FRAME_SAMPLES, total)
            end_sample = min(end_frame * _FRAME_SAMPLES, total)
            if end_sample <= start_sample:
                continue
            segments.append(
                SpeechSegment(
                    start_sec=start_sample / self.sample_rate,
                    end_sec=end_sample / self.sample_rate,
                    audio_tensor=wave[start_sample:end_sample].clone(),
                )
            )
        speech = sum(seg.duration for seg in segments)
        logger.info(
            "VAD[%s]: %d segments, %.1fs speech of %.1fs audio",
            self.backend, len(segments), speech, total / self.sample_rate,
        )  # fmt: skip
        return segments

    def segment_file(self, path: PathLike) -> List[SpeechSegment]:
        """Decode an audio/video file (``.wav``/``.mp3``/``.m4a``/``.mp4``) and segment it."""
        return self.segment(load_audio(path, self.sample_rate))
