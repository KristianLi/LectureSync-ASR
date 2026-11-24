"""Shared runtime utilities: configuration, device management, audio I/O and logging.

Everything that touches a ``torch.device`` goes through :class:`DeviceManager` so the
rest of the code base never hard-codes ``"mps"``, ``"cuda"`` or ``"cpu"``.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Union

# Must be set before torch initialises its MPS backend: operators that Metal does not
# implement then run on the CPU instead of raising NotImplementedError.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"
SAMPLE_RATE: int = 16_000

PathLike = Union[str, Path]

# --------------------------------------------------------------------------- logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "lecturesync"


def get_logger(name: str = _ROOT_LOGGER_NAME, level: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger, configuring the package root handler once.

    Args:
        name: Logger name; names outside the ``lecturesync`` namespace are nested under it.
        level: Optional level name (``"DEBUG"``, ``"INFO"`` ...) applied to the package root.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("LECTURESYNC_LOG_LEVEL", "INFO").upper())
        root.propagate = False
    if level:
        root.setLevel(level.upper())
    if name == _ROOT_LOGGER_NAME or name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


logger = get_logger(__name__)

# --------------------------------------------------------------------------- config


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` in place and return ``base``."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def parse_overrides(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Turn ``["asr.model_name=openai/whisper-tiny", "vad.threshold=0.4"]`` into a nested dict.

    Values are parsed as YAML scalars, so numbers, booleans and ``null`` keep their types.
    """
    result: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override must look like section.key=value, got: {pair!r}")
        dotted, raw = pair.split("=", 1)
        node = result
        keys = dotted.strip().split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return result


def load_config(
    path: Optional[PathLike] = None, overrides: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Load the YAML configuration, layered as defaults <- ``path`` <- ``overrides``.

    Args:
        path: Optional user config; missing sections fall back to ``configs/default.yaml``.
        overrides: Nested mapping applied last (see :func:`parse_overrides`).
    """
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path, "r", encoding="utf-8") as handle:
            _deep_update(config, yaml.safe_load(handle) or {})
    if overrides:
        _deep_update(config, overrides)
    return config


# --------------------------------------------------------------------------- devices


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available() and backend.is_built())


class DeviceManager:
    """Select the compute device and a numerically safe dtype for it.

    Selection order for ``preferred="auto"`` is Apple ``mps`` -> ``cuda`` -> ``cpu``.

    Precision policy:

    * ``cuda`` uses ``float16`` (large speed-up, well supported).
    * ``mps`` uses ``float32`` by default. ``float16`` is only used when explicitly
      requested *and* :meth:`probe_half_precision` shows the operators Whisper needs
      run and stay finite on this machine's Metal backend; otherwise it falls back.
    * ``cpu`` always uses ``float32``.
    """

    def __init__(self, preferred: str = "auto", dtype: str = "auto") -> None:
        self.device: torch.device = self._select_device(preferred)
        self.dtype: torch.dtype = self._select_dtype(dtype)
        logger.info("Device manager ready: %s", self.describe())

    # -- construction helpers -------------------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DeviceManager":
        """Build a manager from the ``runtime`` section of the configuration."""
        runtime = config.get("runtime", {}) if config else {}
        return cls(str(runtime.get("device", "auto")), str(runtime.get("dtype", "auto")))

    @staticmethod
    def _select_device(preferred: str) -> torch.device:
        preferred = (preferred or "auto").lower()
        available = {
            "mps": _mps_available(),
            "cuda": torch.cuda.is_available(),
            "cpu": True,
        }
        if preferred != "auto":
            kind = preferred.split(":")[0]
            if available.get(kind, False):
                return torch.device(preferred)
            logger.warning("Requested device %r is unavailable; auto-selecting instead.", preferred)
        for kind in ("mps", "cuda", "cpu"):
            if available[kind]:
                return torch.device(kind)
        return torch.device("cpu")

    def _select_dtype(self, requested: str) -> torch.dtype:
        requested = (requested or "auto").lower()
        kind = self.device.type
        if kind == "cpu":
            if requested in {"float16", "fp16", "half"}:
                logger.warning("float16 inference is not supported on CPU; using float32.")
            return torch.float32
        if kind == "cuda":
            return torch.float32 if requested in {"float32", "fp32"} else torch.float16
        # mps
        if requested in {"float16", "fp16", "half"}:
            if self.probe_half_precision():
                return torch.float16
            logger.warning("MPS float16 probe failed; falling back to float32.")
        return torch.float32

    # -- capability probing ---------------------------------------------------------

    def probe_half_precision(self) -> bool:
        """Check that the float16 operators Whisper relies on work on this device.

        Runs a miniature conv -> GELU -> LayerNorm -> attention -> log-softmax chain in
        half precision and verifies that the result is finite. Any exception or NaN
        means the caller should stay in ``float32``.
        """
        if self.device.type == "cpu":
            return False
        try:
            with torch.no_grad():
                x = torch.randn(1, 8, 64, device=self.device, dtype=torch.float16)
                weight = torch.randn(16, 8, 3, device=self.device, dtype=torch.float16)
                h = torch.nn.functional.gelu(torch.nn.functional.conv1d(x, weight, padding=1))
                h = torch.nn.functional.layer_norm(h.transpose(1, 2), (16,))
                attn = torch.nn.functional.scaled_dot_product_attention(h, h, h)
                out = torch.log_softmax(attn @ attn.transpose(1, 2), dim=-1)
                self.synchronize()
                return bool(torch.isfinite(out.float()).all().item())
        except Exception as exc:  # noqa: BLE001 - any backend failure means "unsupported"
            logger.debug("Half-precision probe raised %s: %s", type(exc).__name__, exc)
            return False

    def fallback_to_float32(self) -> torch.dtype:
        """Downgrade the managed dtype to float32 (used after a runtime fp16 failure)."""
        self.dtype = torch.float32
        return self.dtype

    # -- runtime helpers ------------------------------------------------------------

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def synchronize(self) -> None:
        """Block until all queued GPU work has finished.

        MPS and CUDA execute asynchronously, so wall-clock timings are only meaningful
        when this is called immediately before reading the clock.
        """
        if self.is_mps:
            torch.mps.synchronize()
        elif self.is_cuda:
            torch.cuda.synchronize(self.device)

    def empty_cache(self) -> None:
        """Release cached allocator blocks back to the OS / driver."""
        if self.is_mps:
            torch.mps.empty_cache()
        elif self.is_cuda:
            torch.cuda.empty_cache()

    def memory_stats(self) -> Dict[str, float]:
        """Return accelerator memory counters in MiB (empty on CPU)."""
        mib = 1024.0 * 1024.0
        if self.is_mps:
            return {
                "mps_allocated_mb": torch.mps.current_allocated_memory() / mib,
                "mps_driver_mb": torch.mps.driver_allocated_memory() / mib,
            }
        if self.is_cuda:
            return {
                "cuda_allocated_mb": torch.cuda.memory_allocated(self.device) / mib,
                "cuda_peak_mb": torch.cuda.max_memory_allocated(self.device) / mib,
            }
        return {}

    def describe(self) -> str:
        """Human-readable summary, e.g. ``mps (Apple M4) / float32``."""
        name = self.device.type
        if self.is_cuda:
            name = f"cuda ({torch.cuda.get_device_name(self.device)})"
        elif self.is_mps:
            name = f"mps ({cpu_brand()})"
        return f"{name} / {str(self.dtype).replace('torch.', '')}"


def cpu_brand() -> str:
    """Best-effort CPU / SoC name (``Apple M4``, ``Intel(R) Xeon(R) ...``)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    import platform

    return platform.processor() or platform.machine() or "unknown CPU"


@contextmanager
def timed(label: str, device_manager: Optional[DeviceManager] = None) -> Iterator[Dict[str, float]]:
    """Context manager measuring a block's wall time with correct GPU synchronisation.

    Yields a dict whose ``"seconds"`` entry is filled in when the block exits.
    """
    record: Dict[str, float] = {"seconds": 0.0}
    if device_manager is not None:
        device_manager.synchronize()
    start = time.perf_counter()
    try:
        yield record
    finally:
        if device_manager is not None:
            device_manager.synchronize()
        record["seconds"] = time.perf_counter() - start
        logger.info("%s finished in %.2fs", label, record["seconds"])


# --------------------------------------------------------------------------- audio

_SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
SUPPORTED_MEDIA_SUFFIXES = _SOUNDFILE_SUFFIXES | {".m4a", ".mp4", ".mov", ".mkv", ".webm", ".aac"}


def ffmpeg_available() -> bool:
    """Return True when an ``ffmpeg`` binary is on ``PATH``."""
    return shutil.which("ffmpeg") is not None


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float waveform, preferring torchaudio's polyphase kernel."""
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    try:
        import torchaudio.functional as taf

        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return taf.resample(tensor, orig_sr, target_sr).numpy()
    except Exception as exc:  # noqa: BLE001 - torchaudio is optional at runtime
        logger.debug("torchaudio resample unavailable (%s); using librosa.", exc)
        import librosa

        return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def _load_with_soundfile(path: Path, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), sr, sample_rate)


def _load_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    """Decode any container ffmpeg understands straight to 16-bit-free float32 PCM."""
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
    ]  # fmt: skip
    proc = subprocess.run(command, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode(errors='ignore')[:400]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_with_afconvert(path: Path, sample_rate: int) -> np.ndarray:
    """Decode through macOS CoreAudio (``afconvert``), so .m4a/.mp4 work without ffmpeg."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lecturesync_") as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        command = [
            "afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}", "-c", "1",
            str(path), str(wav_path),
        ]  # fmt: skip
        proc = subprocess.run(command, capture_output=True, check=False)
        if proc.returncode != 0 or not wav_path.is_file():
            raise RuntimeError(f"afconvert failed: {proc.stderr.decode(errors='ignore')[:300]}")
        return _load_with_soundfile(wav_path, sample_rate)


def _load_with_librosa(path: Path, sample_rate: int) -> np.ndarray:
    """Last-resort decoder (whatever backends the installed librosa/soundfile provide)."""
    import warnings

    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio.astype(np.float32, copy=False)


def load_audio(path: PathLike, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio or video file to a mono ``float32`` waveform in ``[-1, 1]``.

    Decoder order: ``soundfile`` for PCM-style containers and MP3, then the ``ffmpeg``
    CLI when installed (broadest format coverage), then macOS ``afconvert`` (CoreAudio,
    handles ``.m4a``/``.mp4`` with no extra install), then ``librosa``.

    Args:
        path: Media file (``.wav``, ``.mp3``, ``.m4a``, ``.mp4`` ...).
        sample_rate: Target sampling rate in Hz.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If no decoder could read the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    decoders = []
    if path.suffix.lower() in _SOUNDFILE_SUFFIXES:
        decoders.append(("soundfile", _load_with_soundfile))
    if ffmpeg_available():
        decoders.append(("ffmpeg", _load_with_ffmpeg))
    if shutil.which("afconvert") is not None:
        decoders.append(("afconvert", _load_with_afconvert))
    decoders.append(("librosa", _load_with_librosa))

    errors = []
    for name, decoder in decoders:
        try:
            audio = decoder(path, sample_rate)
            if audio.size == 0:
                raise RuntimeError("decoder returned no samples")
            logger.debug("Decoded %s with %s (%.1fs)", path.name, name, audio.size / sample_rate)
            return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
        except Exception as exc:  # noqa: BLE001 - try the next decoder
            errors.append(f"{name}: {exc}")
    hint = "" if ffmpeg_available() else " Installing ffmpeg (`brew install ffmpeg`) adds support for more formats."
    raise RuntimeError(f"Could not decode {path.name}. Tried " + "; ".join(errors) + "." + hint)


def save_wav(path: PathLike, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write a mono waveform as 16-bit PCM WAV and return the path."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
    return path


# --------------------------------------------------------------------------- text

_PUNCT_RE = re.compile(r"[^\w\s'\-]")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace (WER / term-recall scoring)."""
    text = _PUNCT_RE.sub(" ", text.lower().replace("’", "'"))
    text = text.replace("-", " ")
    return _SPACE_RE.sub(" ", text).strip()


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS`` (or ``H:MM:SS`` past one hour)."""
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (``ru_maxrss`` is bytes on macOS, KiB on Linux)."""
    import resource
    import sys

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0
