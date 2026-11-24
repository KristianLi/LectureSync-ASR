"""LectureSync-ASR: local slide/audio alignment and accent-robust lecture transcription.

The package is intentionally import-light: heavy dependencies (torch, transformers,
sentence-transformers) are only imported by the modules that need them, so tools such
as ``python -m src.train_lora --help`` stay fast.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
