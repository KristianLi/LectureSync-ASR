"""End-to-end orchestration: media + PDF -> aligned transcript, vocabulary and notes.

:class:`LecturePipeline` owns one instance of every stage and keeps the models loaded
between runs, so the Gradio app, the CLI and the tests all share the same code path::

    audio/video --> VAD --> Whisper (+LoRA, +glossary prompt) --> transcript segments
    PDF slides  --> text + PNG --> glossary ----^                        |
                         \\--> MiniLM embeddings --> monotonic DP <------/
                                                         |
                                     per-slide transcript + vocabulary cards + Markdown notes

Command line::

    python -m src.pipeline --audio lecture.mp4 --pdf slides.pdf --out outputs/lecture05
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .aligner import SlideAligner, SlideDeck, attach_pages
from .asr import WhisperASR
from .utils import (
    DeviceManager,
    PathLike,
    format_timestamp,
    get_logger,
    load_audio,
    load_config,
    parse_overrides,
    save_wav,
)
from .vad import VoiceActivityDetector
from .vocab import CATEGORY_LABELS, TermExtractor, VocabEntry, highlight_terms, load_glossary_file

logger = get_logger(__name__)

StageCallback = Callable[[float, str], None]


@dataclass
class LectureResult:
    """Everything produced for one lecture.

    Attributes:
        audio_path: 16 kHz mono WAV used for playback (always seekable in the browser).
        pdf_path: Source slide deck.
        duration: Audio duration in seconds.
        deck: Slide texts and rendered page images.
        segments: Transcript segments, each with ``id``, ``start``, ``end``, ``text``,
            ``confidence`` and the aligned zero-based ``page``.
        vocabulary: Ranked study vocabulary.
        glossary: Terms injected into the Whisper prompt.
        page_ranges: ``(start_sec, end_sec)`` narrated for every slide, ``None`` if skipped.
        timings: Wall-clock seconds per stage.
        meta: Model names, device and backend information.
    """

    audio_path: Path
    pdf_path: Path
    duration: float
    deck: SlideDeck
    segments: List[Dict[str, Any]]
    vocabulary: List[VocabEntry]
    glossary: List[str]
    page_ranges: List[Optional[tuple]]
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def segments_for_page(self, page: int) -> List[Dict[str, Any]]:
        """Transcript segments aligned to ``page``."""
        return [seg for seg in self.segments if seg["page"] == page]

    def vocabulary_for_page(self, page: int) -> List[VocabEntry]:
        """Vocabulary entries spoken on, or printed on, ``page``."""
        return [entry for entry in self.vocabulary if page in entry.pages]

    def page_start(self, page: int) -> Optional[float]:
        """Second at which the lecturer starts talking about ``page``."""
        span = self.page_ranges[page] if 0 <= page < len(self.page_ranges) else None
        return float(span[0]) if span else None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary (no audio samples)."""
        return {
            "audio_path": str(self.audio_path),
            "pdf_path": str(self.pdf_path),
            "duration": self.duration,
            "slides": [
                {
                    "index": page.index,
                    "title": page.title,
                    "text": page.text,
                    "image_path": str(page.image_path) if page.image_path else None,
                    "range": list(self.page_ranges[page.index]) if self.page_ranges[page.index] else None,
                }
                for page in self.deck.pages
            ],
            "segments": self.segments,
            "vocabulary": [entry.to_dict() for entry in self.vocabulary],
            "glossary": self.glossary,
            "timings": self.timings,
            "meta": self.meta,
        }


class LecturePipeline:
    """VAD -> ASR -> slide alignment -> vocabulary, with models cached across runs.

    Args:
        config: Full configuration mapping (see ``configs/default.yaml``); the defaults
            are loaded when omitted.
        device_manager: Optional pre-built :class:`DeviceManager` shared by all stages.
    """

    def __init__(
        self, config: Optional[Mapping[str, Any]] = None, device_manager: Optional[DeviceManager] = None
    ) -> None:
        self.config: Dict[str, Any] = dict(config) if config is not None else load_config()
        get_logger(level=str(self.config.get("runtime", {}).get("log_level", "INFO")))
        self.device_manager = device_manager or DeviceManager.from_config(self.config)
        self.sample_rate: int = int(self.config.get("audio", {}).get("sample_rate", 16_000))
        self.vad = VoiceActivityDetector(self.config.get("vad"), self.sample_rate)
        self.asr = WhisperASR(self.config.get("asr"), self.device_manager)
        self.aligner = SlideAligner(self.config.get("aligner"), self.device_manager)

    def run(
        self,
        audio_path: PathLike,
        pdf_path: PathLike,
        glossary_terms: Optional[Sequence[str]] = None,
        work_dir: Optional[PathLike] = None,
        progress: Optional[StageCallback] = None,
    ) -> LectureResult:
        """Process one lecture.

        Args:
            audio_path: Recording or video (``.wav``/``.mp3``/``.m4a``/``.mp4``).
            pdf_path: Slide deck.
            glossary_terms: Optional user glossary, merged ahead of the slide-derived terms.
            work_dir: Output directory for slide PNGs and the playback WAV. Defaults to
                ``<runtime.work_dir>/<pdf stem>``.
            progress: Optional ``callback(fraction, message)`` for UI progress bars.

        Returns:
            A :class:`LectureResult`.
        """

        def report(fraction: float, message: str) -> None:
            logger.info("[%3d%%] %s", int(fraction * 100), message)
            if progress is not None:
                progress(fraction, message)

        audio_path, pdf_path = Path(audio_path), Path(pdf_path)
        base_dir = Path(self.config.get("runtime", {}).get("work_dir", "outputs"))
        out_dir = Path(work_dir) if work_dir is not None else base_dir / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        timings: Dict[str, float] = {}

        report(0.02, "Reading slides")
        tick = time.perf_counter()
        deck = SlideDeck.from_pdf(pdf_path, out_dir / "slides", dpi=self.aligner.render_dpi)
        extractor = TermExtractor(self.config.get("vocab"), custom_terms=glossary_terms)
        glossary = extractor.extract_glossary(deck.texts)
        timings["slides_sec"] = time.perf_counter() - tick

        report(0.08, "Decoding audio")
        tick = time.perf_counter()
        audio = load_audio(audio_path, self.sample_rate)
        duration = len(audio) / self.sample_rate
        playback_path = save_wav(out_dir / "audio_16k.wav", audio, self.sample_rate)
        timings["decode_sec"] = time.perf_counter() - tick

        report(0.15, "Detecting speech (VAD)")
        tick = time.perf_counter()
        speech = self.vad.segment(audio)
        timings["vad_sec"] = time.perf_counter() - tick

        report(0.25, f"Transcribing {len(speech)} speech chunks with {self.asr.model_name}")
        self.device_manager.synchronize()
        tick = time.perf_counter()
        self.asr.load()
        prompt = self.asr.set_glossary(glossary)

        def asr_progress(done: int, total: int) -> None:
            report(0.25 + 0.55 * done / max(1, total), f"Transcribing ({done}/{total} chunks)")

        transcript = self.asr.transcribe(speech, progress=asr_progress)
        self.device_manager.synchronize()
        timings["asr_sec"] = time.perf_counter() - tick

        report(0.82, "Aligning transcript to slides")
        tick = time.perf_counter()
        alignment = self.aligner.align(transcript, deck)
        segments = attach_pages(transcript, alignment.assignments)
        self.device_manager.synchronize()
        timings["align_sec"] = time.perf_counter() - tick

        report(0.93, "Extracting vocabulary")
        tick = time.perf_counter()
        vocabulary = extractor.extract(segments, deck.texts)
        timings["vocab_sec"] = time.perf_counter() - tick

        timings["total_sec"] = sum(timings.values())
        meta = {
            "device": self.device_manager.describe(),
            "asr_model": self.asr.model_name,
            "lora_checkpoint": str(self.asr.lora_checkpoint) if self.asr.lora_checkpoint else None,
            "embedding_model": self.aligner.model_name,
            "vad_backend": self.vad.backend,
            "prompt": prompt,
            "speech_chunks": len(speech),
            "real_time_factor": timings["total_sec"] / duration if duration > 0 else None,
        }
        report(1.0, f"Done: {len(segments)} segments on {len(deck)} slides in {timings['total_sec']:.1f}s")
        return LectureResult(
            audio_path=playback_path,
            pdf_path=pdf_path,
            duration=duration,
            deck=deck,
            segments=segments,
            vocabulary=vocabulary,
            glossary=glossary,
            page_ranges=alignment.page_ranges,
            timings=timings,
            meta=meta,
        )


# --------------------------------------------------------------------------- notes export


def _table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def export_markdown(
    result: LectureResult, out_path: Optional[PathLike] = None, embed_images: bool = True
) -> str:
    """Render per-slide study notes as Markdown.

    Each slide gets its narration time range, the time-stamped transcript with terms in
    bold, and a vocabulary table (term, type, first timestamp, context sentence).

    Args:
        result: Pipeline output.
        out_path: When given, the Markdown is also written to this file.
        embed_images: Link the rendered slide PNGs (relative to ``out_path`` when possible).

    Returns:
        The Markdown document as a string.
    """
    out_file = Path(out_path) if out_path is not None else None
    all_terms = [entry.term for entry in result.vocabulary]
    lines: List[str] = [
        f"# Lecture Notes: {result.pdf_path.stem}",
        "",
        f"> Generated locally by LectureSync-ASR | duration {format_timestamp(result.duration)}"
        f" | {len(result.deck)} slides | {len(result.segments)} transcript segments"
        f" | ASR `{result.meta.get('asr_model', 'whisper')}` on `{result.meta.get('device', 'cpu')}`",
        "",
    ]

    for page in result.deck.pages:
        segments = result.segments_for_page(page.index)
        vocabulary = result.vocabulary_for_page(page.index)
        span = result.page_ranges[page.index]
        lines.append(f"## Slide {page.index + 1}: {page.title}")
        lines.append("")
        if embed_images and page.image_path is not None:
            image_ref = str(page.image_path)
            if out_file is not None:
                try:
                    image_ref = str(Path(page.image_path).resolve().relative_to(out_file.resolve().parent))
                except ValueError:
                    image_ref = str(Path(page.image_path).resolve())
            lines += [f"![Slide {page.index + 1}]({image_ref})", ""]
        if span is not None:
            lines.append(f"**Narration:** {format_timestamp(span[0])} - {format_timestamp(span[1])}")
        else:
            lines.append("**Narration:** _not discussed in the recording_")
        lines += ["", "### Transcript", ""]
        if segments:
            for seg in segments:
                stamp = f"`[{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}]`"
                lines.append(f"- {stamp} {highlight_terms(seg['text'], all_terms, style='markdown')}")
        else:
            lines.append("_No transcript segments were aligned to this slide._")
        lines += ["", "### Key vocabulary", ""]
        if vocabulary:
            lines.append("| Term | Type | First heard | In context |")
            lines.append("| --- | --- | --- | --- |")
            for entry in vocabulary:
                on_page = [occ for occ in entry.occurrences if occ.page == page.index]
                occurrence = on_page[0] if on_page else (entry.occurrences[0] if entry.occurrences else None)
                when = format_timestamp(occurrence.time) if occurrence else "on slide only"
                context = _table_cell(occurrence.context) if occurrence else "_printed on the slide, not heard_"
                label = CATEGORY_LABELS.get(entry.category, entry.category)
                lines.append(f"| **{_table_cell(entry.term)}** | {label} | {when} | {context} |")
        else:
            lines.append("_No key terms detected for this slide._")
        lines.append("")

    lines += ["## Glossary index", ""]
    if result.vocabulary:
        lines.append("| Term | Type | Times heard | Slides | First heard |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in sorted(result.vocabulary, key=lambda e: e.term.lower()):
            slides = ", ".join(str(p + 1) for p in sorted(entry.pages)) or "-"
            when = format_timestamp(entry.first_time) if entry.first_time is not None else "-"
            label = CATEGORY_LABELS.get(entry.category, entry.category)
            lines.append(f"| **{_table_cell(entry.term)}** | {label} | {entry.count} | {slides} | {when} |")
    else:
        lines.append("_No vocabulary extracted._")
    lines.append("")

    markdown = "\n".join(lines)
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(markdown, encoding="utf-8")
        logger.info("Wrote study notes to %s", out_file)
    return markdown


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align a lecture recording with its PDF slides, fully offline.")
    parser.add_argument("--audio", required=True, help="Lecture recording or video (.wav/.mp3/.m4a/.mp4)")
    parser.add_argument("--pdf", required=True, help="Slide deck (.pdf)")
    parser.add_argument("--glossary", default=None, help="Optional text file with extra terms")
    parser.add_argument("--out", default=None, help="Output directory (default: outputs/<pdf name>)")
    parser.add_argument("--config", default=None, help="YAML config overriding configs/default.yaml")
    parser.add_argument("--lora", default=None, help="LoRA adapter directory from src/train_lora.py")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set asr.model_name=openai/whisper-tiny")  # fmt: skip
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: run the pipeline and write ``notes.md`` and ``result.json``."""
    args = build_arg_parser().parse_args(argv)
    overrides = parse_overrides(args.overrides)
    if args.lora:
        overrides.setdefault("asr", {})["lora_checkpoint"] = args.lora
    config = load_config(args.config, overrides)
    pipeline = LecturePipeline(config)
    out_dir = Path(args.out) if args.out else None
    result = pipeline.run(args.audio, args.pdf, load_glossary_file(args.glossary), work_dir=out_dir)
    out_dir = result.audio_path.parent
    export_markdown(result, out_dir / "notes.md")
    (out_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notes:  {out_dir / 'notes.md'}")
    print(f"Result: {out_dir / 'result.json'}")
    for name, seconds in result.timings.items():
        print(f"  {name:<12} {seconds:7.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
