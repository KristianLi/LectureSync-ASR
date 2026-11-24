"""Gradio study workbench: slide-synchronised listening practice, fully offline.

Run with ``python app/ui.py`` and open http://127.0.0.1:7860.

Layout:

* **Upload panel** - lecture recording or video, PDF slides, optional glossary.
* **Slide navigator** - rendered slide with Previous / Next buttons and a page selector.
  Selecting a slide shows only the transcript aligned to it and moves the audio player
  to the moment the lecturer starts talking about it.
* **Transcript and vocabulary cards** - time-stamped lines with CS/AI terms highlighted;
  clicking a line or a card replays that sentence. 0.75x / 1.0x / 1.25x speed buttons.
* **Export** - one click downloads Markdown notes structured per slide.

The audio element is a plain ``<audio>`` tag inside a ``gr.HTML`` block so that seeking,
playback rate and "highlight the line being spoken" are driven by a few lines of
JavaScript (see ``HEAD_HTML``) rather than by round trips to the Python process.
"""

from __future__ import annotations

import argparse
import copy
import html
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.pipeline import LecturePipeline, LectureResult, export_markdown  # noqa: E402
from src.utils import format_timestamp, get_logger, load_config, parse_overrides  # noqa: E402
from src.vocab import CATEGORY_LABELS, VocabEntry, highlight_terms, load_glossary_file  # noqa: E402

logger = get_logger("ui")

HOST = "127.0.0.1"
PORT = 7860
WHISPER_CHOICES = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3-turbo",
]
LOW_CONFIDENCE = 0.6

# --------------------------------------------------------------------------- pipeline cache

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: Optional[LecturePipeline] = None
_PIPELINE_KEY: Optional[Tuple[str, str]] = None
_BASE_CONFIG: Dict[str, Any] = load_config()


def ui_work_dir() -> Path:
    """Directory that holds per-lecture artefacts served to the browser."""
    return (ROOT / str(_BASE_CONFIG["runtime"].get("work_dir", "outputs")) / "ui").resolve()


def get_pipeline(model_name: str, lora_checkpoint: str) -> LecturePipeline:
    """Return a pipeline for the requested ASR model, reusing loaded weights when possible."""
    global _PIPELINE, _PIPELINE_KEY
    key = (model_name, lora_checkpoint or "")
    with _PIPELINE_LOCK:
        if _PIPELINE is None or _PIPELINE_KEY != key:
            if _PIPELINE is not None:
                _PIPELINE.asr.model = None  # drop the previous Whisper weights before loading new ones
                _PIPELINE.device_manager.empty_cache()
            config = copy.deepcopy(_BASE_CONFIG)
            config["asr"] = dict(config["asr"], model_name=model_name, lora_checkpoint=lora_checkpoint or None)
            _PIPELINE, _PIPELINE_KEY = LecturePipeline(config), key
        return _PIPELINE


# --------------------------------------------------------------------------- HTML rendering

EMPTY_PLAYER = '<div class="ls-empty">Upload a recording and its slides, then press <b>Process lecture</b>.</div>'
EMPTY_PANEL = '<div class="ls-empty">Nothing to show yet.</div>'


def file_url(path: Path) -> str:
    """URL under which Gradio serves a file from an allowed directory."""
    return f"/gradio_api/file={Path(path).resolve()}"


def render_player(result: LectureResult) -> str:
    """The audio element shared by every slide of one lecture."""
    return (
        '<div class="ls-player-wrap">'
        f'<audio id="ls-player" controls preload="metadata" src="{html.escape(file_url(result.audio_path))}"></audio>'
        f'<div class="ls-player-meta">{format_timestamp(result.duration)} total'
        f" &middot; {html.escape(str(result.meta.get('asr_model', '')))}"
        f" &middot; {html.escape(str(result.meta.get('device', '')))}</div>"
        "</div>"
    )


def render_transcript(result: LectureResult, page: int) -> str:
    """Transcript lines aligned to ``page``; each line seeks the player when clicked."""
    segments = result.segments_for_page(page)
    start = result.page_start(page)
    terms = [entry.term for entry in result.vocabulary]
    attrs = f'data-render-id="{uuid.uuid4().hex}" data-page-start="{start if start is not None else -1}"'
    if not segments:
        return (
            f'<div class="ls-transcript" {attrs}><div class="ls-empty">'
            "No narration was aligned to this slide. The lecturer probably skipped it.</div></div>"
        )
    rows: List[str] = []
    for seg in segments:
        low = float(seg.get("confidence", 1.0)) < LOW_CONFIDENCE
        title = "Low ASR confidence - worth listening to carefully" if low else "Click to replay this sentence"
        rows.append(
            f'<div class="ls-line{" ls-lowconf" if low else ""}" data-ls-seek="{seg["start"]:.2f}" '
            f'data-start="{seg["start"]:.2f}" data-end="{seg["end"]:.2f}" title="{title}">'
            f'<span class="ls-time">{format_timestamp(seg["start"])}</span>'
            f'<span class="ls-text">{highlight_terms(seg["text"], terms)}</span>'
            "</div>"
        )
    return f'<div class="ls-transcript" {attrs}>{"".join(rows)}</div>'


def _vocab_card(entry: VocabEntry, page: Optional[int]) -> str:
    on_page = [occ for occ in entry.occurrences if page is None or occ.page == page]
    occurrence = on_page[0] if on_page else (entry.occurrences[0] if entry.occurrences else None)
    label = CATEGORY_LABELS.get(entry.category, entry.category)
    head = (
        f'<div class="ls-card-head"><span class="ls-card-term">{html.escape(entry.term)}</span>'
        f'<span class="ls-badge ls-badge-{entry.category}">{label}</span></div>'
    )
    if occurrence is None:
        body = '<div class="ls-card-ctx ls-muted">Printed on the slide but not heard in the recording.</div>'
    else:
        where = f"Slide {occurrence.page + 1} &middot; " if page is None and occurrence.page >= 0 else ""
        times = f" &middot; heard {entry.count}&times;" if entry.count > 1 else ""
        body = (
            f'<button class="ls-chip" data-ls-seek="{occurrence.segment_start:.2f}" '
            f'title="Replay the sentence">&#9654; {where}{format_timestamp(occurrence.time)}</button>'
            f'<span class="ls-muted">{times}</span>'
            f'<div class="ls-card-ctx">&ldquo;{highlight_terms(occurrence.context, [entry.term])}&rdquo;</div>'
        )
    return f'<div class="ls-card">{head}{body}</div>'


def render_vocab(result: LectureResult, page: Optional[int]) -> str:
    """Vocabulary cards for one slide, or for the whole lecture when ``page`` is ``None``."""
    entries = result.vocabulary if page is None else result.vocabulary_for_page(page)
    if not entries:
        return '<div class="ls-empty">No key terms detected here.</div>'
    def heard_at(entry: VocabEntry) -> Tuple[bool, float]:
        times = [occ.time for occ in entry.occurrences if page is None or occ.page == page]
        return (not times, min(times) if times else 0.0)

    entries = sorted(entries, key=heard_at)  # study order: as the lecturer says them, slide-only terms last
    return f'<div class="ls-cards">{"".join(_vocab_card(entry, page) for entry in entries)}</div>'


def alignment_note(result: LectureResult) -> str:
    """One sentence on how slides were matched (video frames or transcript only)."""
    visual = result.meta.get("visual") or {}
    if visual.get("used"):
        return (
            f"video frames + transcript (slides visible in {100 * visual.get('slide_visible_fraction', 0):.0f}% "
            "of the video)"
        )
    return "transcript only (no slides found in the picture, or audio-only input)"


def page_choices(result: LectureResult) -> List[Tuple[str, int]]:
    """Dropdown entries such as ``3 / 12 - Gradient Descent (04:10)``."""
    choices: List[Tuple[str, int]] = []
    total = len(result.deck)
    for slide in result.deck.pages:
        start = result.page_start(slide.index)
        stamp = format_timestamp(start) if start is not None else "not narrated"
        title = slide.title if len(slide.title) <= 48 else slide.title[:45] + "..."
        choices.append((f"{slide.index + 1} / {total} - {title} ({stamp})", slide.index))
    return choices


# --------------------------------------------------------------------------- callbacks


def process_lecture(
    audio_file: Optional[str],
    pdf_file: Optional[str],
    glossary_file: Optional[str],
    extra_terms: str,
    model_name: str,
    lora_checkpoint: str,
    progress: gr.Progress = gr.Progress(),
):
    """Run the pipeline on the uploaded files and populate every panel."""
    if not audio_file or not pdf_file:
        raise gr.Error("Please upload both a lecture recording (audio or video) and the PDF slides.")
    lora_checkpoint = (lora_checkpoint or "").strip()
    if lora_checkpoint and not (Path(lora_checkpoint) / "adapter_config.json").is_file():
        raise gr.Error(f"No LoRA adapter found in: {lora_checkpoint}")

    terms = load_glossary_file(glossary_file) if glossary_file else []
    terms += [t.strip() for t in re.split(r"[,;\n]", extra_terms or "") if t.strip()]

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(pdf_file).stem)[:40] or "lecture"
    work_dir = ui_work_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}"
    progress(0.0, desc="Loading models")
    try:
        pipeline = get_pipeline(model_name, lora_checkpoint)
        result = pipeline.run(
            audio_file, pdf_file, glossary_terms=terms, work_dir=work_dir,
            progress=lambda fraction, message: progress(fraction, desc=message),
        )  # fmt: skip
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure in the UI
        logger.exception("Processing failed")
        raise gr.Error(f"Processing failed: {exc}") from exc

    export_markdown(result, work_dir / "notes_with_slides.md", embed_images=True)
    notes_path = work_dir / f"{stem}_notes.md"
    notes_md = export_markdown(result, notes_path, embed_images=False)

    narrated = sum(1 for span in result.page_ranges if span is not None)
    status = (
        f"**Done.** {len(result.segments)} transcript segments aligned to {narrated} of {len(result.deck)} slides, "
        f"{len(result.vocabulary)} key terms. Processed {format_timestamp(result.duration)} of audio in "
        f"{result.timings['total_sec']:.1f} s (RTF {result.meta['real_time_factor']:.2f}) on {result.meta['device']}.\n\n"
        f"**Slide matching:** {alignment_note(result)}\n\n"
        f"**Glossary injected into Whisper:** {', '.join(result.glossary[:12]) or 'none'}"
        f"{' ...' if len(result.glossary) > 12 else ''}\n\n**Files:** `{work_dir}`"
    )
    return (
        result,
        gr.Dropdown(choices=page_choices(result), value=0, interactive=True),
        status,
        render_player(result),
        gr.DownloadButton(value=str(notes_path), interactive=True),
        notes_md,
        render_vocab(result, None),
    )


def show_page(result: Optional[LectureResult], page: Optional[int]):
    """Render the slide image, its transcript and its vocabulary cards."""
    if result is None or page is None:
        return None, "", EMPTY_PANEL, EMPTY_PANEL
    page = max(0, min(int(page), len(result.deck) - 1))
    slide = result.deck.pages[page]
    span = result.page_ranges[page]
    when = f"{format_timestamp(span[0])} - {format_timestamp(span[1])}" if span else "not narrated"
    later = result.visits(page)[1:]
    if later:
        when += " (shown again at " + ", ".join(format_timestamp(start) for start, _ in later[:4]) + ")"
    info = f"**Slide {page + 1} of {len(result.deck)}** &nbsp;|&nbsp; narration {when} &nbsp;|&nbsp; {len(result.segments_for_page(page))} segments"
    image = str(slide.image_path) if slide.image_path else None
    return image, info, render_transcript(result, page), render_vocab(result, page)


def step_page(result: Optional[LectureResult], page: Optional[int], delta: int) -> Optional[int]:
    """Move ``delta`` slides forward or back, clamped to the deck."""
    if result is None:
        return page
    return max(0, min((page or 0) + delta, len(result.deck) - 1))


# --------------------------------------------------------------------------- front-end assets

HEAD_HTML = """
<script>
(() => {
  if (window.lsReady) return;
  window.lsReady = true;
  window.lsRate = 1.0;
  let lastRender = null;
  const player = () => document.getElementById('ls-player');

  // Seek the shared player; waits for metadata when the file has just been attached.
  window.lsSeek = (seconds, play) => {
    const audio = player();
    if (!audio || !(seconds >= 0)) return;
    const go = () => {
      audio.currentTime = seconds;
      audio.playbackRate = window.lsRate;
      if (play) audio.play().catch(() => {});
    };
    if (audio.readyState >= 1) go(); else audio.addEventListener('loadedmetadata', go, { once: true });
  };

  window.lsSetRate = (rate) => {
    window.lsRate = rate;
    const audio = player();
    if (audio) audio.playbackRate = rate;
    document.querySelectorAll('.ls-speed-btn').forEach((el) => {
      const button = el.tagName === 'BUTTON' ? el : el.querySelector('button') || el;
      button.classList.toggle('ls-speed-active', Math.abs(parseFloat(button.textContent) - rate) < 0.01);
    });
  };

  // Transcript lines and vocabulary chips carry data-ls-seek="<seconds>".
  document.addEventListener('click', (event) => {
    const target = event.target.closest && event.target.closest('[data-ls-seek]');
    if (target) window.lsSeek(parseFloat(target.dataset.lsSeek), true);
  });

  // Media events do not bubble, so listen in the capture phase.
  document.addEventListener('timeupdate', (event) => {
    if (!event.target || event.target.id !== 'ls-player') return;
    const now = event.target.currentTime;
    document.querySelectorAll('.ls-line').forEach((line) => {
      const active = now >= parseFloat(line.dataset.start) && now < parseFloat(line.dataset.end);
      line.classList.toggle('ls-line-active', active);
    });
  }, true);
  document.addEventListener('loadedmetadata', (event) => {
    if (event.target && event.target.id === 'ls-player') event.target.playbackRate = window.lsRate;
  }, true);

  // When a new slide's transcript is rendered, cue the player to that slide's start.
  const cue = () => {
    const transcript = document.querySelector('.ls-transcript[data-render-id]');
    if (!transcript || transcript.dataset.renderId === lastRender) return;
    lastRender = transcript.dataset.renderId;
    const audio = player();
    if (audio) audio.pause();
    window.lsSeek(parseFloat(transcript.dataset.pageStart), false);
    window.lsSetRate(window.lsRate);
  };
  const start = () => new MutationObserver(cue).observe(document.body, { childList: true, subtree: true });
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
})();
</script>
"""

CSS = """
.ls-header h1 { margin-bottom: 2px; }
.ls-header p { margin-top: 0; opacity: .75; }
.ls-empty { padding: 18px; border: 1px dashed var(--border-color-primary); border-radius: 10px;
  color: var(--body-text-color-subdued); text-align: center; font-size: .95rem; }
.ls-player-wrap audio { width: 100%; }
.ls-player-meta { font-size: .8rem; color: var(--body-text-color-subdued); margin-top: 4px; }
.ls-transcript { max-height: 430px; overflow-y: auto; padding-right: 4px; }
.ls-line { display: flex; gap: 10px; padding: 7px 9px; border-radius: 8px; cursor: pointer;
  line-height: 1.5; border-left: 3px solid transparent; }
.ls-line:hover { background: var(--background-fill-secondary); }
.ls-line-active { background: var(--background-fill-secondary); border-left-color: var(--color-accent); }
.ls-time { flex: 0 0 auto; font-variant-numeric: tabular-nums; font-size: .82rem; padding-top: 2px;
  color: var(--color-accent); font-weight: 600; }
.ls-lowconf .ls-text { text-decoration: underline dotted var(--body-text-color-subdued); text-underline-offset: 4px; }
mark.ls-term { background: color-mix(in srgb, var(--color-accent) 22%, transparent); color: inherit;
  padding: 0 3px; border-radius: 4px; font-weight: 600; }
.ls-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px; }
.ls-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 12px;
  background: var(--background-fill-primary); }
.ls-card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 6px; }
.ls-card-term { font-weight: 700; font-size: 1.02rem; }
.ls-badge { font-size: .7rem; padding: 1px 7px; border-radius: 999px; white-space: nowrap;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.ls-badge-cs_ai, .ls-badge-custom { border-color: var(--color-accent); color: var(--color-accent); }
.ls-chip { font-size: .8rem; border: 1px solid var(--border-color-primary); border-radius: 999px; padding: 1px 9px;
  background: var(--background-fill-secondary); color: var(--body-text-color); cursor: pointer; }
.ls-chip:hover { border-color: var(--color-accent); }
.ls-card-ctx { font-size: .88rem; line-height: 1.45; margin-top: 6px; }
.ls-muted { color: var(--body-text-color-subdued); font-size: .8rem; }
.ls-speed-btn, .ls-speed-btn button { transition: none !important; }
.ls-speed-btn.ls-speed-active, .ls-speed-btn button.ls-speed-active { background: var(--color-accent) !important;
  border-color: var(--color-accent) !important; color: #fff !important; }
"""


# --------------------------------------------------------------------------- app


def build_app() -> gr.Blocks:
    """Assemble the Gradio Blocks application."""
    default_model = str(_BASE_CONFIG["asr"].get("model_name", "openai/whisper-small"))
    model_choices = WHISPER_CHOICES if default_model in WHISPER_CHOICES else [default_model, *WHISPER_CHOICES]

    with gr.Blocks(title="LectureSync-ASR") as demo:
        state = gr.State(None)
        gr.HTML(
            '<div class="ls-header"><h1>LectureSync-ASR</h1>'
            "<p>Slide-synchronised lecture transcripts for focused listening practice. "
            "Everything runs on this machine: no cloud APIs, no uploads.</p></div>"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=300):
                gr.Markdown("### 1. Upload")
                audio_in = gr.File(
                    label="Lecture audio or video", type="filepath",
                    file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mov", ".flac", ".ogg", ".aac", ".webm", ".mkv"],
                )  # fmt: skip
                pdf_in = gr.File(label="Slides (.pdf)", type="filepath", file_types=[".pdf"])
                with gr.Accordion("Optional: glossary and model", open=False):
                    glossary_in = gr.File(
                        label="Custom glossary (one term per line)", type="filepath", file_types=[".txt", ".csv", ".md"]
                    )
                    terms_in = gr.Textbox(label="Extra terms", placeholder="e.g. RANSAC, epipolar geometry, LoRA", lines=2)
                    model_in = gr.Dropdown(model_choices, value=default_model, label="Whisper model")
                    lora_in = gr.Textbox(
                        label="LoRA adapter directory",
                        value=str(_BASE_CONFIG["asr"].get("lora_checkpoint") or ""),
                        placeholder="checkpoints/whisper-small-lora-edacc",
                    )
                process_btn = gr.Button("Process lecture", variant="primary")
                status = gr.Markdown("Upload a recording and its slides to begin.")
                export_btn = gr.DownloadButton("Export Markdown study notes", interactive=False)

            with gr.Column(scale=7):
                gr.Markdown("### 2. Study slide by slide")
                with gr.Row(equal_height=True):
                    prev_btn = gr.Button("< Previous", scale=1, min_width=110)
                    page_dd = gr.Dropdown(choices=[], value=None, label="Slide", show_label=False, scale=6,
                                          interactive=False, container=False)  # fmt: skip
                    next_btn = gr.Button("Next >", scale=1, min_width=110)
                page_info = gr.Markdown("")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6, min_width=380):
                        slide_img = gr.Image(label="Slide", show_label=False, interactive=False, height=430)
                    with gr.Column(scale=5, min_width=340):
                        player = gr.HTML(EMPTY_PLAYER)
                        with gr.Row():
                            speed_slow = gr.Button("0.75x", size="sm", elem_classes="ls-speed-btn", min_width=70)
                            speed_normal = gr.Button("1.0x", size="sm", elem_classes="ls-speed-btn", min_width=70)
                            speed_fast = gr.Button("1.25x", size="sm", elem_classes="ls-speed-btn", min_width=70)
                        transcript = gr.HTML(EMPTY_PANEL)
                with gr.Tabs():
                    with gr.Tab("Vocabulary on this slide"):
                        vocab_page = gr.HTML(EMPTY_PANEL)
                    with gr.Tab("All vocabulary"):
                        vocab_all = gr.HTML(EMPTY_PANEL)
                    with gr.Tab("Notes preview"):
                        notes_preview = gr.Markdown("_Process a lecture to preview the exported notes._")

        page_outputs = [slide_img, page_info, transcript, vocab_page]
        process_btn.click(
            process_lecture,
            inputs=[audio_in, pdf_in, glossary_in, terms_in, model_in, lora_in],
            outputs=[state, page_dd, status, player, export_btn, notes_preview, vocab_all],
        ).then(show_page, inputs=[state, page_dd], outputs=page_outputs)
        page_dd.change(show_page, inputs=[state, page_dd], outputs=page_outputs)
        prev_btn.click(lambda result, page: step_page(result, page, -1), inputs=[state, page_dd], outputs=page_dd)
        next_btn.click(lambda result, page: step_page(result, page, +1), inputs=[state, page_dd], outputs=page_dd)
        speed_slow.click(None, js="() => window.lsSetRate(0.75)")
        speed_normal.click(None, js="() => window.lsSetRate(1.0)")
        speed_fast.click(None, js="() => window.lsSetRate(1.25)")

    demo.queue(default_concurrency_limit=1)
    return demo


def launch_kwargs(host: str = HOST, port: int = PORT, share: bool = False) -> Dict[str, Any]:
    """Arguments for ``Blocks.launch`` (Gradio 6 takes theme, css and head here)."""
    work_dir = ui_work_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    return {
        "server_name": host,
        "server_port": port,
        "share": share,
        "allowed_paths": [str(work_dir)],
        "theme": gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        "css": CSS,
        "head": HEAD_HTML,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python app/ui.py [--port 7860] [--set asr.model_name=...]``."""
    parser = argparse.ArgumentParser(description="Launch the LectureSync-ASR study workbench.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (off by default)")
    parser.add_argument("--config", default=None, help="YAML config overriding configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    global _BASE_CONFIG
    _BASE_CONFIG = load_config(args.config, parse_overrides(args.overrides))
    demo = build_app()
    logger.info("Starting LectureSync-ASR on http://%s:%d", args.host, args.port)
    demo.launch(**launch_kwargs(args.host, args.port, args.share))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
