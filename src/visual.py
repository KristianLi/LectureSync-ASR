"""Visual slide matching: find out which slide is on screen in a lecture video.

Text alone cannot tell slides apart when several consecutive pages share their wording
and differ only in a figure (step-by-step animations, worked examples). If the recording
is a screen capture, the picture settles the question directly: every sampled video
frame is compared with a low-resolution rendering of every slide.

Steps:

1. **Frames** - only key frames are decoded (PyAV, ``skip_frame = NONKEY``), which reads
   an hour of 720p video in about a second. Where two neighbouring key frames show
   different slides, the stretch between them is decoded as well so the page turn is
   located to about a second.
2. **Slide region** - the slide rarely fills the frame (pillar-boxing, a presenter
   inset). A coarse-to-fine search over rectangles with the deck's aspect ratio picks the
   one whose content correlates best with the deck across a sample of frames.
3. **Similarity** - zero-mean normalised correlation between the cropped frame and each
   slide, both reduced to a small grey patch. A frame showing slide ``p`` scores close
   to 1 for ``p``; a camera shot, a live demo or a blank screen scores low for all.

The result is a :class:`VisualEvidence` object that the aligner fuses with the
transcript similarity. Recordings without a video stream, or whose picture never shows
the slides (a camera pointed at the lecturer), yield ``None`` and the alignment falls
back to text only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import PathLike, get_logger

logger = get_logger(__name__)

Rect = Tuple[int, int, int, int]  # x, y, width, height in analysis-frame pixels

_SEARCH_PATCH = (32, 24)  # width, height of the patch used while searching for the slide region
_MATCH_PATCH = (192, 144)  # width, height of the patch used for the final similarity


@dataclass
class VisualEvidence:
    """Frame-by-frame similarity between a video and a slide deck.

    Attributes:
        times: ``(T,)`` frame timestamps in seconds, ascending.
        similarity: ``(T, M)`` normalised correlation of every frame with every slide.
        region: Rectangle of the slide inside the analysis frame ``(x, y, w, h)``.
        frame_size: Analysis frame size ``(width, height)``.
        region_score: Quality of the detected region in ``[0, 1]``.
    """

    times: np.ndarray
    similarity: np.ndarray
    region: Rect
    frame_size: Tuple[int, int]
    region_score: float

    @property
    def best(self) -> np.ndarray:
        """Highest correlation of each frame with any slide."""
        return self.similarity.max(axis=1) if self.similarity.size else np.zeros(0)

    @property
    def pages(self) -> np.ndarray:
        """Best-matching slide index for each frame."""
        return self.similarity.argmax(axis=1) if self.similarity.size else np.zeros(0, dtype=np.int64)

    def coverage(self, min_corr: float = 0.8) -> float:
        """Fraction of frames in which a slide is clearly visible."""
        return float((self.best >= min_corr).mean()) if len(self.times) else 0.0

    def segment_scores(
        self,
        starts: Sequence[float],
        ends: Sequence[float],
        min_corr: float = 0.6,
        full_corr: float = 0.8,
        temperature: float = 0.02,
    ) -> np.ndarray:
        """Turn frame evidence into a per-segment, per-slide score in ``[0, 1]``.

        For each frame the correlations are sharpened with ``exp((c - c_max) / temperature)``
        so the best slide scores 1 and look-alike neighbours clearly less, then weighted
        by a confidence gate that is 0 below ``min_corr`` and 1 above ``full_corr``. A
        segment receives the mean over the frames inside its time span. A segment that
        falls between two samples takes the mean of the frame before and the frame after
        it, provided they are close enough in time to bracket it reliably: page turns are
        sampled densely, so two neighbouring frames that agree mean the slide did not
        change in between.

        Returns:
            ``(N, M)`` array; an all-zero row means "no visual evidence for this segment".
        """
        n_pages = self.similarity.shape[1] if self.similarity.ndim == 2 else 0
        scores = np.zeros((len(starts), n_pages), dtype=np.float32)
        if len(self.times) == 0 or n_pages == 0:
            return scores
        best = self.best
        gate = np.clip((best - min_corr) / max(full_corr - min_corr, 1e-6), 0.0, 1.0)
        sharp = np.exp((self.similarity - best[:, None]) / max(temperature, 1e-6)) * gate[:, None]
        gaps = np.diff(self.times)
        # Largest distance between samples that still counts as "bracketing" a segment:
        # a little more than the key-frame spacing of this video.
        max_gap = max(2.0, 1.5 * float(np.percentile(gaps, 90))) if len(gaps) else 2.0
        last = len(self.times) - 1
        for index, (start, end) in enumerate(zip(starts, ends)):
            lo = int(np.searchsorted(self.times, start, side="left"))
            hi = int(np.searchsorted(self.times, end, side="right"))
            if hi > lo:
                scores[index] = sharp[lo:hi].mean(axis=0)
                continue
            before, after = lo - 1, hi  # hi == lo here: no sample inside the segment
            if 0 <= before and after <= last and self.times[after] - self.times[before] <= max_gap:
                scores[index] = 0.5 * (sharp[before] + sharp[after])
            elif 0 <= before and start - self.times[before] <= 0.5 * max_gap:
                scores[index] = sharp[before]
            elif after <= last and self.times[after] - end <= 0.5 * max_gap:
                scores[index] = sharp[after]
        return scores


# --------------------------------------------------------------------------- video decoding


def has_video_stream(path: PathLike) -> bool:
    """True when ``path`` is a media file with at least one video stream."""
    try:
        import av

        with av.open(str(path)) as container:
            return any(stream.type == "video" for stream in container.streams) and bool(container.streams.video)
    except Exception:  # noqa: BLE001 - audio-only files, unsupported containers, missing PyAV
        return False


def read_keyframes(path: PathLike, size: Tuple[int, int] = (640, 360)) -> Tuple[np.ndarray, np.ndarray]:
    """Decode only the key frames of a video as small grey images.

    Args:
        path: Video file.
        size: Output ``(width, height)``; the picture is scaled without cropping.

    Returns:
        ``(times, frames)`` with ``times`` in seconds and ``frames`` of shape
        ``(T, height, width)``, dtype ``uint8``.
    """
    import av

    times: List[float] = []
    frames: List[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            times.append(float(frame.pts * stream.time_base))
            frames.append(frame.to_ndarray(width=size[0], height=size[1], format="gray"))
    if not frames:
        return np.zeros(0, dtype=np.float64), np.zeros((0, size[1], size[0]), dtype=np.uint8)
    order = np.argsort(times)
    return np.asarray(times, dtype=np.float64)[order], np.stack(frames)[order]


def read_frames_between(
    path: PathLike, start: float, end: float, step: float = 1.0, size: Tuple[int, int] = (640, 360)
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode frames in the open interval ``(start, end)``, roughly one every ``step`` seconds.

    Seeks to the key frame at or before ``start`` and decodes forward, so the cost is
    one group of pictures per call.
    """
    import av

    times: List[float] = []
    frames: List[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(start, 0.0) / stream.time_base), stream=stream, backward=True, any_frame=False)
        next_time = start + step
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            now = float(frame.pts * stream.time_base)
            if now >= end - 1e-3:
                break
            if now + 1e-3 >= next_time:
                times.append(now)
                frames.append(frame.to_ndarray(width=size[0], height=size[1], format="gray"))
                next_time = now + step
    if not frames:
        return np.zeros(0, dtype=np.float64), np.zeros((0, size[1], size[0]), dtype=np.uint8)
    return np.asarray(times, dtype=np.float64), np.stack(frames)


# --------------------------------------------------------------------------- image helpers


def render_slides_gray(pdf_path: PathLike, dpi: int = 60) -> Tuple[List[np.ndarray], float]:
    """Render every PDF page as a small grey image.

    Returns:
        ``(images, aspect)`` where ``aspect`` is the page width divided by its height.
    """
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - older PyMuPDF only ships the fitz name
        import fitz as pymupdf

    images: List[np.ndarray] = []
    with pymupdf.open(str(pdf_path)) as document:
        rect = document[0].rect
        aspect = float(rect.width) / max(float(rect.height), 1.0)
        for page in document:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
            images.append(np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width).copy())
    return images, aspect


def _resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


def _unit_rows(stack: np.ndarray) -> np.ndarray:
    """Flatten images to zero-mean, unit-norm rows (featureless images become zero rows)."""
    rows = stack.reshape(len(stack), -1).astype(np.float32)
    rows -= rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    # A (nearly) uniform image has no structure to match: a standard deviation below two
    # grey levels would only amplify compression noise.
    return np.where(norms > 2.0 * np.sqrt(rows.shape[1]), rows / np.maximum(norms, 1e-6), 0.0)


def _crop_patches(frames: np.ndarray, rect: Rect, size: Tuple[int, int]) -> np.ndarray:
    x, y, w, h = rect
    return np.stack([_resize(frame[y : y + h, x : x + w], size) for frame in frames])


def _candidate_rects(frame_w: int, frame_h: int, aspect: float, step: int) -> List[Rect]:
    rects: List[Rect] = []
    for scale in np.linspace(0.4, 1.0, 13):
        h = int(round(frame_h * scale))
        w = int(round(h * aspect))
        if w > frame_w:  # slide wider than the frame: fit the width instead
            w = int(round(frame_w * scale))
            h = int(round(w / aspect))
        if w < 16 or h < 12 or h > frame_h:
            continue
        for y in range(0, frame_h - h + 1, step):
            for x in range(0, frame_w - w + 1, step):
                rects.append((x, y, w, h))
    return list(dict.fromkeys(rects))


def locate_slide_region(
    frames: np.ndarray, slides: Sequence[np.ndarray], aspect: float, max_samples: int = 16
) -> Tuple[Optional[Rect], float]:
    """Find the rectangle of the video frame in which the slides are shown.

    Each candidate rectangle is scored by how well its content matches *some* slide,
    taking the 70th percentile over a sample of frames so that stretches without a slide
    (camera shots, demos) do not veto an otherwise consistent region.

    Args:
        frames: ``(T, H, W)`` grey frames.
        slides: Grey slide renderings.
        aspect: Slide width / height.
        max_samples: Number of frames examined.

    Returns:
        ``(rect, score)``; ``rect`` is ``None`` when there are no frames.
    """
    if len(frames) == 0 or len(slides) == 0:
        return None, 0.0
    frame_h, frame_w = frames.shape[1:]
    picks = np.unique(np.linspace(0.08 * (len(frames) - 1), 0.92 * (len(frames) - 1), min(max_samples, len(frames))).astype(int))
    sample = frames[picks]
    slide_rows = _unit_rows(np.stack([_resize(slide, _SEARCH_PATCH) for slide in slides]))

    def score(rects: Sequence[Rect]) -> np.ndarray:
        out = np.zeros(len(rects), dtype=np.float32)
        for index, rect in enumerate(rects):
            corr = _unit_rows(_crop_patches(sample, rect, _SEARCH_PATCH)) @ slide_rows.T
            out[index] = float(np.percentile(corr.max(axis=1), 70))
        return out

    coarse_step = max(4, frame_w // 40)
    rects = _candidate_rects(frame_w, frame_h, aspect, coarse_step)
    scores = score(rects)
    best_rect, best_score = rects[int(scores.argmax())], float(scores.max())

    # Refine position and size around the coarse winner.
    x0, y0, w0, h0 = best_rect
    fine: List[Rect] = []
    for dh in range(-coarse_step, coarse_step + 1, 2):
        h = h0 + dh
        w = int(round(h * aspect))
        if h < 12 or h > frame_h or w > frame_w:
            continue
        for dy in range(-coarse_step, coarse_step + 1, 2):
            for dx in range(-coarse_step, coarse_step + 1, 2):
                x, y = x0 + dx, y0 + dy
                if x >= 0 and y >= 0 and x + w <= frame_w and y + h <= frame_h:
                    fine.append((x, y, w, h))
    if fine:
        fine_scores = score(fine)
        if float(fine_scores.max()) > best_score:
            best_rect, best_score = fine[int(fine_scores.argmax())], float(fine_scores.max())
    return best_rect, best_score


def frame_slide_similarity(frames: np.ndarray, rect: Rect, slides: Sequence[np.ndarray]) -> np.ndarray:
    """Normalised correlation ``(T, M)`` between the cropped frames and the slides."""
    if len(frames) == 0:
        return np.zeros((0, len(slides)), dtype=np.float32)
    slide_rows = _unit_rows(np.stack([_resize(slide, _MATCH_PATCH) for slide in slides]))
    return _unit_rows(_crop_patches(frames, rect, _MATCH_PATCH)) @ slide_rows.T


# --------------------------------------------------------------------------- public entry point


def match_video_to_slides(
    video_path: PathLike, pdf_path: PathLike, config: Optional[Mapping[str, Any]] = None
) -> Optional[VisualEvidence]:
    """Compare a lecture video with a slide deck.

    Args:
        video_path: Recording; files without a video stream return ``None``.
        pdf_path: Slide deck.
        config: The ``visual`` section of the configuration.

    Returns:
        :class:`VisualEvidence`, or ``None`` when the file has no picture or the picture
        never shows the slides (region score below ``min_region_score``).
    """
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", True)) or not has_video_stream(video_path):
        return None
    size = (int(cfg.get("frame_width", 640)), int(cfg.get("frame_height", 360)))
    min_region_score = float(cfg.get("min_region_score", 0.75))
    min_visible = float(cfg.get("min_visible_fraction", 0.05))
    confident = float(cfg.get("full_corr", 0.8))
    refine_step = float(cfg.get("refine_step_sec", 1.0))
    max_refinements = int(cfg.get("max_refinements", 400))

    times, frames = read_keyframes(video_path, size)
    if len(times) < 2:
        return None
    slides, aspect = render_slides_gray(pdf_path)
    rect, region_score = locate_slide_region(frames, slides, aspect)
    if rect is None or region_score < min_region_score:
        logger.info(
            "Video does not show the slides (best region score %.2f < %.2f); using transcript only",
            region_score, min_region_score,
        )  # fmt: skip
        return None
    similarity = frame_slide_similarity(frames, rect, slides)
    visible = float((similarity.max(axis=1) >= confident).mean())
    if visible < min_visible:
        logger.info("Slides are visible in only %.0f%% of the video; using transcript only", 100 * visible)
        return None

    # Refine page turns: decode between key frames that show different slides.
    if refine_step > 0:
        pages, best = similarity.argmax(axis=1), similarity.max(axis=1)
        changes = [
            k for k in range(len(times) - 1)
            if pages[k] != pages[k + 1] and max(best[k], best[k + 1]) >= confident and times[k + 1] - times[k] > 1.5 * refine_step
        ]  # fmt: skip
        extra_times: List[np.ndarray] = []
        extra_sims: List[np.ndarray] = []
        for k in changes[:max_refinements]:
            between_times, between_frames = read_frames_between(video_path, times[k], times[k + 1], refine_step, size)
            if len(between_times):
                extra_times.append(between_times)
                extra_sims.append(frame_slide_similarity(between_frames, rect, slides))
        if extra_times:
            times = np.concatenate([times, *extra_times])
            similarity = np.concatenate([similarity, *extra_sims])
            order = np.argsort(times, kind="stable")
            times, similarity = times[order], similarity[order]

    evidence = VisualEvidence(times, similarity.astype(np.float32), rect, size, region_score)
    logger.info(
        "Visual matching: %d frames, slide region %s in %dx%d (score %.2f), slide visible in %.0f%% of frames",
        len(times), rect, size[0], size[1], region_score, 100 * evidence.coverage(confident),
    )  # fmt: skip
    return evidence


def describe(evidence: Optional[VisualEvidence]) -> dict:
    """JSON-friendly summary for result metadata."""
    if evidence is None:
        return {"used": False}
    return {
        "used": True,
        "frames": int(len(evidence.times)),
        "region": list(evidence.region),
        "frame_size": list(evidence.frame_size),
        "region_score": round(evidence.region_score, 3),
        "slide_visible_fraction": round(evidence.coverage(0.8), 3),
    }
