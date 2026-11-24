"""End-to-end smoke tests for LectureSync-ASR.

Everything the tests need is generated on the fly:

* a two-page computer-science slide deck (PyMuPDF),
* a short speech-shaped WAV file (NumPy + soundfile),
* on macOS additionally a real spoken lecture synthesised with the system ``say`` command.

The suite then drives ``VAD -> ASR -> Aligner -> Vocab`` on the auto-detected device
(MPS on Apple Silicon, CUDA, or CPU) and checks output schemas, the monotonicity of the
alignment and that no tensor ends up on an unexpected device.

Run with ``pytest tests/test_pipeline.py -v`` or ``python tests/test_pipeline.py``.
Whisper-tiny is used by default to keep the run short; set ``LECTURESYNC_TEST_MODEL`` to
test another checkpoint. Models are downloaded on first use; tests that need a model are
skipped when it is neither cached nor downloadable.
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from src.aligner import SlideAligner, SlideDeck, attach_pages, jump_penalty, monotonic_dp_align  # noqa: E402
from src.asr import WhisperASR  # noqa: E402
from src.pipeline import LecturePipeline, LectureResult, export_markdown  # noqa: E402
from src.utils import SAMPLE_RATE, DeviceManager, format_timestamp, load_audio, load_config, normalize_text  # noqa: E402
from src.vad import SpeechSegment, VoiceActivityDetector  # noqa: E402
from src.visual import has_video_stream, match_video_to_slides  # noqa: E402
from src.vocab import TermExtractor, clean_slide_texts, highlight_terms, term_recall  # noqa: E402

TEST_MODEL = os.environ.get("LECTURESYNC_TEST_MODEL", "openai/whisper-tiny")

SLIDES = [
    (
        "Gradient Descent and Backpropagation",
        [
            "Loss function and objective",
            "Gradient descent update rule",
            "Learning rate and convergence",
            "Backpropagation uses the chain rule",
        ],
    ),
    (
        "Homography and Eigenvalues",
        [
            "Homography maps points between two planes",
            "Estimate with RANSAC from keypoint matches",
            "Singular value decomposition",
            "Eigenvalue and eigenvector of the covariance matrix",
        ],
    ),
]

NARRATION = [
    "Good morning everyone. Today we will talk about how neural networks are trained. "
    "We define a loss function, and then we minimize it using gradient descent. "
    "The learning rate controls how big each step is. "
    "To compute the gradients, we use backpropagation, which is the chain rule applied layer by layer.",
    "For the second topic today, we switch to geometry. "
    "A homography is a matrix that maps points from one plane to another plane. "
    "We estimate it using RANSAC on keypoint matches, and we solve the linear system "
    "with the singular value decomposition of the covariance matrix.",
]

# --------------------------------------------------------------------------- generators


def make_slides_pdf(path: Path) -> Path:
    """Write a two-page mock CS slide deck with PyMuPDF."""
    import pymupdf

    document = pymupdf.open()
    for title, bullets in SLIDES:
        page = document.new_page(width=960, height=540)
        page.insert_text((60, 90), title, fontsize=34)
        for row, bullet in enumerate(bullets):
            page.insert_text((80, 170 + 55 * row), f"- {bullet}", fontsize=22)
    document.save(str(path))
    document.close()
    return path


def speech_like(seconds: float, f0: float = 120.0, seed: int = 0) -> np.ndarray:
    """Harmonic stack with syllable-rate amplitude modulation: loud, voiced, speech-shaped."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    pitch = f0 + 15.0 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    voiced = sum(np.sin(k * phase) / k for k in range(1, 10))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    return (0.2 * voiced * envelope + 0.002 * rng.standard_normal(t.shape)).astype(np.float32)


def silence(seconds: float, seed: int = 1) -> np.ndarray:
    """Near-silent room tone."""
    rng = np.random.default_rng(seed)
    return (0.0005 * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def make_synthetic_wav(path: Path) -> Path:
    """Two utterances separated by a 1.5 s pause; the first contains a 0.4 s hesitation.

    Expected VAD behaviour: the 0.4 s gap (< 0.8 s) is merged, the 1.5 s gap splits,
    giving exactly two speech segments.
    """
    audio = np.concatenate(
        [silence(0.5), speech_like(1.6), silence(0.4, 2), speech_like(1.4, 130.0, 3),
         silence(1.5, 4), speech_like(2.6, 110.0, 5), silence(0.5, 6)]
    )  # fmt: skip
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
    return path


def synthesize_speech(path: Path) -> Path:
    """Render the narration with the macOS ``say`` command (1.5 s pause between slides)."""
    parts: List[np.ndarray] = []
    for index, paragraph in enumerate(NARRATION):
        part_path = path.with_name(f"say_{index}.wav")
        subprocess.run(
            ["say", "-o", str(part_path), "--file-format=WAVE", f"--data-format=LEI16@{SAMPLE_RATE}", paragraph],
            check=True, capture_output=True, timeout=120,
        )  # fmt: skip
        audio, sr = sf.read(str(part_path), dtype="float32")
        assert sr == SAMPLE_RATE
        parts += [audio, np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)]
    sf.write(str(path), np.concatenate(parts), SAMPLE_RATE, subtype="PCM_16")
    return path


# A deck whose middle pages share every word and differ only in a drawing: the case text
# alignment cannot solve. ``None`` in the timeline is a stretch where the video shows
# something else (camera, demo), and page 1 is revisited to exercise backward moves.
ANIMATION_TIMELINE = [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (1, 4), (None, 4), (5, 4)]
VIDEO_FPS = 2
VIDEO_SIZE = (640, 360)
SLIDE_RECT = (106, 20, 427, 320)  # x, y, w, h of the 4:3 slide inside the 16:9 frame


def make_animation_deck(path: Path) -> Path:
    """Six 4:3 pages; pages 2-4 have identical text and a shape in a different place."""
    import pymupdf

    document = pymupdf.open()
    layouts = [
        ("Linear filtering", ["A kernel slides over the image", "Each output pixel is a weighted sum"], None),
        ("Border handling", ["Pad with zeros", "Replicate the edge pixels"], (80, 330, 300, 480)),
        ("Worked example", ["Apply the kernel step by step"], (60, 220, 200, 360)),
        ("Worked example", ["Apply the kernel step by step"], (290, 220, 430, 360)),
        ("Worked example", ["Apply the kernel step by step"], (520, 220, 660, 360)),
        ("Summary", ["Filters are small kernels", "Borders need care"], (420, 380, 680, 500)),
    ]
    for index, (title, bullets, box) in enumerate(layouts):
        page = document.new_page(width=720, height=540)
        page.insert_text((50, 80), title, fontsize=36)
        for row, bullet in enumerate(bullets):
            page.insert_text((70, 140 + 44 * row), f"- {bullet}", fontsize=22)
        if box is not None:
            page.draw_rect(pymupdf.Rect(*box), color=(0, 0, 0), fill=(0.15 + 0.12 * index, 0.3, 0.75 - 0.1 * index), width=2)
            page.draw_circle(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), 28, color=(1, 1, 1), fill=(1, 1, 1))
    document.save(str(path))
    document.close()
    return path


def make_slide_video(path: Path, pdf_path: Path, with_audio: bool = False) -> Path:
    """Encode ``ANIMATION_TIMELINE`` as a pillar-boxed screen recording (MPEG-4, key frame every second)."""
    import av
    import pymupdf
    from PIL import Image

    x, y, w, h = SLIDE_RECT
    rng = np.random.default_rng(0)
    with pymupdf.open(str(pdf_path)) as document:
        renders = []
        for page in document:
            pixmap = page.get_pixmap(dpi=72, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples).resize((w, h), Image.BILINEAR)
            renders.append(np.asarray(image))

    with av.open(str(path), mode="w") as container:
        video = container.add_stream("mpeg4", rate=VIDEO_FPS)
        video.width, video.height = VIDEO_SIZE
        video.pix_fmt = "yuv420p"
        video.bit_rate = 2_000_000
        video.gop_size = VIDEO_FPS
        audio = None
        if with_audio:
            audio = container.add_stream("aac", rate=SAMPLE_RATE)
            audio.layout = "mono"

        frame_index = 0
        for page, seconds in ANIMATION_TIMELINE:
            for _ in range(seconds * VIDEO_FPS):
                canvas = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
                if page is None:  # "camera": smooth blobs unrelated to any slide
                    blobs = rng.integers(0, 255, (9, 16, 3), dtype=np.uint8)
                    canvas = np.asarray(Image.fromarray(blobs).resize(VIDEO_SIZE, Image.BICUBIC)).copy()
                else:
                    canvas[y : y + h, x : x + w] = renders[page]
                frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                frame.pts = frame_index
                frame_index += 1
                for packet in video.encode(frame):
                    container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)

        if audio is not None:
            total = sum(seconds for _, seconds in ANIMATION_TIMELINE)
            wave = np.concatenate([speech_like(3.0, 110.0 + 5 * k, k) if k % 2 == 0 else silence(1.0, k) for k in range(total // 2)])
            wave = np.pad(wave, (0, max(0, total * SAMPLE_RATE - len(wave))))[: total * SAMPLE_RATE]
            for offset in range(0, len(wave), 1024):
                chunk = wave[offset : offset + 1024]
                if len(chunk) < 1024:
                    chunk = np.pad(chunk, (0, 1024 - len(chunk)))
                audio_frame = av.AudioFrame.from_ndarray(chunk[None, :].astype(np.float32), format="fltp", layout="mono")
                audio_frame.sample_rate = SAMPLE_RATE
                audio_frame.pts = offset
                for packet in audio.encode(audio_frame):
                    container.mux(packet)
            for packet in audio.encode(None):
                container.mux(packet)
    return path


def timeline_page(time_sec: float):
    """Ground-truth page shown at ``time_sec`` (``None`` during the camera stretch)."""
    cursor = 0.0
    for page, seconds in ANIMATION_TIMELINE:
        if cursor <= time_sec < cursor + seconds:
            return page
        cursor += seconds
    return ANIMATION_TIMELINE[-1][0]


def model_or_skip(action, what: str):
    """Run ``action`` and skip the test if model weights cannot be obtained (offline, no cache)."""
    try:
        return action()
    except OSError as exc:  # huggingface_hub raises OSError subclasses for missing files / no network
        pytest.skip(f"{what} is not cached and could not be downloaded: {exc}")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("lecturesync")


@pytest.fixture(scope="session")
def config() -> Dict[str, Any]:
    cfg = load_config()
    cfg["asr"]["model_name"] = TEST_MODEL
    cfg["asr"]["lora_checkpoint"] = None
    cfg["runtime"]["log_level"] = "WARNING"
    return cfg


@pytest.fixture(scope="session")
def slides_pdf(workdir: Path) -> Path:
    return make_slides_pdf(workdir / "mock_cs_lecture.pdf")


@pytest.fixture(scope="session")
def synthetic_wav(workdir: Path) -> Path:
    return make_synthetic_wav(workdir / "synthetic.wav")


@pytest.fixture(scope="session")
def animation_deck(workdir: Path) -> Path:
    return make_animation_deck(workdir / "animation_deck.pdf")


@pytest.fixture(scope="session")
def slide_video(workdir: Path, animation_deck: Path) -> Path:
    return make_slide_video(workdir / "screen_recording.mp4", animation_deck)


@pytest.fixture(scope="session")
def device_manager(config: Dict[str, Any]) -> DeviceManager:
    return DeviceManager.from_config(config)


@pytest.fixture(scope="session")
def pipeline(config: Dict[str, Any], device_manager: DeviceManager) -> LecturePipeline:
    pipe = LecturePipeline(copy.deepcopy(config), device_manager)
    model_or_skip(pipe.asr.load, TEST_MODEL)
    model_or_skip(pipe.aligner.load, pipe.aligner.model_name)
    return pipe


# --------------------------------------------------------------------------- unit tests (no models)


def test_device_manager_picks_a_usable_device(device_manager: DeviceManager) -> None:
    assert device_manager.device.type in {"mps", "cuda", "cpu"}
    assert device_manager.dtype in {torch.float32, torch.float16}
    if device_manager.device.type in {"mps", "cpu"}:
        assert device_manager.dtype == torch.float32  # the safe default away from CUDA
    x = torch.randn(4, 4, device=device_manager.device, dtype=device_manager.dtype)
    y = x @ x.T
    device_manager.synchronize()  # must not raise on any backend
    assert y.device.type == device_manager.device.type
    assert isinstance(device_manager.probe_half_precision(), bool)
    assert DeviceManager("cpu", "float16").dtype == torch.float32  # fp16 request on CPU falls back


def test_generated_fixtures(slides_pdf: Path, synthetic_wav: Path, workdir: Path) -> None:
    deck = SlideDeck.from_pdf(slides_pdf, image_dir=workdir / "slides_check", dpi=72)
    assert len(deck) == 2
    assert deck.pages[0].title == SLIDES[0][0]
    assert "RANSAC" in deck.pages[1].text
    assert all(page.image_path is not None and page.image_path.stat().st_size > 1000 for page in deck.pages)

    audio = load_audio(synthetic_wav)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) / SAMPLE_RATE - 8.5) < 0.05
    assert float(np.abs(audio).max()) <= 1.0


def test_audio_loader_resamples(workdir: Path) -> None:
    path = workdir / "tone_44k.wav"
    t = np.arange(44_100) / 44_100
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1).astype(np.float32)
    sf.write(str(path), 0.5 * stereo, 44_100)
    audio = load_audio(path)
    assert abs(len(audio) - SAMPLE_RATE) <= 2
    assert 0.3 < float(np.abs(audio).max()) <= 1.0


def test_vad_merges_short_pauses_and_splits_long_ones(config: Dict[str, Any], synthetic_wav: Path) -> None:
    vad = VoiceActivityDetector(dict(config["vad"], backend="energy"))
    segments = vad.segment_file(synthetic_wav)
    assert vad.backend == "energy"
    assert len(segments) == 2, [(s.start_sec, s.end_sec) for s in segments]
    first, second = segments
    assert first.start_sec < 0.7 and 3.7 < first.end_sec < 4.2  # the 0.4 s hesitation was merged
    assert 5.2 < second.start_sec < 5.6
    for seg in segments:
        assert isinstance(seg, SpeechSegment)
        assert seg.audio_tensor.device.type == "cpu" and seg.audio_tensor.dtype == torch.float32
        assert abs(seg.audio_tensor.numel() / SAMPLE_RATE - seg.duration) < 1e-3


def test_vad_caps_segments_at_30_seconds(config: Dict[str, Any]) -> None:
    vad = VoiceActivityDetector(dict(config["vad"], backend="energy"))
    segments = vad.segment(speech_like(70.0))
    assert len(segments) >= 3
    assert max(seg.duration for seg in segments) <= 30.0 + 1e-6
    assert sum(seg.duration for seg in segments) > 65.0  # nothing was dropped, only split
    for earlier, later in zip(segments, segments[1:]):
        assert earlier.end_sec <= later.start_sec + 1e-6


def test_vad_silero_backend_or_graceful_fallback(config: Dict[str, Any], synthetic_wav: Path) -> None:
    vad = VoiceActivityDetector(config["vad"])
    segments = vad.segment_file(synthetic_wav)  # must never raise, online or offline
    assert vad.backend in {"silero_hub_cache", "silero_hub", "silero_pip", "energy"}
    assert all(seg.duration <= 30.0 for seg in segments)


def test_monotonic_dp_alignment_is_monotonic_and_robust() -> None:
    # 9 segments, 3 pages. Segment 2 is a distractor that "sounds like" page 2, and segment 6
    # looks like page 0 again; a per-row argmax would jump forward and then go backwards.
    similarity = np.array(
        [
            [0.80, 0.10, 0.05],
            [0.70, 0.20, 0.10],
            [0.30, 0.10, 0.55],  # distractor
            [0.75, 0.15, 0.05],
            [0.20, 0.70, 0.10],
            [0.10, 0.80, 0.15],
            [0.60, 0.40, 0.10],  # refers back to an earlier slide
            [0.10, 0.20, 0.85],
            [0.05, 0.15, 0.90],
        ]
    )
    greedy = similarity.argmax(axis=1).tolist()
    assert greedy != sorted(greedy)  # the greedy path really is non-monotonic
    path = monotonic_dp_align(similarity, jump_lambda=0.12, skip_penalty=0.5)
    assert path == [0, 0, 0, 0, 1, 1, 1, 2, 2]
    assert all(a <= b for a, b in zip(path, path[1:]))

    # A large penalty keeps everything on one page; zero penalty still may not go backwards.
    assert len(set(monotonic_dp_align(similarity, jump_lambda=10.0))) == 1
    free = monotonic_dp_align(similarity, jump_lambda=0.0, skip_penalty=0.0)
    assert all(a <= b for a, b in zip(free, free[1:]))

    # Brute-force check of optimality on a small random problem.
    rng = np.random.default_rng(0)
    small = rng.random((6, 3))
    lam, skip = 0.2, 0.5

    def objective(pages: List[int]) -> float:
        total, previous = 0.0, 0
        for row, page in enumerate(pages):
            total += small[row, page] - lam * float(jump_penalty(np.array(page - previous), skip))
            previous = page
        return total

    import itertools

    candidates = [list(p) for p in itertools.product(range(3), repeat=6) if list(p) == sorted(p)]
    best = max(candidates, key=objective)
    assert abs(objective(monotonic_dp_align(small, lam, skip)) - objective(best)) < 1e-9

    assert monotonic_dp_align(np.zeros((0, 3))) == []
    assert monotonic_dp_align(np.array([[0.1, 0.9]]), jump_lambda=0.1) == [1]


def test_vocab_extraction_highlighting_and_recall(config: Dict[str, Any]) -> None:
    page_texts = ["\n".join([title, *bullets]) for title, bullets in SLIDES]
    extractor = TermExtractor(config["vocab"], custom_terms=["ResNet"])
    glossary = extractor.extract_glossary(page_texts)
    assert glossary[0] == "ResNet"  # user terms come first
    for term in ("gradient descent", "backpropagation", "homography", "RANSAC", "eigenvalue"):
        assert term in glossary

    segments = [
        {"id": 0, "start": 0.0, "end": 6.0, "page": 0,
         "text": "We minimize the loss function using gradient descent with a small learning rate."},
        {"id": 1, "start": 6.0, "end": 12.0, "page": 0,
         "text": "Back propagation is the chain rule. This is a robust and scalable approach."},
        {"id": 2, "start": 12.0, "end": 20.0, "page": 1,
         "text": "A homography is estimated with RANSAC and the singular value decomposition."},
    ]  # fmt: skip
    entries = {entry.term: entry for entry in extractor.extract(segments, page_texts)}
    descent = entries["gradient descent"]
    assert descent.category == "cs_ai" and descent.count == 1 and descent.slide_pages == [0]
    assert descent.occurrences[0].page == 0 and 0.0 < descent.occurrences[0].time < 6.0
    assert "gradient descent" in descent.occurrences[0].context
    assert entries["backpropagation"].occurrences[0].segment_id == 1  # matched through the "back propagation" alias
    assert entries["homography"].occurrences[0].page == 1
    assert entries["robust"].category == "academic"
    assert "gradient" not in entries  # only ever heard inside "gradient descent"
    assert entries["eigenvalue"].occurrences == [] and entries["eigenvalue"].slide_pages == [1]  # slide-only term

    assert highlight_terms("Use <gradient descent>.", ["gradient descent"]) == (
        'Use &lt;<mark class="ls-term">gradient descent</mark>&gt;.'
    )
    assert highlight_terms("Back-propagation works.", ["backpropagation"], style="markdown") == "**Back-propagation** works."
    recall = term_recall(["gradient descent and eigenvalues"], ["gradient decent and eigenvalue"], ["gradient descent", "eigenvalue"])
    assert recall["matched"] == 1 and recall["total"] == 2 and recall["recall"] == 0.5
    assert normalize_text("Back-Propagation, it's GREAT!") == "back propagation it's great"
    assert format_timestamp(3725) == "1:02:05" and format_timestamp(65.4) == "01:05"


def test_slide_text_cleaning_and_glossary_ranking(config: Dict[str, Any]) -> None:
    """Footers, URLs and label fragments must not reach the Whisper prompt."""
    footer = "CS101 Machine Perception | Dr Ada Example"
    pages = [
        f"Linear filtering\nA kernel slides over the image\nSee https://example.org/filters?id=3\n{footer}",
        f"Linear filtering\nBorder handling: x1 x0 pad with 0\nExplain border handling choices\n{footer}",
        f"Separable filters\nBorder handling again\nMail ada@example.org for help, 12 KB\n{footer}",
        f"Summary\nSeparable filters are cheap\n{footer}",
        f"Questions\n{footer}",
    ]
    cleaned = clean_slide_texts(pages)
    assert all(footer not in page for page in cleaned)  # repeated on every page -> boilerplate
    assert "example.org" not in " ".join(cleaned) and "Linear filtering" in cleaned[0]
    assert clean_slide_texts(pages[:2])[0].endswith(footer)  # too few pages to call anything boilerplate

    glossary = TermExtractor(config["vocab"]).extract_glossary(pages)
    lowered = [term.lower() for term in glossary]
    assert "kernel" in lowered and "border handling" in lowered and "separable filters" in lowered
    for junk in ("http", "x1", "x0", "kb", "ada example", "cs101 machine perception", "explain border handling choices"):
        assert junk not in lowered, (junk, glossary)
    assert not any("example" in term or "http" in term for term in lowered)


def test_visual_matching_identifies_the_slide_on_screen(
    config: Dict[str, Any], animation_deck: Path, slide_video: Path, synthetic_wav: Path
) -> None:
    """Frames of a synthetic screen recording are matched to the right page, look-alikes included."""
    assert has_video_stream(slide_video) and not has_video_stream(synthetic_wav)
    assert match_video_to_slides(synthetic_wav, animation_deck, config["visual"]) is None  # audio only

    evidence = match_video_to_slides(slide_video, animation_deck, config["visual"])
    assert evidence is not None and evidence.similarity.shape[1] == 6
    x, y, w, h = evidence.region
    assert abs(x - SLIDE_RECT[0]) <= 8 and abs(y - SLIDE_RECT[1]) <= 8 and abs(h - SLIDE_RECT[3]) <= 12
    assert evidence.region_score > 0.9

    checked = 0
    for time_sec, page, best in zip(evidence.times, evidence.pages, evidence.best):
        truth = timeline_page(float(time_sec))
        boundary = min(abs(float(time_sec) % 4.0), 4.0 - float(time_sec) % 4.0) < 0.6  # a page turn is in progress
        if truth is None:
            assert boundary or best < 0.6, f"camera frame at {time_sec:.1f}s matched slide {page} with {best:.2f}"
        elif not boundary:
            assert page == truth and best > 0.9, f"{time_sec:.1f}s: got {page} ({best:.2f}), expected {truth}"
            checked += 1
    assert checked >= 20
    assert 0.7 < evidence.coverage() < 0.95  # everything except the 4 s camera stretch

    # Per-segment scores: decisive where a slide is shown, silent during the camera stretch.
    scores = evidence.segment_scores([8.5, 25.0], [10.5, 27.0])
    assert scores[0].argmax() == 2 and scores[0].max() > 0.9 and scores[1].max() == 0.0

    # A deck that is not the one in the video must be rejected rather than force-matched.
    other = make_slides_pdf(animation_deck.with_name("other_deck.pdf"))
    assert match_video_to_slides(slide_video, other, config["visual"]) is None


def test_dp_backward_moves_are_opt_in() -> None:
    similarity = np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9], [0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1]])
    strict = monotonic_dp_align(similarity, 0.1, 0.5)
    assert all(a <= b for a, b in zip(strict, strict[1:]))
    assert monotonic_dp_align(similarity, 0.1, 0.5, backward_penalty=2.0) == [0, 2, 0, 0, 1]
    # Relaxed transitions only into flagged segments: the move back at index 2 is not allowed.
    gated = monotonic_dp_align(similarity, 0.1, 0.5, backward_penalty=2.0, relaxed_steps=[True, True, False, False, True])
    assert gated[2] >= gated[1] and gated[3] >= gated[2]
    with pytest.raises(ValueError):
        monotonic_dp_align(similarity, 0.1, 0.5, backward_penalty=2.0, relaxed_steps=[True])


def test_hallucination_filter() -> None:
    """Decoder loops and low-confidence stock phrases are removed; real speech is untouched."""
    asr = WhisperASR({"filter_hallucinations": True}, DeviceManager("cpu"))  # no model needed for the filter

    def item(text: str, confidence: float = 0.9, start: float = 0.0) -> Dict[str, Any]:
        return {"start": start, "end": start + 1.0, "text": text, "confidence": confidence}

    assert asr._drop_hallucinations([item("Thank you.", 0.95)]) == []  # a chunk that is only a stock phrase
    assert asr._drop_hallucinations([item("Thank you.", 0.9, i) for i in range(20)]) == []  # looped stock phrase
    assert asr._drop_hallucinations([item("...", 0.3)]) == []

    mixed = [item("So the kernel is flipped."), item("Thank you.", 0.4, 1.0), item("Any questions?", 0.8, 2.0)]
    assert [x["text"] for x in asr._drop_hallucinations(mixed)] == ["So the kernel is flipped.", "Any questions?"]

    polite = [item("That is all for today."), item("Thank you.", 0.93, 1.0)]
    assert asr._drop_hallucinations(polite) == polite  # clearly spoken, next to real content -> kept

    loop = [item("and then we add the bias,", 0.8, i) for i in range(7)] + [item("which gives the output.", 0.9, 7.0)]
    assert [x["text"] for x in asr._drop_hallucinations(loop)] == ["and then we add the bias,", "which gives the output."]

    twice = [item("No.", 0.9), item("No.", 0.9, 1.0), item("That is wrong.", 0.9, 2.0)]
    assert asr._drop_hallucinations(twice) == twice  # two repeats are normal speech

    asr.filter_hallucinations = False
    assert asr._drop_hallucinations([item("Thank you.", 0.2)]) != []


# --------------------------------------------------------------------------- model-backed tests


def assert_transcript_schema(segments: List[Dict[str, Any]], duration: float) -> None:
    for index, seg in enumerate(segments):
        assert set(seg) >= {"id", "start", "end", "text", "confidence"}
        assert seg["id"] == index
        assert isinstance(seg["text"], str) and seg["text"].strip()
        assert 0.0 <= seg["start"] <= seg["end"] <= duration + 0.5
        assert 0.0 <= seg["confidence"] <= 1.0
    starts = [seg["start"] for seg in segments]
    assert starts == sorted(starts)


def test_asr_timestamp_parsing_and_devices(pipeline: LecturePipeline, device_manager: DeviceManager) -> None:
    asr = pipeline.asr
    assert next(asr.model.parameters()).device.type == device_manager.device.type
    assert next(asr.model.parameters()).dtype == device_manager.dtype

    features, mask = asr.extract_features([speech_like(2.0), speech_like(1.0)])
    assert features.device.type == device_manager.device.type and features.dtype == device_manager.dtype
    assert mask.device.type == device_manager.device.type and features.shape[0] == 2

    begin = asr._timestamp_begin
    tokens = [begin, 400, 401, begin + 100, begin + 100, 402, begin + 150, 403]
    pieces = asr._split_on_timestamps(tokens, [-0.1] * len(tokens), duration=4.0)
    assert [(round(s, 2), round(e, 2), toks) for s, e, toks, _ in pieces] == [
        (0.0, 2.0, [400, 401]), (2.0, 3.0, [402]), (3.0, 4.0, [403]),
    ]  # fmt: skip
    prefix = [asr._sot_id, *asr._prefix_ids[1:]]
    assert asr._content_tokens([*prefix, begin, 400, asr._eos_id, asr._eos_id]) == [begin, 400]
    assert asr._content_tokens([begin, 400, asr._eos_id]) == [begin, 400]  # prefix already stripped

    prompt = asr.set_glossary(["homography", "eigenvalue", "homography"])
    assert "homography" in prompt and prompt.count("homography") == 1
    assert asr._prompt_ids is not None and asr._prompt_ids.device.type == device_manager.device.type
    assert asr.set_glossary([]) == "" and asr._prompt_ids is None


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_end_to_end_on_synthetic_audio(
    device: str, config: Dict[str, Any], pipeline: LecturePipeline, slides_pdf: Path, synthetic_wav: Path, workdir: Path
) -> None:
    """VAD -> ASR -> Aligner -> Vocab on generated inputs, on the accelerator and on CPU."""
    if device == "auto":
        pipe = pipeline
    else:
        cfg = copy.deepcopy(config)
        cfg["runtime"]["device"] = "cpu"
        pipe = LecturePipeline(cfg)
        model_or_skip(pipe.asr.load, TEST_MODEL)
    pipe.vad = VoiceActivityDetector(dict(config["vad"], backend="energy"))  # deterministic segments for a synthetic signal

    result = pipe.run(synthetic_wav, slides_pdf, glossary_terms=["ResNet"], work_dir=workdir / f"e2e_{device}")
    assert isinstance(result, LectureResult)
    expected = pipe.device_manager.device.type
    assert next(pipe.asr.model.parameters()).device.type == expected
    assert pipe.aligner.encode(["gradient descent"]).device.type == expected
    assert result.meta["speech_chunks"] == 2 and result.meta["vad_backend"] == "energy"
    assert abs(result.duration - 8.5) < 0.05
    assert result.audio_path.is_file() and len(result.deck) == 2
    assert all(page.image_path is not None and page.image_path.is_file() for page in result.deck.pages)
    assert result.glossary[0] == "ResNet" and "gradient descent" in result.glossary
    assert "gradient descent" in result.meta["prompt"]
    assert result.meta["visual"] == {"used": False, "segment_coverage": 0.0} and result.meta["alignment_mode"] == "text"

    assert_transcript_schema(result.segments, result.duration)
    pages = [seg["page"] for seg in result.segments]
    assert all(0 <= p < 2 for p in pages) and pages == sorted(pages)
    assert len(result.page_ranges) == 2
    # Slide-only vocabulary is available even when the synthetic "speech" yields no real words.
    assert any(entry.term == "homography" and 1 in entry.slide_pages for entry in result.vocabulary)
    assert set(result.timings) >= {"vad_sec", "asr_sec", "align_sec", "vocab_sec", "total_sec"}

    notes = export_markdown(result, workdir / f"e2e_{device}" / "notes.md")
    assert "## Slide 1: Gradient Descent and Backpropagation" in notes
    assert "## Slide 2: Homography and Eigenvalues" in notes and "## Glossary index" in notes
    assert isinstance(result.to_dict()["segments"], list)


def test_alignment_and_vocab_on_scripted_transcript(pipeline: LecturePipeline, slides_pdf: Path, config: Dict[str, Any]) -> None:
    """Real MiniLM embeddings + DP on a known transcript: page turn found at the right sentence."""
    deck = SlideDeck.from_pdf(slides_pdf)
    sentences = [
        "Okay, let's get started.",
        "We define a loss function and minimize it with gradient descent.",
        "The learning rate controls the step size, and backpropagation gives us the gradients.",
        "Later today we will also see a homography, but first, any questions?",  # mentions a later slide
        "If the learning rate is too large, training does not converge.",
        "Remember that backpropagation is just the chain rule.",
        "Now we switch to geometry: a homography maps points between two planes.",
        "We estimate it with RANSAC on keypoint matches.",
        "Finally we take the singular value decomposition and look at the eigenvalues.",
    ]
    segments = [
        {"id": i, "start": 5.0 * i, "end": 5.0 * i + 4.5, "text": text, "confidence": 0.9}
        for i, text in enumerate(sentences)
    ]
    alignment = pipeline.aligner.align(segments, deck)
    assert alignment.similarity.shape == (9, 2)
    greedy = alignment.similarity.argmax(axis=1).tolist()
    assert greedy[3] == 1 and greedy != sorted(greedy)  # per-segment argmax flips to slide 2 and back
    assert alignment.assignments == [0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert alignment.page_ranges[0] == (0.0, 29.5) and alignment.page_ranges[1] == (30.0, 44.5)

    aligned = attach_pages(segments, alignment.assignments)
    entries = {e.term: e for e in TermExtractor(config["vocab"]).extract(aligned, deck.texts)}
    assert entries["RANSAC"].occurrences[0].page == 1 and 35.0 <= entries["RANSAC"].first_time <= 39.5
    assert entries["learning rate"].count == 2 and entries["learning rate"].occurrences[0].page == 0
    assert [occ.page for occ in entries["homography"].occurrences] == [0, 1]  # mentioned early, explained later


def test_visual_evidence_resolves_look_alike_slides(
    pipeline: LecturePipeline, config: Dict[str, Any], animation_deck: Path, slide_video: Path
) -> None:
    """With identical narration on every slide, only the picture can place the segments."""
    deck = SlideDeck.from_pdf(animation_deck)
    segments = [
        {"id": i, "start": 2.0 * i + 0.3, "end": 2.0 * i + 1.7, "text": "And now we apply the kernel to the next position.", "confidence": 0.9}
        for i in range(16)
    ]
    truth = [timeline_page(seg["start"] + 0.7) for seg in segments]

    text_only = pipeline.aligner.align(segments, deck)
    assert text_only.mode == "text" and text_only.assignments == sorted(text_only.assignments)
    wrong = sum(1 for got, want in zip(text_only.assignments, truth) if want is not None and got != want)
    assert wrong >= 6  # text cannot separate pages that say the same thing

    evidence = match_video_to_slides(slide_video, animation_deck, config["visual"])
    fused = pipeline.aligner.align(segments, deck, evidence)
    assert fused.mode == "text+visual" and fused.visual_coverage > 0.8
    for got, want in zip(fused.assignments, truth):
        if want is not None:
            assert got == want
    camera = [got for got, want in zip(fused.assignments, truth) if want is None]
    assert camera == [1] * len(camera)  # nothing on screen: speech stays with the slide shown last
    assert len(fused.page_visits[1]) == 2 and fused.page_ranges[1][0] < 6.0  # page 1 was revisited
    assert fused.page_ranges[5] is not None and fused.page_ranges[5][0] > 27.0


def test_end_to_end_on_video_input(
    pipeline: LecturePipeline, config: Dict[str, Any], animation_deck: Path, workdir: Path
) -> None:
    """A video file goes through audio decoding, visual matching and fused alignment."""
    video = make_slide_video(workdir / "screen_recording_with_audio.mp4", animation_deck, with_audio=True)
    audio = load_audio(video)
    assert abs(len(audio) / SAMPLE_RATE - 32.0) < 0.5

    pipeline.vad = VoiceActivityDetector(dict(config["vad"], backend="energy"))
    result = pipeline.run(video, animation_deck, work_dir=workdir / "e2e_video")
    assert result.meta["visual"]["used"] is True and result.meta["alignment_mode"] == "text+visual"
    assert result.meta["visual"]["slide_visible_fraction"] > 0.7
    assert "visual_sec" in result.timings and len(result.page_visits) == 6
    assert_transcript_schema(result.segments, result.duration)
    for seg in result.segments:
        want = timeline_page(0.5 * (seg["start"] + seg["end"]))
        if want is not None and seg["end"] - seg["start"] < 3.0:
            assert abs(seg["page"] - want) <= 1 or seg["page"] == timeline_page(seg["start"])
    notes = export_markdown(result)
    assert "alignment: text+visual" in notes


def test_lora_adapter_loading(config: Dict[str, Any], device_manager: DeviceManager, workdir: Path) -> None:
    """A PEFT LoRA checkpoint can be attached both merged and as a live adapter."""
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    base = model_or_skip(lambda: WhisperForConditionalGeneration.from_pretrained(TEST_MODEL), TEST_MODEL)
    adapter_dir = workdir / "lora_adapter"
    get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])).save_pretrained(str(adapter_dir))
    assert (adapter_dir / "adapter_config.json").is_file()

    clip = SpeechSegment(0.0, 2.0, torch.from_numpy(speech_like(2.0)))
    for merge in (True, False):
        asr = WhisperASR(dict(config["asr"], merge_lora=merge), device_manager, lora_checkpoint=adapter_dir).load()
        has_lora_layers = any("lora_" in name for name, _ in asr.model.named_parameters())
        assert has_lora_layers == (not merge)
        assert next(asr.model.parameters()).device.type == device_manager.device.type
        assert_transcript_schema(asr.transcribe([clip]), 2.0)

    with pytest.raises(FileNotFoundError):
        WhisperASR(config["asr"], device_manager, lora_checkpoint=workdir / "missing_adapter").load()


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("say") is None, reason="needs the macOS `say` command")
def test_end_to_end_on_synthesized_speech(pipeline: LecturePipeline, slides_pdf: Path, workdir: Path, config: Dict[str, Any]) -> None:
    """Real speech through the default Silero VAD: slides are found and terms carry timestamps."""
    wav = synthesize_speech(workdir / "spoken_lecture.wav")
    pipeline.vad = VoiceActivityDetector(config["vad"])
    result = pipeline.run(wav, slides_pdf, work_dir=workdir / "e2e_speech")

    assert_transcript_schema(result.segments, result.duration)
    assert len(result.segments) >= 4
    pages = [seg["page"] for seg in result.segments]
    assert pages == sorted(pages) and set(pages) == {0, 1}

    first_part = sf.info(str(workdir / "say_0.wav")).duration
    turn = result.page_start(1)
    # The turn must fall at the topic change. A content-free bridge sentence ("for the second topic
    # today...") may legitimately stay with either slide, hence the tolerance of one sentence.
    assert turn is not None and abs(turn - (first_part + 1.5)) < 6.0

    # Whisper-tiny mishears a word here and there, so require most (not all) terms per slide.
    spoken = {entry.term: entry for entry in result.vocabulary if entry.occurrences}
    slide1_terms = ["loss function", "gradient descent", "learning rate", "backpropagation", "chain rule"]
    slide2_terms = ["homography", "RANSAC", "keypoint", "singular value decomposition", "covariance matrix"]
    heard1 = [t for t in slide1_terms if t in spoken]
    heard2 = [t for t in slide2_terms if t in spoken]
    assert len(heard1) >= 3 and len(heard2) >= 3, (heard1, heard2, [seg["text"] for seg in result.segments])
    for term in heard1:
        assert spoken[term].occurrences[0].page == 0 and spoken[term].first_time < first_part + 1.5
    for term in heard2:
        assert spoken[term].occurrences[0].page == 1 and spoken[term].first_time > first_part
    assert float(np.mean([seg["confidence"] for seg in result.segments])) > 0.5

    notes = export_markdown(result)
    assert f"**{heard1[0]}**" in notes and f"| **{heard2[0]}** |" in notes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-rs"]))
