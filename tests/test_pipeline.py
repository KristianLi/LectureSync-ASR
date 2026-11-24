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
from src.vocab import TermExtractor, highlight_terms, term_recall  # noqa: E402

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
