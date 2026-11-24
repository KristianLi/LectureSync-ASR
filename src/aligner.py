"""Cross-modal alignment between a slide deck (PDF) and a lecture transcript.

Three pieces live here:

* :class:`SlideDeck` extracts per-page text with PyMuPDF and renders every page to PNG.
* :class:`SlideAligner` embeds transcript segments and slide pages with a
  Sentence-Transformers model and builds the ``N x M`` cosine-similarity matrix ``S``.
* :func:`monotonic_dp_align` assigns one slide ``p_i`` to every segment ``i`` by maximising

  .. math:: \\sum_i S[i, p_i] - \\lambda \\cdot \\text{jump\\_penalty}(p_i - p_{i-1})

  subject to ``p_0 <= p_1 <= ... <= p_{N-1}``.

Why dynamic programming instead of a per-segment ``argmax``: lecturers constantly use
words that belong to other slides ("as we saw earlier", "we will get to transformers
later"). A greedy assignment flickers between pages; the monotonic constraint makes
going back impossible, and the jump penalty makes moving forward cost something, so the
path only advances when the following segments keep supporting the new slide. That is
exactly the moment the lecturer turns the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .utils import DeviceManager, PathLike, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- slides


@dataclass
class SlidePage:
    """Text and rendered image of one PDF page.

    Attributes:
        index: Zero-based page index.
        text: Plain text extracted from the page (may be empty for image-only slides).
        title: First non-empty line, used as a heading in the UI and exported notes.
        image_path: PNG rendering of the page, or ``None`` if rendering was skipped.
    """

    index: int
    text: str
    title: str
    image_path: Optional[Path] = None


class SlideDeck:
    """A PDF slide deck loaded with PyMuPDF.

    Example:
        >>> deck = SlideDeck.from_pdf("lecture05.pdf", image_dir="outputs/slides")
        >>> deck.pages[0].title, deck.pages[0].image_path
    """

    def __init__(self, pages: Sequence[SlidePage], source: Optional[Path] = None) -> None:
        self.pages: List[SlidePage] = list(pages)
        self.source: Optional[Path] = source

    def __len__(self) -> int:
        return len(self.pages)

    @property
    def texts(self) -> List[str]:
        """Page texts in page order."""
        return [page.text for page in self.pages]

    @classmethod
    def from_pdf(
        cls, pdf_path: PathLike, image_dir: Optional[PathLike] = None, dpi: int = 150
    ) -> "SlideDeck":
        """Extract text from every page and optionally render the pages to PNG.

        Args:
            pdf_path: Slide deck in PDF format.
            image_dir: Directory for ``page_001.png`` ...; rendering is skipped when ``None``.
            dpi: Rendering resolution (150 dpi gives roughly 2000 px wide 16:9 slides).

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the PDF has no pages.
        """
        try:
            import pymupdf  # PyMuPDF >= 1.24.3
        except ImportError:  # pragma: no cover - older PyMuPDF only ships the fitz name
            import fitz as pymupdf

        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        out_dir = Path(image_dir) if image_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        pages: List[SlidePage] = []
        with pymupdf.open(str(pdf_path)) as document:
            if document.page_count == 0:
                raise ValueError(f"PDF has no pages: {pdf_path}")
            for index, page in enumerate(document):
                text = " ".join(page.get_text("text").split("\x00")).strip()
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                title = lines[0][:120] if lines else f"Slide {index + 1}"
                image_path: Optional[Path] = None
                if out_dir is not None:
                    image_path = out_dir / f"page_{index + 1:03d}.png"
                    page.get_pixmap(dpi=dpi, alpha=False).save(str(image_path))
                pages.append(SlidePage(index=index, text="\n".join(lines), title=title, image_path=image_path))
        logger.info("Loaded %d slides from %s", len(pages), pdf_path.name)
        return cls(pages, source=pdf_path)


# --------------------------------------------------------------------------- dynamic programming


def jump_penalty(delta: np.ndarray, skip_penalty: float = 0.5) -> np.ndarray:
    """Cost (in units of ``lambda``) of moving ``delta`` pages forward.

    ``0`` for staying, ``1`` for turning to the next page, and ``1 + skip_penalty * (delta - 1)``
    when pages are skipped. Backward moves are forbidden (infinite cost).
    """
    delta = np.asarray(delta, dtype=np.float64)
    cost = np.where(delta <= 0, 0.0, 1.0 + skip_penalty * (delta - 1.0))
    return np.where(delta < 0, np.inf, cost)


def monotonic_dp_align(
    similarity: np.ndarray, jump_lambda: float = 0.12, skip_penalty: float = 0.5
) -> List[int]:
    """Viterbi-style monotonic alignment of ``N`` segments to ``M`` pages.

    Recurrence, with ``pen = jump_penalty``::

        D[0, p] = S[0, p] - lambda * pen(p)                       # lecture starts at page 0
        D[i, p] = S[i, p] + max_{q <= p} ( D[i-1, q] - lambda * pen(p - q) )

    The optimal path is recovered by backtracking from ``argmax_p D[N-1, p]``.

    Args:
        similarity: ``(N, M)`` matrix; larger means segment ``i`` matches page ``p`` better.
        jump_lambda: Weight ``lambda`` of the jump penalty.
        skip_penalty: Extra per-skipped-page cost, see :func:`jump_penalty`.

    Returns:
        Page index per segment, guaranteed non-decreasing. Complexity is ``O(N * M^2)``
        time (vectorised over ``M^2``) and ``O(N * M)`` memory.
    """
    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"similarity must be 2-D, got shape {scores.shape}")
    n_segments, n_pages = scores.shape
    if n_segments == 0:
        return []
    if n_pages == 0:
        raise ValueError("Cannot align against an empty slide deck")

    page_ids = np.arange(n_pages)
    # transition[q, p] = lambda * pen(p - q); +inf where p < q forbids going backwards.
    transition = jump_lambda * jump_penalty(page_ids[None, :] - page_ids[:, None], skip_penalty)

    best = np.empty((n_segments, n_pages), dtype=np.float64)
    backpointer = np.zeros((n_segments, n_pages), dtype=np.int64)
    best[0] = scores[0] - jump_lambda * jump_penalty(page_ids, skip_penalty)
    for i in range(1, n_segments):
        candidates = best[i - 1][:, None] - transition  # [q, p]
        backpointer[i] = np.argmax(candidates, axis=0)
        best[i] = scores[i] + candidates[backpointer[i], page_ids]

    path = np.empty(n_segments, dtype=np.int64)
    path[-1] = int(np.argmax(best[-1]))
    for i in range(n_segments - 1, 0, -1):
        path[i - 1] = backpointer[i, path[i]]
    return [int(p) for p in path]


# --------------------------------------------------------------------------- aligner


@dataclass
class AlignmentResult:
    """Outcome of aligning a transcript to a slide deck.

    Attributes:
        assignments: Page index per transcript segment (non-decreasing).
        similarity: The ``(N, M)`` similarity matrix that was optimised.
        page_ranges: For every page, ``(start_sec, end_sec)`` of its narration or ``None``.
        objective: Value of the DP objective for ``assignments``.
    """

    assignments: List[int]
    similarity: np.ndarray
    page_ranges: List[Optional[tuple]] = field(default_factory=list)
    objective: float = 0.0


class SlideAligner:
    """Embed transcript segments and slides, then align them monotonically.

    Args:
        config: The ``aligner`` section of the YAML configuration.
        device_manager: Shared :class:`DeviceManager`; the encoder and every embedding
            tensor live on its device, and similarities are moved to the CPU as NumPy.
    """

    def __init__(
        self, config: Optional[Mapping[str, Any]] = None, device_manager: Optional[DeviceManager] = None
    ) -> None:
        cfg = dict(config or {})
        self.model_name: str = str(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.batch_size: int = int(cfg.get("batch_size", 64))
        self.context_window: int = int(cfg.get("context_window", 1))
        self.context_min_words: int = int(cfg.get("context_min_words", 12))
        self.lexical_weight: float = float(cfg.get("lexical_weight", 0.25))
        self.jump_lambda: float = float(cfg.get("jump_penalty", 0.12))
        self.skip_penalty: float = float(cfg.get("skip_penalty", 0.5))
        self.render_dpi: int = int(cfg.get("render_dpi", 150))
        self.device_manager: DeviceManager = device_manager or DeviceManager()
        self._encoder: Any = None

    # ------------------------------------------------------------------ embeddings

    def load(self) -> "SlideAligner":
        """Load the sentence encoder onto the managed device. Idempotent."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading %s on %s", self.model_name, self.device_manager.device)
            self._encoder = SentenceTransformer(self.model_name, device=str(self.device_manager.device))
        return self

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return L2-normalised sentence embeddings, shape ``(len(texts), dim)``, on the model device."""
        self.load()
        if len(texts) == 0:
            dim = int(self._encoder.get_sentence_embedding_dimension() or 384)
            return torch.zeros((0, dim), device=self.device_manager.device)
        embeddings = self._encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.to(self.device_manager.device).float()

    def _contextualise(self, texts: Sequence[str]) -> List[str]:
        """Give short utterances some surrounding speech before they are embedded.

        "Okay, any questions so far?" carries no topical signal, so segments with fewer
        than ``context_min_words`` words borrow up to ``context_window`` neighbours on
        each side. Longer segments are embedded on their own: mixing a neighbour into
        them would blur exactly the slide boundaries the aligner has to find.
        """
        if self.context_window <= 0:
            return list(texts)
        out: List[str] = []
        for index, text in enumerate(texts):
            lo = hi = index
            words = len(text.split())
            for _ in range(self.context_window):
                if words >= self.context_min_words:
                    break
                lo, hi = max(0, lo - 1), min(len(texts) - 1, hi + 1)
                words = sum(len(t.split()) for t in texts[lo : hi + 1])
            out.append(" ".join(t for t in texts[lo : hi + 1] if t))
        return out

    @staticmethod
    def _lexical_similarity(segment_texts: Sequence[str], page_texts: Sequence[str]) -> Optional[np.ndarray]:
        """TF-IDF cosine similarity; exact technical terms shared by speech and slide score highly."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = list(page_texts) + list(segment_texts)
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
            )
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:  # vocabulary empty (e.g. only stop words)
            return None
        pages = matrix[: len(page_texts)]
        segments = matrix[len(page_texts) :]
        return np.asarray((segments @ pages.T).todense(), dtype=np.float32)

    def similarity_matrix(self, segment_texts: Sequence[str], page_texts: Sequence[str]) -> np.ndarray:
        """Build the ``(N, M)`` similarity matrix ``S``.

        ``S = (1 - w) * cos(E_seg, E_page) + w * tfidf_cos`` with ``w = lexical_weight``.
        Pages without extractable text (image-only slides) get each row's minimum so they
        never attract segments on their own.
        """
        n_segments, n_pages = len(segment_texts), len(page_texts)
        if n_segments == 0 or n_pages == 0:
            return np.zeros((n_segments, n_pages), dtype=np.float32)

        contextual = self._contextualise(segment_texts)
        segment_emb = self.encode(contextual)
        page_emb = self.encode([text if text.strip() else "empty slide" for text in page_texts])
        semantic = (segment_emb @ page_emb.T).cpu().numpy().astype(np.float32)

        similarity = semantic
        if self.lexical_weight > 0:
            lexical = self._lexical_similarity(contextual, page_texts)
            if lexical is not None:
                similarity = (1.0 - self.lexical_weight) * semantic + self.lexical_weight * lexical

        empty_pages = [i for i, text in enumerate(page_texts) if not text.strip()]
        if empty_pages and len(empty_pages) < n_pages:
            row_min = similarity.min(axis=1)
            for page in empty_pages:
                similarity[:, page] = row_min
        return similarity.astype(np.float32)

    # ------------------------------------------------------------------ alignment

    def align(self, segments: Sequence[Mapping[str, Any]], deck: SlideDeck) -> AlignmentResult:
        """Assign a slide to every transcript segment.

        Args:
            segments: Transcript segments with at least ``text``, ``start`` and ``end``.
            deck: The slide deck to align against.

        Returns:
            :class:`AlignmentResult`; ``assignments[i]`` is the page of ``segments[i]``.
        """
        texts = [str(seg.get("text", "")) for seg in segments]
        similarity = self.similarity_matrix(texts, deck.texts)
        assignments = monotonic_dp_align(similarity, self.jump_lambda, self.skip_penalty)

        page_ranges: List[Optional[tuple]] = [None] * len(deck)
        for seg, page in zip(segments, assignments):
            start, end = float(seg["start"]), float(seg["end"])
            current = page_ranges[page]
            page_ranges[page] = (start, end) if current is None else (min(current[0], start), max(current[1], end))

        objective = 0.0
        previous = 0
        for index, page in enumerate(assignments):
            objective += float(similarity[index, page])
            objective -= self.jump_lambda * float(jump_penalty(np.array(page - previous), self.skip_penalty))
            previous = page
        covered = sum(1 for r in page_ranges if r is not None)
        logger.info(
            "Aligned %d segments to %d/%d slides (objective %.3f)",
            len(assignments), covered, len(deck), objective,
        )  # fmt: skip
        return AlignmentResult(assignments, similarity, page_ranges, objective)


def attach_pages(
    segments: Sequence[Mapping[str, Any]], assignments: Sequence[int]
) -> List[Dict[str, Any]]:
    """Return copies of ``segments`` with a ``page`` key (zero-based slide index)."""
    if len(segments) != len(assignments):
        raise ValueError("segments and assignments must have the same length")
    return [dict(seg, page=int(page)) for seg, page in zip(segments, assignments)]
