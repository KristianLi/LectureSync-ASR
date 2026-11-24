"""CS/AI terminology and academic vocabulary extraction.

Two consumers use this module:

* **ASR prompting** - :meth:`TermExtractor.extract_glossary` reads the slide text and
  returns the terms worth injecting into Whisper's prompt.
* **Study aids** - :meth:`TermExtractor.extract` finds terminology in the aligned
  transcript and attaches, for every term, the slide it belongs to, the timestamp at
  which the lecturer said it and the sentence it was said in.

Candidates come from three sources that are scored together:

1. a built-in lexicon of core CS/AI terms and of high-register academic words,
2. noun-phrase candidates mined from the slides with a RAKE-style heuristic (maximal
   runs of content words between stop words and punctuation),
3. TF-IDF over the slide pages, which favours phrases that characterise a single slide
   over boilerplate repeated on every page.
"""

from __future__ import annotations

import html
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Pattern, Sequence, Set, Tuple

from .utils import get_logger, normalize_text

logger = get_logger(__name__)

# --------------------------------------------------------------------------- lexicon

_CS_AI_TERMS_RAW = """
machine learning; deep learning; supervised learning; unsupervised learning; self-supervised learning;
reinforcement learning; transfer learning; representation learning; contrastive learning; few-shot learning;
neural network; convolutional neural network; recurrent neural network; multilayer perceptron; perceptron;
feedforward network; residual connection; skip connection; batch normalization; layer normalization;
dropout; activation function; sigmoid; softmax; ReLU; tanh; logits; embedding; latent space;
backpropagation; gradient descent; stochastic gradient descent; mini-batch; learning rate; momentum;
weight decay; regularization; overfitting; underfitting; generalization; bias-variance tradeoff;
loss function; cross-entropy; mean squared error; objective function; cost function; hyperparameter;
optimizer; Adam; convergence; vanishing gradient; exploding gradient; gradient clipping; early stopping;
fine-tuning; pretraining; low-rank adaptation; LoRA; quantization; knowledge distillation; pruning;
transformer; attention mechanism; self-attention; multi-head attention; cross-attention; positional encoding;
encoder; decoder; sequence-to-sequence; language model; large language model; tokenizer; tokenization;
beam search; greedy decoding; autoregressive; perplexity; word embedding; BERT; GPT; LSTM; GRU; RNN; CNN;
convolution; kernel; stride; padding; pooling; max pooling; feature map; receptive field;
object detection; semantic segmentation; image classification; optical flow; feature extraction;
homography; epipolar geometry; fundamental matrix; essential matrix; camera calibration; intrinsic matrix;
extrinsic parameters; stereo vision; triangulation; structure from motion; bundle adjustment; RANSAC; SIFT;
keypoint; descriptor; edge detection; Gaussian blur; image gradient; affine transformation;
projective transformation; rigid transformation; point cloud; depth map;
generative adversarial network; variational autoencoder; autoencoder; diffusion model; denoising;
Markov decision process; policy gradient; Q-learning; value function; reward function; exploration;
temporal difference; Bellman equation; Monte Carlo; Markov chain; hidden Markov model;
eigenvalue; eigenvector; eigendecomposition; singular value decomposition; SVD; principal component analysis;
PCA; orthogonal; orthonormal; determinant; matrix factorization; positive definite; covariance matrix;
linear transformation; dot product; inner product; cross product; Jacobian; Hessian; gradient; partial derivative;
chain rule; Taylor expansion; convex optimization; Lagrange multiplier; constrained optimization; least squares;
linear regression; logistic regression; support vector machine; decision tree; random forest; gradient boosting;
k-means; k-nearest neighbors; clustering; dimensionality reduction; kernel trick; naive Bayes;
Bayesian inference; maximum likelihood; maximum a posteriori; prior distribution; posterior distribution;
likelihood; expectation maximization; Gaussian distribution; normal distribution; probability density;
random variable; conditional probability; Bayes theorem; entropy; KL divergence; mutual information;
variance; standard deviation; expectation; confidence interval; hypothesis testing; statistical significance;
precision and recall; F1 score; accuracy; confusion matrix; ROC curve; cross-validation; ground truth;
training set; validation set; test set; data augmentation; class imbalance; benchmark; baseline; ablation;
word error rate; speech recognition; voice activity detection; spectrogram; mel spectrogram; Fourier transform;
sampling rate; dynamic programming; memoization; recursion; divide and conquer; greedy algorithm;
time complexity; space complexity; big O notation; asymptotic; amortized; NP-hard; NP-complete; polynomial time;
binary search; hash table; hash function; linked list; binary tree; binary search tree; heap; priority queue;
stack; queue; graph traversal; breadth-first search; depth-first search; shortest path; Dijkstra;
minimum spanning tree; topological sort; adjacency matrix; adjacency list; directed acyclic graph;
sorting algorithm; quicksort; merge sort; invariant; loop invariant; pseudocode; data structure;
operating system; multithreading; thread; concurrency; parallelism; mutex; semaphore; deadlock; race condition;
virtual memory; page table; cache; scheduler; context switch; system call; file system; interrupt;
compiler; interpreter; parser; abstract syntax tree; garbage collection; type inference; polymorphism;
inheritance; encapsulation; interface; abstraction; database; relational database; SQL; database index; transaction;
normalization; distributed system; consensus; replication; fault tolerance; consistency; latency; throughput;
scalability; load balancing; TCP; UDP; HTTP; packet; routing; protocol; encryption; public key; hash;
GPU; CPU; tensor; vectorization; inference; floating point; half precision; mixed precision; batch size;
epoch; iteration; checkpoint; pipeline; API; runtime; heuristic; cosine similarity; nearest neighbor;
semantic search; retrieval; alignment; Viterbi; monotonic alignment
"""

_ACADEMIC_WORDS_RAW = """
abstraction; ambiguous; analogous; approximate; approximation; arbitrary; assumption; asymptotically;
axiom; bottleneck; canonical; coarse; coefficient; compelling; comprehensive; constraint; counterintuitive;
criterion; decompose; deterministic; discrete; discrepancy; empirical; empirically; equivalent; exhaustive;
explicit; exponential; feasible; formalize; framework; granularity; hierarchical; hypothesis; implicit;
incremental; infer; inherent; intractable; intuition; intuitive; invariant; iterative; leverage; magnitude;
marginal; methodology; mitigate; negligible; non-trivial; notation; optimal; orthogonal; paradigm; parameter;
plausible; preliminary; prerequisite; probabilistic; proportional; qualitative; quantitative; redundancy;
redundant; rigorous; robust; robustness; saturate; scalable; sparse; sparsity; stochastic; subsequent;
suboptimal; sufficient; synthesize; tractable; trade-off; trivial; underlying; validate; variability; versatile;
without loss of generality; in practice; in principle; rule of thumb; state of the art; proof of concept;
edge case; corner case; ground truth; order of magnitude
"""

# Spoken / spelled variants that should count as the canonical lexicon entry.
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "backpropagation": ("back propagation", "back-propagation", "backprop"),
    "eigenvalue": ("eigen value", "eigen-value"),
    "eigenvector": ("eigen vector", "eigen-vector"),
    "eigendecomposition": ("eigen decomposition", "eigenvalue decomposition"),
    "fine-tuning": ("finetuning", "fine tuning", "fine-tune", "fine tune", "finetune"),
    "pretraining": ("pre-training", "pre training", "pretrained", "pre-trained"),
    "overfitting": ("overfit", "over-fitting", "over fitting"),
    "underfitting": ("underfit", "under-fitting"),
    "hyperparameter": ("hyper-parameter", "hyper parameter"),
    "tokenization": ("tokenisation",),
    "regularization": ("regularisation",),
    "quantization": ("quantisation",),
    "normalization": ("normalisation",),
    "k-nearest neighbors": ("k nearest neighbours", "k-nearest neighbours", "knn"),
    "bias-variance tradeoff": ("bias variance trade-off", "bias-variance trade-off"),
    "trade-off": ("tradeoff", "trade off"),
    "sequence-to-sequence": ("seq2seq", "sequence to sequence"),
    "mini-batch": ("minibatch", "mini batch"),
    "big O notation": ("big o", "big-o notation"),
    "low-rank adaptation": ("low rank adaptation",),
}

_STOPWORDS: Set[str] = set(
    """
a about above after again against all also am an and any are aren as at be because been before being below
between both but by can cannot could did didn do does doesn doing don down during each either else even ever
every few for from further get gets got had has have having he her here hers herself him himself his how however
i if in into is isn it its itself just let like ll may me might more most much must my myself no nor not now of
off on once one only onto or other our ours ourselves out over own per please same shall she should so some such
than that the their theirs them themselves then there these they this those through thus to too under until up
upon us very via was we were what when where whether which while who whom why will with within without would
yes yet you your yours yourself yourselves
actually already always another anything around away back basically become becomes bit called came case cases
come comes different done easy end enough etc example examples fact far figure find first five following four
general give given gives go goes going gonna good great hard help here high idea important instead keep kind
know known large last later least less let lets little long look looks lot low made main make makes many maybe
mean means need needs never new next nice note notice number often okay ok old ones order part point points
possible pretty probably problem problems put question questions quite rather really recall remember result
results right said say says second see seen set show shown shows simple simply since slide slides small
something sometimes sort start step steps still stuff sure take takes talk tell term terms thing things think
third three time times today together took try two use used uses using usually want way ways week well went
whole work works yeah year years lecture lectures course class chapter section page outline overview summary
agenda introduction conclusion reference references figure table university professor dr department
""".split()
)

_GENERIC_SINGLE_WORDS: Set[str] = set(
    """
approach approaches change changes compute computed computing consider define defined definition describe
described design different either example function functions general input inputs output outputs method
methods model models number numbers process properties property provide requires simple single size value
values various version works based called common given known large small using where which while within
""".split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PHRASE_BREAK_RE = re.compile(r"[^\w\s\-']|[\n\r\t]|\s[-–—•]\s|_")


def _parse_terms(raw: str) -> List[str]:
    return [re.sub(r"\s+", " ", term).strip() for term in raw.replace("\n", " ").split(";") if term.strip()]


CS_AI_TERMS: Tuple[str, ...] = tuple(dict.fromkeys(_parse_terms(_CS_AI_TERMS_RAW)))
ACADEMIC_WORDS: Tuple[str, ...] = tuple(dict.fromkeys(_parse_terms(_ACADEMIC_WORDS_RAW)))

CATEGORY_CS_AI = "cs_ai"
CATEGORY_ACADEMIC = "academic"
CATEGORY_SLIDE = "slide_keyphrase"
CATEGORY_CUSTOM = "custom"

CATEGORY_LABELS: Dict[str, str] = {
    CATEGORY_CS_AI: "CS/AI term",
    CATEGORY_ACADEMIC: "Academic word",
    CATEGORY_SLIDE: "Slide key phrase",
    CATEGORY_CUSTOM: "Custom glossary",
}

_CATEGORY_WEIGHT: Dict[str, float] = {
    CATEGORY_CUSTOM: 3.0,
    CATEGORY_CS_AI: 2.5,
    CATEGORY_SLIDE: 1.5,
    CATEGORY_ACADEMIC: 1.0,
}

# --------------------------------------------------------------------------- data model


@dataclass
class TermOccurrence:
    """One place where a term was spoken.

    Attributes:
        segment_id: ``id`` of the transcript segment containing the term.
        time: Estimated second at which the term is spoken (interpolated inside the segment).
        segment_start: Start of the containing segment, a safe point to begin replay.
        page: Zero-based slide index the segment is aligned to (``-1`` if unknown).
        context: The sentence in which the lecturer used the term.
    """

    segment_id: int
    time: float
    segment_start: float
    page: int
    context: str


@dataclass
class VocabEntry:
    """A study-worthy term with everything needed to review it in context.

    Attributes:
        term: Canonical spelling.
        category: One of ``cs_ai``, ``academic``, ``slide_keyphrase``, ``custom``.
        score: Ranking score (higher is more important).
        count: Number of times the term was spoken.
        slide_pages: Zero-based pages whose text contains the term.
        occurrences: Spoken occurrences in chronological order.
    """

    term: str
    category: str
    score: float
    count: int = 0
    slide_pages: List[int] = field(default_factory=list)
    occurrences: List[TermOccurrence] = field(default_factory=list)

    @property
    def first_time(self) -> Optional[float]:
        """Timestamp of the first spoken occurrence, or ``None`` if only on slides."""
        return self.occurrences[0].time if self.occurrences else None

    @property
    def pages(self) -> List[int]:
        """Pages this entry should be listed under (spoken pages first, then slide pages)."""
        ordered = [occ.page for occ in self.occurrences if occ.page >= 0] + list(self.slide_pages)
        return list(dict.fromkeys(ordered))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["first_time"] = self.first_time
        return data


# --------------------------------------------------------------------------- matching helpers


def term_pattern(term: str, aliases: Sequence[str] = ()) -> Pattern[str]:
    """Compile a regex matching ``term`` (and aliases) as whole words.

    Spaces and hyphens are interchangeable, a plural ``s``/``es`` is tolerated, and
    matching is case-insensitive except for acronyms of three letters or fewer.
    """
    alternatives = []
    for variant in (term, *aliases):
        parts = [re.escape(part) for part in re.split(r"[\s\-]+", variant.strip()) if part]
        if parts:
            alternatives.append(r"[\s\-]*".join(parts) if len(parts) > 1 else parts[0])
    body = "|".join(sorted(alternatives, key=len, reverse=True))
    is_short_acronym = term.isupper() and len(term) <= 3
    flags = 0 if is_short_acronym else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?:e?s)?(?![A-Za-z0-9])", flags)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def candidate_phrases(text: str, max_words: int = 4) -> List[str]:
    """RAKE-style noun-phrase candidates: maximal runs of content words.

    The text is cut at punctuation, line breaks and stop words; every remaining run of
    one to ``max_words`` tokens is a candidate, returned with its original casing.
    """
    phrases: List[str] = []
    for chunk in _PHRASE_BREAK_RE.split(text):
        run: List[str] = []
        for token in _TOKEN_RE.findall(chunk) + [""]:
            lowered = token.lower()
            if token and lowered not in _STOPWORDS and not token.isdigit() and len(token) > 1:
                run.append(token)
                continue
            if 0 < len(run) <= max_words:
                phrases.append(" ".join(run))
            run = []
    return phrases


def _looks_technical(word: str) -> bool:
    """True for single words that are almost certainly names of things: acronyms
    (``RANSAC``), CamelCase (``ResNet``), words with digits (``word2vec``) or hyphens."""
    return (
        (word.isupper() and len(word) >= 2)
        or any(ch.isdigit() for ch in word)
        or any(ch.isupper() for ch in word[1:])
        or "-" in word
    )


def _singular_key(phrase: str) -> str:
    """Lower-cased phrase with a trailing plural ``s`` removed, for duplicate detection."""
    key = re.sub(r"[\s\-]+", " ", phrase.lower()).strip()
    return key[:-1] if key.endswith("s") and not key.endswith("ss") and len(key) > 3 else key


def _overlaps(key: str, others: Iterable[str]) -> bool:
    """True when ``key`` contains, or is contained in, one of ``others`` (singular keys)."""
    return any(key in other or other in key for other in others)


def load_glossary_file(path: Optional[str]) -> List[str]:
    """Read a user glossary: one term per line, or comma/semicolon separated. ``#`` starts a comment."""
    if not path:
        return []
    terms: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.split("#", 1)[0]
            for piece in re.split(r"[,;\t]", line):
                piece = piece.strip().strip("\"'")
                if piece:
                    terms.append(piece)
    return list(dict.fromkeys(terms))


def highlight_terms(text: str, terms: Iterable[str], style: str = "html") -> str:
    """Mark every occurrence of ``terms`` in ``text``.

    Args:
        text: Plain text (HTML-escaped internally for ``style="html"``).
        terms: Terms to mark; longer terms win when they overlap.
        style: ``"html"`` wraps matches in ``<mark class="ls-term">``; ``"markdown"`` in ``**``.
    """
    escaped = html.escape(text, quote=False) if style == "html" else text
    ordered = sorted({t for t in terms if t}, key=len, reverse=True)
    if not ordered:
        return escaped
    spans: List[Tuple[int, int]] = []
    for term in ordered:
        for match in term_pattern(term, _ALIASES.get(term, ())).finditer(escaped):
            if all(match.end() <= s or match.start() >= e for s, e in spans):
                spans.append((match.start(), match.end()))
    if not spans:
        return escaped
    open_tag, close_tag = ('<mark class="ls-term">', "</mark>") if style == "html" else ("**", "**")
    out: List[str] = []
    cursor = 0
    for start, end in sorted(spans):
        out.append(escaped[cursor:start])
        out.append(f"{open_tag}{escaped[start:end]}{close_tag}")
        cursor = end
    out.append(escaped[cursor:])
    return "".join(out)


def term_recall(references: Sequence[str], hypotheses: Sequence[str], terms: Sequence[str]) -> Dict[str, Any]:
    """Domain technical-term recall of ASR hypotheses against reference transcripts.

    For every term, each occurrence in a reference counts as a target; it is recalled if
    the paired hypothesis contains the term at least as many times (clipped counts).

    Returns:
        ``{"recall", "matched", "total", "per_term": {term: (matched, total)}}``.
        ``recall`` is ``nan`` when no reference contains any term.
    """
    patterns = {term: term_pattern(normalize_text(term), tuple(normalize_text(a) for a in _ALIASES.get(term, ())))
                for term in terms}  # fmt: skip
    matched_total, target_total = 0, 0
    per_term: Dict[str, Tuple[int, int]] = {}
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    for term, pattern in patterns.items():
        matched, total = 0, 0
        for ref, hyp in zip(refs, hyps):
            n_ref = len(pattern.findall(ref))
            if n_ref:
                total += n_ref
                matched += min(n_ref, len(pattern.findall(hyp)))
        if total:
            per_term[term] = (matched, total)
            matched_total += matched
            target_total += total
    recall = matched_total / target_total if target_total else float("nan")
    return {"recall": recall, "matched": matched_total, "total": target_total, "per_term": per_term}


# --------------------------------------------------------------------------- extractor


class TermExtractor:
    """Extract and rank CS/AI terminology from slides and transcripts.

    Args:
        config: The ``vocab`` section of the YAML configuration.
        custom_terms: User glossary; these always rank first.
    """

    def __init__(
        self, config: Optional[Mapping[str, Any]] = None, custom_terms: Optional[Sequence[str]] = None
    ) -> None:
        cfg = dict(config or {})
        self.top_k: int = int(cfg.get("top_k", 60))
        self.max_glossary_terms: int = int(cfg.get("max_glossary_terms", 40))
        self.include_slide_only: bool = bool(cfg.get("include_slide_only", True))
        self.min_phrase_chars: int = int(cfg.get("min_phrase_chars", 5))
        self.custom_terms: List[str] = list(dict.fromkeys(t.strip() for t in (custom_terms or []) if t.strip()))

        self._lexicon: Dict[str, str] = {}
        for term in ACADEMIC_WORDS:
            self._lexicon[term] = CATEGORY_ACADEMIC
        for term in CS_AI_TERMS:
            self._lexicon[term] = CATEGORY_CS_AI
        for term in self.custom_terms:
            self._lexicon[term] = CATEGORY_CUSTOM
        self._canonical_by_lower: Dict[str, str] = {term.lower(): term for term in self._lexicon}
        self._patterns: Dict[str, Pattern[str]] = {}

    # ------------------------------------------------------------------ internals

    def _pattern(self, term: str) -> Pattern[str]:
        if term not in self._patterns:
            self._patterns[term] = term_pattern(term, _ALIASES.get(term, ()))
        return self._patterns[term]

    def _lexicon_hits(self, text: str) -> Dict[str, int]:
        """Lexicon terms present in ``text`` with their match counts."""
        hits: Dict[str, int] = {}
        if not text.strip():
            return hits
        for term in self._lexicon:
            count = len(self._pattern(term).findall(text))
            if count:
                hits[term] = count
        return hits

    def _slide_keyphrases(self, page_texts: Sequence[str]) -> Dict[str, float]:
        """TF-IDF-scored noun-phrase candidates from the slides (documents = pages).

        Multi-word runs are kept as they are. A single word is only kept when it looks
        like the name of something (acronym, CamelCase, digits, hyphen); ordinary single
        words are too noisy without a part-of-speech tagger and the important ones are
        already covered by the lexicon.

        Returns:
            Mapping from the phrase (most frequent surface form) to its score.
        """
        per_page: List[Counter] = []
        surface: Dict[str, Counter] = defaultdict(Counter)
        for text in page_texts:
            counts: Counter = Counter()
            for phrase in candidate_phrases(text):
                key = phrase.lower()
                counts[key] += 1
                surface[key][phrase] += 1
            per_page.append(counts)
        n_docs = max(1, len(per_page))
        doc_freq: Counter = Counter()
        for counts in per_page:
            doc_freq.update(counts.keys())

        scores: Dict[str, float] = defaultdict(float)
        for counts in per_page:
            for key, tf in counts.items():
                words = key.split()
                best_surface = surface[key].most_common(1)[0][0]
                if len(key) < self.min_phrase_chars and not _looks_technical(best_surface):
                    continue
                if len(words) == 1 and not _looks_technical(best_surface):
                    continue
                if len(words) == 1 and key in _GENERIC_SINGLE_WORDS:
                    continue
                idf = math.log((1 + n_docs) / (1 + doc_freq[key])) + 1.0
                length_boost = 1.0 + 0.35 * (len(words) - 1)
                name = best_surface if len(words) == 1 else key
                scores[name] += (1.0 + math.log(tf)) * idf * length_boost
        return dict(scores)

    def _canonical(self, phrase: str) -> str:
        """Map a lower-cased phrase to the lexicon's spelling when it is a known term."""
        return self._canonical_by_lower.get(phrase.lower(), phrase)

    # ------------------------------------------------------------------ glossary for ASR

    def extract_glossary(self, page_texts: Sequence[str], max_terms: Optional[int] = None) -> List[str]:
        """Pick the slide terms that should be injected into the Whisper prompt.

        Order: user glossary, then lexicon terms found on the slides (most frequent
        first), then the highest-scoring slide key phrases.
        """
        limit = int(max_terms or self.max_glossary_terms)
        joined = "\n".join(page_texts)
        lexicon_hits = self._lexicon_hits(joined)
        ranked_lexicon = sorted(
            (t for t in lexicon_hits if self._lexicon[t] == CATEGORY_CS_AI),
            key=lambda t: (-lexicon_hits[t], -len(t)),
        )
        keyphrases = self._slide_keyphrases(page_texts)
        ranked_phrases = sorted(keyphrases, key=lambda p: -keyphrases[p])

        glossary: List[str] = []
        seen: Set[str] = set()
        trusted = len(self.custom_terms) + len(ranked_lexicon)
        ordered = [*self.custom_terms, *ranked_lexicon, *(self._canonical(p) for p in ranked_phrases)]
        for position, term in enumerate(ordered):
            key = _singular_key(term)
            if key in seen:
                continue
            # A mined phrase adds nothing when a chosen term already covers its head words
            # ("softmax layer" next to "softmax", "feature maps" next to "feature map").
            if position >= trusted and _overlaps(key, seen):
                continue
            # A lexicon term already spelled out inside a longer chosen one wastes prompt tokens.
            if len(self.custom_terms) <= position < trusted and any(key in chosen for chosen in seen):
                continue
            seen.add(key)
            glossary.append(term)
            if len(glossary) >= limit:
                break
        return glossary

    # ------------------------------------------------------------------ study vocabulary

    def extract(
        self,
        segments: Sequence[Mapping[str, Any]],
        page_texts: Sequence[str],
        top_k: Optional[int] = None,
    ) -> List[VocabEntry]:
        """Build the ranked vocabulary list for a lecture.

        Args:
            segments: Aligned transcript segments (``id``, ``start``, ``end``, ``text``
                and optionally ``page``).
            page_texts: Slide texts in page order.
            top_k: Maximum number of entries; defaults to the configured value.

        Returns:
            Entries sorted by descending score. Each spoken entry carries timestamps and
            the sentence of every occurrence.
        """
        limit = int(top_k or self.top_k)
        transcript = " ".join(str(seg.get("text", "")) for seg in segments)
        slide_text = "\n".join(page_texts)

        keyphrases = self._slide_keyphrases(page_texts)
        max_kp = max(keyphrases.values(), default=1.0)

        candidates: Dict[str, Tuple[str, float]] = {}  # term -> (category, tf-idf component)
        spoken_hits = self._lexicon_hits(transcript)
        slide_hits = self._lexicon_hits(slide_text)
        for term in set(spoken_hits) | set(slide_hits):
            candidates[term] = (self._lexicon[term], keyphrases.get(term.lower(), 0.0) / max_kp)
        lexicon_keys = {_singular_key(term) for term in candidates}
        for phrase, score in keyphrases.items():
            term = self._canonical(phrase)
            # Mined phrases that merely extend a lexicon term ("softmax layer") are redundant.
            if term in candidates or _overlaps(_singular_key(phrase), lexicon_keys):
                continue
            candidates[term] = (CATEGORY_SLIDE, score / max_kp)

        terms = list(candidates)
        spoken = self._free_matches([str(seg.get("text", "")) for seg in segments], terms)
        printed = self._free_matches(page_texts, terms)

        entries: List[VocabEntry] = []
        for term, (category, tfidf) in candidates.items():
            occurrences = self._occurrences(segments, spoken.get(term, []))
            slide_pages = sorted({index for index, _, _ in printed.get(term, [])})
            if not occurrences and not slide_pages:
                continue
            if not occurrences and (not self.include_slide_only or category == CATEGORY_ACADEMIC):
                continue
            if category == CATEGORY_SLIDE and not occurrences and tfidf < 0.5:
                continue
            count = len(spoken.get(term, []))
            score = _CATEGORY_WEIGHT[category] + tfidf
            score += 0.5 if (occurrences and slide_pages) else 0.0
            score += 0.3 if len(term.split()) > 1 else 0.0
            score += 0.15 * math.log1p(count)
            score -= 0.75 if not occurrences else 0.0
            entries.append(
                VocabEntry(
                    term=term,
                    category=category,
                    score=round(score, 4),
                    count=count,
                    slide_pages=slide_pages,
                    occurrences=occurrences,
                )
            )

        entries.sort(key=lambda e: (-e.score, e.first_time if e.first_time is not None else float("inf"), e.term))
        entries = entries[:limit]
        logger.info(
            "Vocabulary: %d entries (%d spoken, %d slide-only)",
            len(entries), sum(1 for e in entries if e.occurrences), sum(1 for e in entries if not e.occurrences),
        )  # fmt: skip
        return entries

    def _free_matches(self, texts: Sequence[str], terms: Sequence[str]) -> Dict[str, List[Tuple[int, int, int]]]:
        """Find every term in every text, ignoring matches nested inside a longer term.

        ``pooling`` inside ``max pooling`` or ``neural network`` inside ``convolutional
        neural network`` is not an independent use of the shorter term, so it is not
        counted for it. A shorter term is still reported wherever it stands on its own.

        Returns:
            ``{term: [(text_index, start, end), ...]}`` in reading order.
        """
        free: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        for index, text in enumerate(texts):
            if not text:
                continue
            found = [
                (match.start(), match.end(), term)
                for term in terms
                for match in self._pattern(term).finditer(text)
            ]
            for start, end, term in found:
                nested = any(s <= start and end <= e and (e - s) > (end - start) for s, e, _ in found)
                if not nested:
                    free[term].append((index, start, end))
        return free

    @staticmethod
    def _occurrences(
        segments: Sequence[Mapping[str, Any]], matches: Sequence[Tuple[int, int, int]]
    ) -> List[TermOccurrence]:
        """Turn matches into occurrences (one per segment) with a time estimate and context.

        The time inside a segment is interpolated from the character offset of the match.
        """
        occurrences: List[TermOccurrence] = []
        seen_segments: Set[int] = set()
        for index, match_start, match_end in matches:
            if index in seen_segments:
                continue
            seen_segments.add(index)
            seg = segments[index]
            text = str(seg.get("text", ""))
            start, end = float(seg["start"]), float(seg["end"])
            fraction = match_start / max(1, len(text))
            context, cursor = text, 0
            for sentence in split_sentences(text):
                position = text.find(sentence, cursor)
                cursor = position + len(sentence)
                if position <= match_start and match_end <= cursor and len(sentence) >= 40:
                    context = sentence
                    break
            occurrences.append(
                TermOccurrence(
                    segment_id=int(seg.get("id", index)),
                    time=round(start + fraction * (end - start), 2),
                    segment_start=round(start, 2),
                    page=int(seg.get("page", -1)),
                    context=context,
                )
            )
        return occurrences

    # ------------------------------------------------------------------ convenience

    @staticmethod
    def by_page(entries: Sequence[VocabEntry], n_pages: int) -> List[List[VocabEntry]]:
        """Group entries per slide: a term is listed on every page where it is first relevant."""
        grouped: List[List[VocabEntry]] = [[] for _ in range(n_pages)]
        for entry in entries:
            for page in entry.pages:
                if 0 <= page < n_pages:
                    grouped[page].append(entry)
        return grouped

    def benchmark_terms(self) -> List[str]:
        """Terms used for the domain-term recall metric (CS/AI lexicon plus user glossary)."""
        return [t for t, c in self._lexicon.items() if c in {CATEGORY_CS_AI, CATEGORY_CUSTOM}]
