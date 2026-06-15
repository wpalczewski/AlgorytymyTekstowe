"""
Segmentacja polityki prywatności na sekcje.

Kaskada trzech strategii:
  A — heurystyka strukturalna (nagłówki, "Return to top", itp.)
  B — klasyfikacja semantyczna OPP-115 (bloki zdań tej samej kategorii)
  C — TextTiling fallback (sliding-window podobieństwa)

Wynik: lista Section dataclass z polami title, sentences, dominant_category, strategy.
Embeddingi sekcji oblicza matching.py (żeby uniknąć podwójnego ładowania modelu).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import yaml


def _load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class Section:
    title: str
    sentences: list[str]
    dominant_category: Optional[str] = None
    strategy: str = "A"          # A | B | C
    embedding: Optional[object] = None   # wypełnia matching.py


# ---------------------------------------------------------------------------
# Strategia A — heurystyka strukturalna
# ---------------------------------------------------------------------------

_SEPARATORS = re.compile(
    r"^(Return to top|Back to top)\s*$", re.IGNORECASE | re.MULTILINE
)

def _is_heading(line: str, max_len: int) -> bool:
    s = line.strip()
    if not s or len(s) > max_len:
        return False
    # Numeracja: "1.", "1.1", "I.", itp. — nagłówek niezależnie od końcówki
    if re.match(r"^(\d+\.|\d+\.\d+|[IVXivx]+\.)\s", s):
        return True
    # Kończy się ? lub : → nagłówek
    if s.endswith("?") or s.endswith(":"):
        return True
    # Krótka linia bez końcówki zdania + title case lub caps
    if s[-1] not in ".,:;":
        if s.istitle() or s.isupper():
            return True
    return False


def _split_on_separators(text: str, cfg: dict) -> list[str]:
    """Dzieli tekst na bloki wg separatorów 'Return to top' i podobnych."""
    parts = _SEPARATORS.split(text)
    # split zostawia separatory jako grupy — filtrujemy puste i same separatory
    return [p.strip() for p in parts if p and not _SEPARATORS.fullmatch(p.strip())]


def _extract_title_and_body(block: str, max_heading_len: int) -> tuple[str, str]:
    """Z bloku wyciąga pierwszą linię-nagłówek (jeśli jest) i resztę jako treść."""
    lines = block.splitlines()
    # Szukamy pierwszej niepustej linii
    for i, line in enumerate(lines):
        if line.strip():
            if _is_heading(line, max_heading_len):
                title = line.strip()
                body = "\n".join(lines[i + 1:]).strip()
                return title, body
            break
    return "", block.strip()


def _sentences_from_text(text: str, nlp) -> list[str]:
    doc = nlp(text)
    sents = [s.text.strip().replace("\n", " ") for s in doc.sents]
    return [s for s in sents if len(s.split()) > 4]


def segment_strategy_a(text: str, nlp, cfg: dict) -> list[Section]:
    """
    Strategia A: dzieli na bloki wg separatorów 'Return to top',
    potem dla każdego bloku wyodrębnia nagłówek i zdania treści.
    """
    max_len = cfg["segmentation"]["max_heading_len"]
    blocks = _split_on_separators(text, cfg)

    # Jeśli tekst nie ma separatorów — szukamy nagłówków linia po linii
    if len(blocks) <= 1:
        blocks = _split_by_headings(text, max_len)

    sections = []
    for block in blocks:
        title, body = _extract_title_and_body(block, max_len)
        sents = _sentences_from_text(body or block, nlp)
        if sents:
            sections.append(Section(title=title or "Untitled", sentences=sents, strategy="A"))
    return sections


def _split_by_headings(text: str, max_len: int) -> list[str]:
    """Fallback: dzieli tekst na bloki wg wykrytych nagłówków."""
    lines = text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if _is_heading(line, max_len) and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


# ---------------------------------------------------------------------------
# Strategia B — klasyfikacja semantyczna OPP-115
# ---------------------------------------------------------------------------

def segment_strategy_b(sentences: list[str], classifier, cfg: dict) -> list[Section]:
    """
    Strategia B: każde zdanie klasyfikowane przez OPP-115.
    Ciągłe bloki zdań tej samej dominującej kategorii → sekcja.
    """
    window = cfg["segmentation"]["category_window"]
    if not sentences:
        return []

    labeled = [(s, _dominant(classifier.classify_text(s))) for s in sentences]

    sections: list[Section] = []
    buf_sents = [labeled[0][0]]
    buf_cat = labeled[0][1]

    for sent, cat in labeled[1:]:
        if cat == buf_cat or cat is None:
            buf_sents.append(sent)
        else:
            # sprawdzamy momentum: jeśli kolejne `window` zdań mają nową kategorię → nowa sekcja
            sections.append(Section(
                title=buf_cat or "Uncategorized",
                sentences=buf_sents,
                dominant_category=buf_cat,
                strategy="B",
            ))
            buf_sents = [sent]
            buf_cat = cat

    sections.append(Section(
        title=buf_cat or "Uncategorized",
        sentences=buf_sents,
        dominant_category=buf_cat,
        strategy="B",
    ))
    return _merge_small_sections(sections, min_sents=3)


def _dominant(categories: list[str]) -> Optional[str]:
    return categories[0] if categories else None


def _merge_small_sections(sections: list[Section], min_sents: int) -> list[Section]:
    """Scala sekcje z mniej niż min_sents zdań z sąsiadkami."""
    merged: list[Section] = []
    for sec in sections:
        if merged and len(sec.sentences) < min_sents:
            merged[-1].sentences.extend(sec.sentences)
        else:
            merged.append(sec)
    return merged


# ---------------------------------------------------------------------------
# Strategia C — TextTiling fallback
# ---------------------------------------------------------------------------

def segment_strategy_c(sentences: list[str], encoder, cfg: dict) -> list[Section]:
    """
    Strategia C: sliding-window cosine similarity embeddingów zdań.
    Granica sekcji tam gdzie lokalny dołek podobieństwa.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    w = cfg["segmentation"]["texttiling_w"]
    k = cfg["segmentation"]["texttiling_k"]

    if len(sentences) < 2 * w:
        return [Section(title="Section 1", sentences=sentences, strategy="C")]

    embs = encoder.encode(sentences, show_progress_bar=False)

    scores = []
    for i in range(w, len(sentences) - w):
        left = embs[max(0, i - w):i]
        right = embs[i:min(len(embs), i + w)]
        left_mean = left.mean(axis=0, keepdims=True)
        right_mean = right.mean(axis=0, keepdims=True)
        scores.append(float(cosine_similarity(left_mean, right_mean)[0, 0]))

    # Wygładzanie
    if len(scores) >= k:
        kernel = np.ones(k) / k
        smoothed = np.convolve(scores, kernel, mode="same")
    else:
        smoothed = np.array(scores)

    # Lokalne minima jako granice sekcji
    boundaries = _find_valleys(smoothed, offset=w)

    sections = []
    prev = 0
    for i, b in enumerate(boundaries):
        sents = sentences[prev:b]
        if sents:
            sections.append(Section(title=f"Section {i + 1}", sentences=sents, strategy="C"))
        prev = b
    if sentences[prev:]:
        sections.append(Section(title=f"Section {len(boundaries) + 1}", sentences=sentences[prev:], strategy="C"))

    return sections


def _find_valleys(scores, offset: int, min_gap: int = 5) -> list[int]:
    """Zwraca indeksy lokalnych minimów w tablicy scores."""
    valleys = []
    for i in range(1, len(scores) - 1):
        if scores[i] < scores[i - 1] and scores[i] < scores[i + 1]:
            idx = i + offset
            if not valleys or idx - valleys[-1] >= min_gap:
                valleys.append(idx)
    return valleys


# ---------------------------------------------------------------------------
# Główna funkcja
# ---------------------------------------------------------------------------

def segment_policy(
    text: str,
    nlp,
    classifier=None,
    encoder=None,
    cfg_path: str = "config.yaml",
) -> list[Section]:
    """
    Próbuje segmentację kaskadowo A → B → C.
    Zwraca listę Section. Embeddingi są None — wypełnia je matching.py.
    """
    cfg = _load_cfg(cfg_path)

    sections_a = segment_strategy_a(text, nlp, cfg)
    if len(sections_a) >= 3:
        return sections_a

    # Fallback B: zbieramy zdania z A i reklasyfikujemy
    if classifier is not None:
        all_sents = [s for sec in sections_a for s in sec.sentences]
        if not all_sents:
            doc = nlp(text)
            all_sents = [s.text.strip() for s in doc.sents if len(s.text.split()) > 4]
        sections_b = segment_strategy_b(all_sents, classifier, cfg)
        if len(sections_b) >= 3:
            return sections_b

    # Fallback C
    if encoder is not None:
        all_sents = [s for sec in sections_a for s in sec.sentences]
        if not all_sents:
            doc = nlp(text)
            all_sents = [s.text.strip() for s in doc.sents if len(s.text.split()) > 4]
        return segment_strategy_c(all_sents, encoder, cfg)

    return sections_a
