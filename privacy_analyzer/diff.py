"""
Wykrywanie zmian między dopasowanymi sekcjami.

Dla pary (old_section, new_section) oblicza:
  - distance_score     = 1 - cosine(emb_old, emb_new)
  - length_delta       = |n_new - n_old| / max(n_old, 1)
  - sentence-level diff: każde zdanie nowe vs stare → unchanged / rephrased / added
  - Symetrycznie: zdania stare → unchanged / rephrased / removed
  - category_drift     = 1 jeśli dominant_category się zmieniła
  - risk_delta         = |risk_new - risk_old| / 100
  - change_magnitude   = composite 0-100 z wagami z config.yaml
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import yaml

from .segmentation import Section
from .matching import ThematicChain


def _load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class SentenceChange:
    text: str
    status: str           # "unchanged" | "rephrased" | "added" | "removed"
    best_match: Optional[str] = None
    best_score: float = 0.0


@dataclass
class SectionDiff:
    old_title: str
    new_title: str
    year_old: str
    year_new: str
    distance_score: float = 0.0
    length_delta: float = 0.0
    category_drift: float = 0.0
    risk_delta: float = 0.0
    change_magnitude: float = 0.0
    sentence_changes: list[SentenceChange] = field(default_factory=list)
    n_added: int = 0
    n_removed: int = 0
    n_rephrased: int = 0
    n_unchanged: int = 0


def _sent_level_diff(
    old_sents: list[str],
    new_sents: list[str],
    encoder,
    unchanged_thr: float,
    rephrased_thr: float,
) -> list[SentenceChange]:
    """Klasyfikuje każde zdanie nowej sekcji względem starej."""
    if not old_sents or not new_sents:
        return [SentenceChange(text=s, status="added") for s in new_sents]

    old_embs = encoder.encode(old_sents, show_progress_bar=False)
    new_embs = encoder.encode(new_sents, show_progress_bar=False)
    sim = cosine_similarity(new_embs, old_embs)   # (n_new, n_old)

    changes = []
    for i, sent in enumerate(new_sents):
        best_score = float(sim[i].max())
        best_idx = int(sim[i].argmax())
        if best_score >= unchanged_thr:
            status = "unchanged"
        elif best_score >= rephrased_thr:
            status = "rephrased"
        else:
            status = "added"
        changes.append(SentenceChange(
            text=sent,
            status=status,
            best_match=old_sents[best_idx],
            best_score=best_score,
        ))
    return changes


def _removed_sentences(
    old_sents: list[str],
    new_sents: list[str],
    encoder,
    unchanged_thr: float,
    rephrased_thr: float,
) -> list[SentenceChange]:
    """Znajduje zdania starej sekcji bez odpowiednika w nowej → 'removed'."""
    if not new_sents or not old_sents:
        return [SentenceChange(text=s, status="removed") for s in old_sents]

    old_embs = encoder.encode(old_sents, show_progress_bar=False)
    new_embs = encoder.encode(new_sents, show_progress_bar=False)
    sim = cosine_similarity(old_embs, new_embs)   # (n_old, n_new)

    removed = []
    for i, sent in enumerate(old_sents):
        best_score = float(sim[i].max())
        if best_score < rephrased_thr:
            removed.append(SentenceChange(
                text=sent,
                status="removed",
                best_score=best_score,
            ))
    return removed


def compute_section_diff(
    old_sec: Section,
    new_sec: Section,
    year_old: str,
    year_new: str,
    encoder,
    nlp=None,
    cfg_path: str = "config.yaml",
) -> SectionDiff:
    cfg = _load_cfg(cfg_path)
    weights = cfg["diff"]["weights"]
    unch_thr = cfg["diff"]["unchanged_threshold"]
    repr_thr = cfg["diff"]["rephrased_threshold"]

    diff = SectionDiff(
        old_title=old_sec.title,
        new_title=new_sec.title,
        year_old=year_old,
        year_new=year_new,
    )

    # distance_score
    if old_sec.embedding is not None and new_sec.embedding is not None:
        sim = float(cosine_similarity(
            old_sec.embedding.reshape(1, -1),
            new_sec.embedding.reshape(1, -1),
        )[0, 0])
        diff.distance_score = 1.0 - sim
    else:
        diff.distance_score = 0.0

    # length_delta
    n_old = len(old_sec.sentences)
    n_new = len(new_sec.sentences)
    diff.length_delta = abs(n_new - n_old) / max(n_old, 1)

    # category_drift
    diff.category_drift = float(
        old_sec.dominant_category != new_sec.dominant_category
        and old_sec.dominant_category is not None
        and new_sec.dominant_category is not None
    )

    # risk_delta (opcjonalnie, jeśli nlp dostępne)
    if nlp is not None:
        from .features import extract_features, calculate_risk_score
        risk_old = calculate_risk_score(extract_features(" ".join(old_sec.sentences), nlp), cfg_path)
        risk_new = calculate_risk_score(extract_features(" ".join(new_sec.sentences), nlp), cfg_path)
        diff.risk_delta = abs(risk_new - risk_old) / 100.0
    else:
        diff.risk_delta = 0.0

    # sentence-level diff
    added_removed = _sent_level_diff(old_sec.sentences, new_sec.sentences, encoder, unch_thr, repr_thr)
    removed_extra = _removed_sentences(old_sec.sentences, new_sec.sentences, encoder, unch_thr, repr_thr)
    diff.sentence_changes = added_removed + removed_extra

    diff.n_added = sum(1 for c in added_removed if c.status == "added")
    diff.n_rephrased = sum(1 for c in added_removed if c.status == "rephrased")
    diff.n_unchanged = sum(1 for c in added_removed if c.status == "unchanged")
    diff.n_removed = len(removed_extra)

    # new_sentences_ratio, removed_sentences_ratio
    n_total = max(n_old, n_new, 1)
    new_ratio = diff.n_added / n_total
    rem_ratio = diff.n_removed / n_total

    diff.change_magnitude = min(100.0, 100.0 * (
        weights["distance_score"] * diff.distance_score
        + weights["length_delta"] * min(diff.length_delta, 1.0)
        + weights["new_sentences_ratio"] * new_ratio
        + weights["removed_sentences_ratio"] * rem_ratio
        + weights["category_drift"] * diff.category_drift
    ))

    return diff


def compute_diff(
    chains: list[ThematicChain],
    encoder,
    nlp=None,
    cfg_path: str = "config.yaml",
) -> list[dict]:
    """
    Dla każdego łańcucha i każdej pary kolejnych lat oblicza SectionDiff.
    Zwraca listę słowników: {chain_id, label, year_old, year_new, diff: SectionDiff}.
    """
    results = []
    for chain in chains:
        years = sorted(chain.sections.keys())
        for i in range(len(years) - 1):
            yr_old, yr_new = years[i], years[i + 1]
            sec_old = chain.sections.get(yr_old)
            sec_new = chain.sections.get(yr_new)
            if sec_old is None or sec_new is None:
                continue
            d = compute_section_diff(sec_old, sec_new, yr_old, yr_new, encoder, nlp, cfg_path)
            results.append({
                "chain_id": chain.chain_id,
                "label": chain.label,
                "year_old": yr_old,
                "year_new": yr_new,
                "diff": d,
            })
    return results
