"""
Dopasowanie sekcji między wersjami polityk.

Dla każdej pary (stara, nowa) buduje macierz cosine similarity i stosuje
algorytm węgierski (scipy.optimize.linear_sum_assignment) → bijekcja sekcji.
Sekcje o similarity < min_similarity traktowane jako nowe/usunięte.

Dla N>2 polityk buduje chains: łańcuchy sekcji przez wszystkie lata.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import yaml

from .segmentation import Section


def _load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def embed_sections(sections: list[Section], encoder, title_weight: float = 2.0) -> np.ndarray:
    """
    Oblicza embedding każdej sekcji jako:
      (suma embeddingów zdań + title_weight * embedding tytułu) / (n_sents + title_weight)
    Zapisuje wynik do section.embedding i zwraca macierz (n_sections, dim).
    """
    all_texts = []
    for sec in sections:
        all_texts.extend(sec.sentences)
        all_texts.append(sec.title)

    all_embs = encoder.encode(all_texts, show_progress_bar=False)

    result = []
    idx = 0
    for sec in sections:
        n = len(sec.sentences)
        sent_embs = all_embs[idx: idx + n]
        title_emb = all_embs[idx + n]
        idx += n + 1

        if n == 0:
            combined = title_emb
        else:
            combined = (sent_embs.sum(axis=0) + title_weight * title_emb) / (n + title_weight)

        sec.embedding = combined
        result.append(combined)

    return np.array(result)


@dataclass
class SectionMatch:
    old_idx: Optional[int]          # None → sekcja nowa (brak w starej)
    new_idx: Optional[int]          # None → sekcja usunięta (brak w nowej)
    old_section: Optional[Section]
    new_section: Optional[Section]
    similarity: float = 0.0
    is_new: bool = False            # sekcja bez odpowiednika w starej wersji
    is_removed: bool = False        # sekcja bez odpowiednika w nowej wersji


def match_two(
    old_sections: list[Section],
    new_sections: list[Section],
    encoder,
    cfg_path: str = "config.yaml",
) -> list[SectionMatch]:
    """
    Dopasowuje sekcje starej i nowej polityki metodą węgierską.
    Zwraca listę SectionMatch (zawiera też sekcje nowe/usunięte).
    """
    cfg = _load_cfg(cfg_path)
    min_sim = cfg["matching"]["min_similarity"]
    title_w = cfg["matching"]["section_title_weight"]

    old_embs = embed_sections(old_sections, encoder, title_w)
    new_embs = embed_sections(new_sections, encoder, title_w)

    sim_matrix = cosine_similarity(old_embs, new_embs)          # (n_old, n_new)
    cost_matrix = 1.0 - sim_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_old = set()
    matched_new = set()
    matches: list[SectionMatch] = []

    for r, c in zip(row_ind, col_ind):
        sim = float(sim_matrix[r, c])
        if sim >= min_sim:
            matches.append(SectionMatch(
                old_idx=r, new_idx=c,
                old_section=old_sections[r],
                new_section=new_sections[c],
                similarity=sim,
            ))
            matched_old.add(r)
            matched_new.add(c)

    # Sekcje usunięte (w starej, bez pary)
    for r in range(len(old_sections)):
        if r not in matched_old:
            matches.append(SectionMatch(
                old_idx=r, new_idx=None,
                old_section=old_sections[r], new_section=None,
                similarity=0.0, is_removed=True,
            ))

    # Sekcje nowe (w nowej, bez pary)
    for c in range(len(new_sections)):
        if c not in matched_new:
            matches.append(SectionMatch(
                old_idx=None, new_idx=c,
                old_section=None, new_section=new_sections[c],
                similarity=0.0, is_new=True,
            ))

    return matches


@dataclass
class ThematicChain:
    """Oś tematyczna sekcji przez kolejne lata polityki."""
    chain_id: int
    label: str                              # tytuł pierwszej sekcji lub kategoria OPP-115
    sections: dict[str, Optional[Section]]  # rok → sekcja (None = brak w danej wersji)
    similarities: dict[str, float] = field(default_factory=dict)  # "2010→2015" → sim


def build_chains(
    policies: dict[str, list[Section]],   # {"2010": [...], "2015": [...], "2020": [...]}
    encoder,
    cfg_path: str = "config.yaml",
) -> list[ThematicChain]:
    """
    Dla N polityk (posortowanych rosnąco wg roku) buduje łańcuchy sekcji.
    Dopasowuje parami: rok[0]↔rok[1], rok[1]↔rok[2], itd.
    Zwraca listę ThematicChain.
    """
    years = sorted(policies.keys())
    if len(years) < 2:
        if years:
            return [
                ThematicChain(
                    chain_id=i,
                    label=sec.title or sec.dominant_category or f"Section {i}",
                    sections={years[0]: sec},
                )
                for i, sec in enumerate(policies[years[0]])
            ]
        return []

    # Inicjalizacja łańcuchów z pierwszego roku
    chains: list[ThematicChain] = []
    for i, sec in enumerate(policies[years[0]]):
        chains.append(ThematicChain(
            chain_id=i,
            label=sec.title or sec.dominant_category or f"Section {i}",
            sections={years[0]: sec},
        ))
    # Śledzimy które chain_id są "aktywne" (mają sekcję w poprzednim roku)
    chain_map: dict[int, int] = {i: i for i in range(len(chains))}  # old_idx → chain_id

    for y_idx in range(len(years) - 1):
        yr_old, yr_new = years[y_idx], years[y_idx + 1]
        old_secs = policies[yr_old]
        new_secs = policies[yr_new]

        matches = match_two(old_secs, new_secs, encoder, cfg_path)
        edge_key = f"{yr_old}→{yr_new}"

        new_chain_map: dict[int, int] = {}

        for m in matches:
            if m.is_removed:
                # Sekcja znika — łańcuch ma None dla tego i kolejnych lat
                if m.old_idx in chain_map:
                    cid = chain_map[m.old_idx]
                    chains[cid].sections[yr_new] = None
            elif m.is_new:
                # Nowa sekcja → nowy łańcuch
                cid = len(chains)
                chains.append(ThematicChain(
                    chain_id=cid,
                    label=m.new_section.title or m.new_section.dominant_category or f"New {cid}",
                    sections={yr_new: m.new_section},
                ))
                new_chain_map[m.new_idx] = cid
            else:
                # Dopasowanie: kontynuuj łańcuch
                if m.old_idx in chain_map:
                    cid = chain_map[m.old_idx]
                    chains[cid].sections[yr_new] = m.new_section
                    chains[cid].similarities[edge_key] = m.similarity
                    new_chain_map[m.new_idx] = cid

        chain_map = new_chain_map

    return chains
