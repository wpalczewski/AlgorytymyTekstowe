"""Wspólne fixtures: fake encoder (bag-of-words) zamiast Sentence-Transformers,
blank spaCy z sentencizerem zamiast en_core_web_sm — testy są szybkie i offline."""

import re
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent


class FakeEncoder:
    """Deterministyczny encoder: hashowany bag-of-words. Identyczne teksty → cos=1,
    teksty bez wspólnych słów → cos=0."""

    dim = 256

    def encode(self, texts, show_progress_bar=False, **kwargs):
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return out


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def cfg_path():
    return str(REPO_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def nlp():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp
