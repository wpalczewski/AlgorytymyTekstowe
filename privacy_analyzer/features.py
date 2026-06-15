"""
Reguły lingwistyczne wyciągnięte z nlp_engine.py (V1).
Oblicza risk_score 0-100 dla pojedynczej sekcji lub tekstu.
"""

from __future__ import annotations
import yaml


def _load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


POWER_WORDS = {
    "NEGATION": {"not", "never", "no", "neither", "nor", "none", "cannot", "refuse"},
    "MODAL_MUST": {"must", "shall", "required", "obligated", "mandatory", "will"},
    "MODAL_MAY": {"may", "can", "entitled", "allowed", "permit", "could"},
    "EXCEPTIONS": {"unless", "except", "provided", "however", "subject", "notwithstanding", "condition"},
    "SENSITIVE_DATA": {"biometric", "faceprint", "voiceprint", "fingerprint", "location",
                       "tracking", "health", "genetic", "medical", "ssn", "passport"},
}


def extract_features(text: str, nlp) -> dict:
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    sents = list(doc.sents)

    features: dict = {
        "negations": [],
        "modals_must": [],
        "modals_may": [],
        "passive_voice": [],
        "exceptions": [],
        "sensitive_data": [],
        "readability": _readability(tokens, sents),
        "entities": {"dates": [], "money": [], "orgs": []},
    }

    for token in doc:
        lemma = token.lemma_.lower()
        if token.dep_ == "neg" or lemma in POWER_WORDS["NEGATION"]:
            features["negations"].append(f"{lemma}({token.head.lemma_})")
        if token.dep_ == "auxpass":
            features["passive_voice"].append(f"{token.lemma_} {token.head.lemma_}")
        if lemma in POWER_WORDS["MODAL_MUST"]:
            features["modals_must"].append(lemma)
        if lemma in POWER_WORDS["MODAL_MAY"]:
            features["modals_may"].append(lemma)
        if lemma in POWER_WORDS["EXCEPTIONS"]:
            features["exceptions"].append(lemma)
        if lemma in POWER_WORDS["SENSITIVE_DATA"]:
            features["sensitive_data"].append(lemma)

    for ent in doc.ents:
        if ent.label_ == "DATE":
            features["entities"]["dates"].append(ent.text)
        elif ent.label_ == "MONEY":
            features["entities"]["money"].append(ent.text)
        elif ent.label_ == "ORG":
            features["entities"]["orgs"].append(ent.text)

    return features


def _readability(tokens, sents) -> float:
    if not sents or not tokens:
        return 0.0
    avg_sent_len = len(tokens) / len(sents)
    complex_words = [w for w in tokens if len(w.text) > 9]
    pct_complex = (len(complex_words) / len(tokens)) * 100
    return avg_sent_len + pct_complex


def calculate_risk_score(features: dict, cfg_path: str = "config.yaml") -> int:
    cfg = _load_cfg(cfg_path)["risk"]
    score = 0
    score += len(features["negations"]) * cfg["negation_weight"]
    score += len(features["modals_must"]) * cfg["modal_must_weight"]
    score += len(features["passive_voice"]) * cfg["passive_voice_weight"]
    score += len(features["exceptions"]) * cfg["exception_weight"]
    score += len(set(features["sensitive_data"])) * cfg["sensitive_data_weight"]
    if features["readability"] > cfg["readability_penalty_threshold"]:
        score += cfg["readability_penalty"]
    if features["modals_may"] and features["sensitive_data"]:
        score += cfg["modal_may_plus_sensitive_bonus"]
    return min(score, 100)
