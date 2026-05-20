"""
Multilabel klasyfikator OPP-115 na bazie TF-IDF + OneVsRest LogisticRegression.
Wytrenowany na data/multilabel_opp115.csv (dawne Multilabel%20DS.csv).
"""

from __future__ import annotations
import ast
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import yaml


def _load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _parse_label_cell(cell) -> list[str]:
    if pd.isna(cell) or cell == "Null":
        return []
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return [v for v in parsed if v != "Unspecified"]
        return []
    except (ValueError, SyntaxError):
        return []


class OPP115Classifier:
    def __init__(self, cfg_path: str = "config.yaml"):
        cfg = _load_cfg(cfg_path)["classifier"]
        self.mlb = MultiLabelBinarizer()
        self.vectorizer = TfidfVectorizer(
            max_features=cfg["tfidf_max_features"],
            stop_words="english",
            ngram_range=tuple(cfg["tfidf_ngram_range"]),
        )
        self.model = OneVsRestClassifier(
            LogisticRegression(max_iter=cfg["logistic_max_iter"], class_weight="balanced")
        )
        self.threshold = cfg["prediction_threshold"]
        self._trained = False
        self.classes_: list[str] = []

    def train(self, csv_path: str, eval: bool = True) -> dict:
        """
        Trenuje klasyfikator na pliku CSV.
        Zwraca słownik z metrykami (micro/macro F1) jeśli eval=True.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Brak pliku: {csv_path}")

        df = pd.read_csv(csv_path).dropna(subset=["Text"])
        df["labels"] = df["Category"].apply(_parse_label_cell)
        df = df[df["labels"].map(len) > 0]

        X = df["Text"].tolist()
        y = self.mlb.fit_transform(df["labels"])
        self.classes_ = list(self.mlb.classes_)

        metrics = {}
        if eval:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_tr_vec = self.vectorizer.fit_transform(X_train)
            X_te_vec = self.vectorizer.transform(X_test)
            self.model.fit(X_tr_vec, y_train)
            y_pred = self._predict_with_threshold(X_te_vec)
            metrics["micro_f1"] = float(f1_score(y_test, y_pred, average="micro", zero_division=0))
            metrics["macro_f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            metrics["n_train"] = len(X_train)
            metrics["n_test"] = len(X_test)
            # Retrenuj na całości danych po ewaluacji
            X_vec = self.vectorizer.fit_transform(X)
            self.model.fit(X_vec, self.mlb.transform(df["labels"]))
        else:
            X_vec = self.vectorizer.fit_transform(X)
            self.model.fit(X_vec, y)

        self._trained = True
        print(f"Klasyfikator OPP-115 gotowy. Kategorie: {len(self.classes_)}, próbki: {len(df)}")
        if metrics:
            print(f"  micro F1={metrics['micro_f1']:.3f}  macro F1={metrics['macro_f1']:.3f}")
        return metrics

    def classify_text(self, text: str) -> list[str]:
        if not self._trained:
            return []
        X = self.vectorizer.transform([text])
        labels = self._predict_with_threshold(X)
        return list(self.mlb.inverse_transform(labels)[0])

    def classify_batch(self, texts: list[str]) -> list[list[str]]:
        if not self._trained:
            return [[] for _ in texts]
        X = self.vectorizer.transform(texts)
        labels = self._predict_with_threshold(X)
        return [list(row) for row in self.mlb.inverse_transform(labels)]

    def _predict_with_threshold(self, X_vec):
        proba = self.model.predict_proba(X_vec)
        return (proba >= self.threshold).astype(int)
