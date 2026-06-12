"""
CLI: python -m privacy_analyzer analyze data/policies/fb_*.txt --report out/report.html

Parsuje nazwy plików żeby wyciągnąć rok (szuka 4 cyfr w nazwie).
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path


def _extract_year(filename: str) -> str:
    m = re.search(r"(20\d{2}|19\d{2}|\d{2})", Path(filename).stem)
    if not m:
        return Path(filename).stem
    year = m.group(1)
    # znormalizuj 2-cyfrowy rok do 20XX, żeby sortowanie stringów było chronologiczne
    if len(year) == 2:
        year = "20" + year
    return year


def _load_models(cfg_path: str):
    import spacy
    import yaml
    from sentence_transformers import SentenceTransformer

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    try:
        nlp = spacy.load(cfg["model"]["spacy"])
    except OSError:
        print(f"Instaluję model spaCy {cfg['model']['spacy']}...")
        subprocess.run([sys.executable, "-m", "spacy", "download", cfg["model"]["spacy"]], check=True)
        nlp = spacy.load(cfg["model"]["spacy"])

    encoder = SentenceTransformer(cfg["model"]["sentence_transformer"])
    return nlp, encoder


def cmd_analyze(args):
    from .classifier import OPP115Classifier
    from .segmentation import segment_policy
    from .matching import build_chains
    from .diff import compute_diff
    from .report import generate_report

    cfg_path = args.config

    print("Ładowanie modeli...")
    nlp, encoder = _load_models(cfg_path)

    classifier = OPP115Classifier(cfg_path)
    opp_csv = Path(args.opp_csv)
    if opp_csv.exists():
        metrics = classifier.train(str(opp_csv))
    else:
        print(f"UWAGA: {opp_csv} nie znaleziony — strategia B niedostępna.")

    policies: dict[str, list] = {}
    for filepath in args.files:
        year = _extract_year(filepath)
        print(f"Segmentuję {filepath} (rok: {year})...")
        text = Path(filepath).read_text(encoding="utf-8")
        sections = segment_policy(text, nlp, classifier, encoder, cfg_path)
        print(f"  → {len(sections)} sekcji (strategia: {sections[0].strategy if sections else '?'})")
        policies[year] = sections

    if len(policies) < 2:
        print("Potrzebne co najmniej 2 pliki polityk. Wychodzę.")
        sys.exit(1)

    print("Budowanie łańcuchów tematycznych...")
    chains = build_chains(policies, encoder, cfg_path)
    print(f"  → {len(chains)} osi tematycznych")

    print("Obliczanie różnic...")
    diff_results = compute_diff(chains, encoder, nlp, cfg_path)
    print(f"  → {len(diff_results)} par sekcja×okres")

    formats = ["markdown", "html"]
    paths = generate_report(diff_results, output_dir=args.output_dir, formats=formats)
    print("\nGotowe!")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")


def main():
    parser = argparse.ArgumentParser(prog="privacy_analyzer")
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="Analizuj polityki prywatności")
    p_analyze.add_argument("files", nargs="+", help="Ścieżki do plików .txt polityk")
    p_analyze.add_argument("--config", default="config.yaml")
    p_analyze.add_argument("--opp-csv", default="data/multilabel_opp115.csv")
    p_analyze.add_argument("--output-dir", default="out")

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
