# Privacy Policy Analyzer

Narzędzie do porównywania polityk prywatności z różnych lat. Automatycznie segmentuje dokumenty na sekcje tematyczne, dopasowuje je między latami algorytmem węgierskim i wykrywa miejsca, gdzie nastąpiły największe zmiany.

## Pipeline

```
Wczytaj .txt polityk  →  Segmentacja na sekcje (A/B/C)  →  Matching cross-version  →  Sentence-level diff  →  Raport HTML/Markdown
```

**Krok 1 — Segmentacja** (`privacy_analyzer/segmentation.py`): kaskada 3 strategii
- **A** — heurystyka strukturalna: nagłówki, "Return to top", krótkie linie kończące się `?`
- **B** — klasyfikacja semantyczna OPP-115: ciągłe bloki zdań tej samej kategorii (12 kategorii)
- **C** — TextTiling fallback: sliding-window podobieństwa embeddingów

**Krok 2 — Matching** (`privacy_analyzer/matching.py`): embedding sekcji (Sentence-Transformers), macierz cosine similarity, algorytm węgierski (`scipy.optimize.linear_sum_assignment`), łańcuchy dla N>2 polityk

**Krok 3 — Diff** (`privacy_analyzer/diff.py`): sentence-level diff (unchanged/rephrased/added/removed), `change_magnitude` 0-100 z distance score, length delta, category drift, risk delta

**Krok 4 — Raport** (`privacy_analyzer/report.py`): heatmapa sekcji×rok, top 10 zmian, drill-down z kolorowanymi zdaniami

## Instalacja

```bash
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Użycie

### CLI

```bash
# Dwie polityki (jeden okres)
python -m privacy_analyzer analyze data/policies/fb_2010.txt data/policies/fb_2015.txt --output-dir out/

# Wiele polityk z różnych lat (łańcuchy cross-version)
python -m privacy_analyzer analyze data/policies/fb_2010.txt data/policies/fb_2015.txt data/policies/fb_2020.txt --output-dir out/

# Opcje
python -m privacy_analyzer analyze *.txt --config config.yaml --opp-csv data/multilabel_opp115.csv --output-dir out/
```

Rok wykrywany z nazwy pliku (szuka 4 cyfr). Konwencja: `<nazwa>_<rok>.txt` lub `<nazwa><rok>.txt`.

### Streamlit dashboard

```bash
streamlit run privacy_analyzer/app.py
```

Wgraj pliki .txt w panelu bocznym → Analyze.

## Dane

- `data/policies/` — symlinki do plików w `data/`:
  - `fb_2010.txt`, `fb_2015.txt` — polityki prywatności Facebook
  - `tiktok_20.txt`, `tiktok_21.txt` — polityki TikTok
- `data/multilabel_opp115.csv` — symlink do zbioru OPP-115 (1748 próbek, 12 kategorii), używany do trenowania klasyfikatora segmentacji B

## Konfiguracja

Parametry w `config.yaml` — progi podobieństwa, wagi change_magnitude, rozmiary okna TextTiling, hiperparametry klasyfikatora.

## Struktura

```
privacy_analyzer/
  segmentation.py   # krok 1
  classifier.py     # OPP-115 multilabel (TF-IDF + LogReg)
  features.py       # risk score lingwistyczny (spaCy)
  matching.py       # krok 2: Hungarian algorithm
  diff.py           # krok 3: sentence diff + scoring
  report.py         # krok 4: HTML + Markdown
  cli.py            # entrypoint CLI
  app.py            # Streamlit dashboard
  __main__.py
data/
  policies/         # symlinki do .txt polityk
  multilabel_opp115.csv  # symlink do OPP-115
out/                # wyjście (ignorowane przez git)
config.yaml
requirements.txt
```
