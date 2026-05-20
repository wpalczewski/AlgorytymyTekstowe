"""
Streamlit dashboard — uruchom z roota repo:
    streamlit run privacy_analyzer/app.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Dodaj root repo do sys.path żeby import privacy_analyzer działał
# niezależnie od tego skąd uruchomiono streamlit
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Privacy Policy Analyzer", layout="wide")
st.title("Privacy Policy Change Analyzer")

# ---------------------------------------------------------------------------
# Modele ładowane raz (cache_resource = singleton przez całe życie serwera)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_nlp(model_name: str):
    import spacy
    return spacy.load(model_name)


@st.cache_resource
def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource
def load_classifier(cfg_path: str, csv_path: str):
    from privacy_analyzer.classifier import OPP115Classifier
    c = OPP115Classifier(cfg_path)
    if Path(csv_path).exists():
        c.train(csv_path, eval=False)
    else:
        st.warning(f"OPP-115 CSV nie znaleziony: {csv_path} — strategia B niedostępna.")
    return c


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload Policies")
    uploaded = st.file_uploader(
        "Wgraj pliki polityk (.txt)",
        type=["txt"],
        accept_multiple_files=True,
    )
    cfg_path = st.text_input("Config path", value="config.yaml")
    opp_csv = st.text_input("OPP-115 CSV", value="data/multilabel_opp115.csv")
    run_btn = st.button("Analyze", type="primary", disabled=len(uploaded or []) < 2)

if not uploaded:
    st.info("Wgraj co najmniej dwa pliki polityk (.txt) w panelu bocznym, potem kliknij Analyze.")
    st.stop()

if len(uploaded) < 2:
    st.warning("Potrzebne co najmniej 2 pliki.")
    st.stop()

# ---------------------------------------------------------------------------
# Analiza
# ---------------------------------------------------------------------------
if run_btn:
    import yaml
    from privacy_analyzer.segmentation import segment_policy
    from privacy_analyzer.matching import build_chains
    from privacy_analyzer.diff import compute_diff
    from privacy_analyzer.cli import _extract_year

    if not Path(cfg_path).exists():
        st.error(f"Nie znaleziono {cfg_path}. Uruchom z roota repo.")
        st.stop()

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    with st.spinner("Ładowanie modeli..."):
        nlp = load_nlp(cfg["model"]["spacy"])
        encoder = load_encoder(cfg["model"]["sentence_transformer"])
        classifier = load_classifier(cfg_path, opp_csv)

    policies = {}
    progress = st.progress(0, text="Segmentacja...")
    for i, uf in enumerate(uploaded):
        year = _extract_year(uf.name)
        progress.progress((i + 1) / len(uploaded), text=f"Segmentuję {uf.name} ({year})...")
        text = uf.read().decode("utf-8")
        sections = segment_policy(text, nlp, classifier, encoder, cfg_path)
        policies[year] = sections

    progress.empty()

    with st.spinner("Matching + diff..."):
        chains = build_chains(policies, encoder, cfg_path)
        diff_results = compute_diff(chains, encoder, nlp, cfg_path)

    st.session_state["diff_results"] = diff_results
    st.session_state["chains"] = chains
    st.success(f"Gotowe: {len(chains)} osi tematycznych, {len(diff_results)} par sekcja×okres.")

# ---------------------------------------------------------------------------
# Wyniki
# ---------------------------------------------------------------------------
if "diff_results" not in st.session_state:
    st.stop()

diff_results: list = st.session_state["diff_results"]

if not diff_results:
    st.warning("Brak wyników — być może żadne sekcje nie zostały dopasowane (sprawdź próg min_similarity w config.yaml).")
    st.stop()

# Heatmapa
years_all = sorted(
    {item["year_old"] for item in diff_results} | {item["year_new"] for item in diff_results}
)
labels_all = sorted({item["label"][:50] for item in diff_results})

heat_data = {
    yr: {
        item["label"][:50]: round(item["diff"].change_magnitude, 1)
        for item in diff_results
        if item["year_new"] == yr
    }
    for yr in years_all[1:]
}

df_heat = pd.DataFrame(heat_data, index=labels_all).fillna(0)

st.header("Change Heatmap")
st.caption("0–100: im wyżej, tym większa zmiana sekcji między wersjami")
st.dataframe(
    df_heat.style.background_gradient(cmap="Reds", axis=None, vmin=0, vmax=100),
    use_container_width=True,
)

# Drill-down
st.header("Section Drill-Down")

sorted_items = sorted(diff_results, key=lambda x: -x["diff"].change_magnitude)
options = [
    f"{item['label'][:50]}  ({item['year_old']}→{item['year_new']})  [{item['diff'].change_magnitude:.0f}/100]"
    for item in sorted_items
]
selected_idx = st.selectbox("Wybierz sekcję (posortowane wg zmiany):", range(len(options)), format_func=lambda i: options[i])

if selected_idx is not None:
    chosen = sorted_items[selected_idx]
    d = chosen["diff"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Change Magnitude", f"{d.change_magnitude:.1f}/100")
    col2.metric("Distance Score", f"{d.distance_score:.3f}")
    col3.metric("Length Δ", f"{d.length_delta:.2f}")
    col4.metric("Category Drift", "yes" if d.category_drift else "no")
    col5.metric("Risk Δ", f"{d.risk_delta:.2f}")

    tab_diff, tab_stats = st.tabs(["Sentence diff", "Statystyki"])

    with tab_diff:
        st.caption("🟢 dodane  🔴 usunięte  🟡 przeformułowane  ⚪ bez zmian")
        icons = {"added": "🟢", "removed": "🔴", "rephrased": "🟡", "unchanged": "⚪"}
        for change in d.sentence_changes:
            st.markdown(f"{icons[change.status]} {change.text[:250]}")

    with tab_stats:
        stats_data = {
            "Status": ["added", "removed", "rephrased", "unchanged"],
            "Count": [d.n_added, d.n_removed, d.n_rephrased, d.n_unchanged],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
