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

import difflib
import html
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Privacy Policy Analyzer", layout="wide")
st.title("Privacy Policy Change Analyzer")

def word_level_diff(old_text: str, new_text: str) -> tuple[str, str]:
    token_pattern = re.compile(r'\w+|[^\w\s]')
    old_tokens = token_pattern.findall(old_text)
    new_tokens = token_pattern.findall(new_text)
    
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens)
    old_html = []
    new_html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            chunk = " ".join(old_tokens[i1:i2])
            old_html.append(chunk)
            new_html.append(chunk)
        elif tag == 'delete':
            chunk = " ".join(old_tokens[i1:i2])
            old_html.append(f'<span style="background-color: #ffeef0; color: #b31d28; font-weight: bold; text-decoration: line-through; padding: 1px 3px; border-radius: 3px;">{chunk}</span>')
        elif tag == 'insert':
            chunk = " ".join(new_tokens[j1:j2])
            new_html.append(f'<span style="background-color: #e6ffed; color: #22863a; font-weight: bold; padding: 1px 3px; border-radius: 3px;">{chunk}</span>')
        elif tag == 'replace':
            chunk_old = " ".join(old_tokens[i1:i2])
            chunk_new = " ".join(new_tokens[j1:j2])
            old_html.append(f'<span style="background-color: #ffeef0; color: #b31d28; font-weight: bold; text-decoration: line-through; padding: 1px 3px; border-radius: 3px;">{chunk_old}</span>')
            new_html.append(f'<span style="background-color: #e6ffed; color: #22863a; font-weight: bold; padding: 1px 3px; border-radius: 3px;">{chunk_new}</span>')
            
    def clean_join(parts):
        text = " ".join(parts)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()
        
    return clean_join(old_html), clean_join(new_html)


def highlight_keywords(text: str, nlp) -> str:
    import html
    from privacy_analyzer.features import POWER_WORDS
    
    doc = nlp(text)
    
    passive_indices = set()
    neg_indices = set()
    for token in doc:
        if token.dep_ == "auxpass":
            passive_indices.add(token.i)
            passive_indices.add(token.head.i)
        if token.dep_ == "neg":
            neg_indices.add(token.i)
            
    html_parts = []
    for token in doc:
        lemma = token.lemma_.lower()
        txt = html.escape(token.text)
        ws = html.escape(token.whitespace_)
        
        category = None
        if lemma in POWER_WORDS["NEGATION"] or token.i in neg_indices:
            category = "negation"
        elif lemma in POWER_WORDS["MODAL_MUST"]:
            category = "modal_must"
        elif lemma in POWER_WORDS["MODAL_MAY"]:
            category = "modal_may"
        elif lemma in POWER_WORDS["EXCEPTIONS"]:
            category = "exceptions"
        elif lemma in POWER_WORDS["SENSITIVE_DATA"]:
            category = "sensitive_data"
        elif token.i in passive_indices:
            category = "passive_voice"
            
        if category:
            colors = {
                "negation": ("#5c1d24", "#ff9999", "solid"),
                "modal_must": ("#5c4300", "#ffcc00", "solid"),
                "modal_may": ("#1d5c22", "#99ff99", "solid"),
                "exceptions": ("#3a1d5c", "#d6adff", "solid"),
                "sensitive_data": ("#005060", "#99f5ff", "double"),
                "passive_voice": ("#1d3c5c", "#99ccff", "dashed"),
            }
            bg, fg, border_style = colors[category]
            style = f"background-color: {bg}; color: {fg}; font-weight: bold; border-bottom: 2px {border_style} {fg}; padding: 1px 2px; border-radius: 2px;"
            html_parts.append(f'<span style="{style}">{txt}</span>{ws}')
        else:
            html_parts.append(txt + ws)
            
    return "".join(html_parts)


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
        "Upload policies (.txt)",
        type=["txt"],
        accept_multiple_files=True,
    )
    cfg_path = st.text_input("Config path", value="config.yaml")
    opp_csv = st.text_input("OPP-115 CSV", value="data/multilabel_opp115.csv")
    run_btn = st.button("Analyze", type="primary", disabled=len(uploaded or []) < 2)

if not uploaded:
    st.info("Upload atleast 2 files (.txt) on a sidebar, then press Analyze.")
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
    raw_policies = {}
    progress = st.progress(0, text="Segmentacja...")
    for i, uf in enumerate(uploaded):
        year = _extract_year(uf.name)
        progress.progress((i + 1) / len(uploaded), text=f"Segmentuję {uf.name} ({year})...")
        text = uf.read().decode("utf-8")
        raw_policies[year] = text
        sections = segment_policy(text, nlp, classifier, encoder, cfg_path)
        policies[year] = sections

    progress.empty()

    with st.spinner("Matching + diff..."):
        chains = build_chains(policies, encoder, cfg_path)
        diff_results = compute_diff(chains, encoder, nlp, cfg_path)

    st.session_state["diff_results"] = diff_results
    st.session_state["chains"] = chains
    st.session_state["raw_policies"] = raw_policies
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
        item["label"][:50]: round(item["diff"].change_magnitude, 2)
        for item in diff_results
        if item["year_new"] == yr
    }
    for yr in years_all[1:]
}

df_heat = pd.DataFrame(heat_data, index=labels_all).fillna(0)

st.header("Change Heatmap")
st.caption("0–100: im wyżej, tym większa zmiana sekcji między wersjami")
st.dataframe(
    df_heat.style.background_gradient(cmap="Reds", axis=None, vmin=0, vmax=100).format(precision=2),
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
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        nlp_model = load_nlp(cfg["model"]["spacy"])

        old_text = getattr(d, "old_text", "")
        new_text = getattr(d, "new_text", "")
        
        if not old_text or not new_text:
            old_sents = []
            new_sents = []
            for c in d.sentence_changes:
                if c.status == "unchanged":
                    old_sents.append(c.text)
                    new_sents.append(c.text)
                elif c.status == "rephrased":
                    old_sents.append(c.best_match or "")
                    new_sents.append(c.text)
                elif c.status == "added":
                    new_sents.append(c.text)
                elif c.status == "removed":
                    old_sents.append(c.text)
            old_text = " ".join([s for s in old_sents if s])
            new_text = " ".join([s for s in new_sents if s])

        html_old = highlight_keywords(old_text, nlp_model)
        html_new = highlight_keywords(new_text, nlp_model)

        st.markdown(
            """
            <div style="margin-bottom: 15px; padding: 12px; border: 1px solid #333; border-radius: 6px; background-color: #121212; color: #e0e0e0;">
              <strong>Risk Highlights:</strong><br>
              <span style="background-color: #5c1d24; color: #ff9999; font-weight: bold; border-bottom: 2px solid #ff9999; padding: 1px 4px; border-radius: 2px; white-space: nowrap;">Negations</span> | 
              <span style="background-color: #5c4300; color: #ffcc00; font-weight: bold; border-bottom: 2px solid #ffcc00; padding: 1px 4px; border-radius: 2px; white-space: nowrap;">Obligations (must/shall)</span> | 
              <span style="background-color: #1d5c22; color: #99ff99; font-weight: bold; border-bottom: 2px solid #99ff99; padding: 1px 4px; border-radius: 2px; white-space: nowrap;">Permissions (may/can)</span> | 
              <span style="background-color: #3a1d5c; color: #d6adff; font-weight: bold; border-bottom: 2px solid #d6adff; padding: 1px 4px; border-radius: 2px; white-space: nowrap;">Exceptions</span> | 
              <span style="background-color: #005060; color: #99f5ff; font-weight: bold; border-bottom: 2px double #99f5ff; padding: 1px 4px; border-radius: 2px; white-space: nowrap;">Sensitive Data</span> | 
              <span style="background-color: #1d3c5c; color: #99ccff; font-weight: bold; border-bottom: 2px dashed #99ccff; padding: 1px 4px; border-radius: 2px; white-space: nowrap;">Passive Voice</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_old_sec, col_new_sec = st.columns(2)
        with col_old_sec:
            st.subheader(f"Version {chosen['year_old']}")
            st.markdown(
                f'<div style="border: 1px solid #333; border-radius: 6px; padding: 15px; background-color: #0f0f0f; min-height: 300px; font-size: 14px; line-height: 1.6; color: #e0e0e0;">{html_old}</div>',
                unsafe_allow_html=True
            )
        with col_new_sec:
            st.subheader(f"Version {chosen['year_new']}")
            st.markdown(
                f'<div style="border: 1px solid #333; border-radius: 6px; padding: 15px; background-color: #0f0f0f; min-height: 300px; font-size: 14px; line-height: 1.6; color: #e0e0e0;">{html_new}</div>',
                unsafe_allow_html=True
            )
            


    with tab_stats:
        stats_data = {
            "Status": ["added", "removed", "rephrased", "unchanged"],
            "Count": [d.n_added, d.n_removed, d.n_rephrased, d.n_unchanged],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    # Pełne teksty polityk (Side-by-Side)
    if "raw_policies" in st.session_state:
        st.write("---")
        st.header("Full versions side by side")
        
        y_old = chosen["year_old"]
        y_new = chosen["year_new"]
        
        raw_old = st.session_state["raw_policies"].get(y_old, "")
        raw_new = st.session_state["raw_policies"].get(y_new, "")
        
        col_old, col_new = st.columns(2)
        with col_old:
            st.subheader(f"Version {y_old}")
            st.text_area(
                label=f"Pełny tekst wersji {y_old}",
                value=raw_old,
                height=500,
                disabled=True,
                label_visibility="collapsed"
            )
        with col_new:
            st.subheader(f"Version {y_new}")
            st.text_area(
                label=f"Pełny tekst wersji {y_new}",
                value=raw_new,
                height=500,
                disabled=True,
                label_visibility="collapsed"
            )

