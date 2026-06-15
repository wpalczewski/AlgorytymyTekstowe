"""
Generowanie raportu z wyników analizy.

Etap 1: markdown
Etap 2: HTML z kolorowaniem (added=zielony, removed=czerwony, rephrased=żółty)
Streamlit w app.py (osobno).
"""

from __future__ import annotations
import json
from pathlib import Path
from .diff import SectionDiff


_STATUS_COLOR = {
    "added": "#c6efce",
    "removed": "#ffc7ce",
    "rephrased": "#ffeb9c",
    "unchanged": "#ffffff",
}

_STATUS_LABEL = {
    "added": "[+]",
    "removed": "[-]",
    "rephrased": "[~]",
    "unchanged": "[ ]",
}


def _magnitude_bar(value: float, width: int = 20) -> str:
    filled = int(round(value / 100 * width))
    return "█" * filled + "░" * (width - filled)


def generate_markdown(diff_results: list[dict], output_path: str = "report.md") -> str:
    lines = [
        "# Privacy Policy Change Report\n",
        "## Summary\n",
    ]

    # Top 10 zmian
    sorted_diffs = sorted(diff_results, key=lambda x: x["diff"].change_magnitude, reverse=True)
    lines.append("### Top 10 largest changes\n")
    lines.append("| Rank | Section | Period | Change Magnitude |")
    lines.append("|------|---------|--------|-----------------|")
    for i, item in enumerate(sorted_diffs[:10], 1):
        d: SectionDiff = item["diff"]
        bar = _magnitude_bar(d.change_magnitude)
        lines.append(f"| {i} | {item['label'][:60]} | {item['year_old']}→{item['year_new']} | {d.change_magnitude:.1f}/100 `{bar}` |")

    lines.append("\n---\n")
    lines.append("## Section-by-Section Analysis\n")

    # Grupuj wg chain_id
    by_chain: dict[int, list[dict]] = {}
    for item in diff_results:
        by_chain.setdefault(item["chain_id"], []).append(item)

    for chain_id, items in sorted(by_chain.items()):
        label = items[0]["label"]
        lines.append(f"### {label}\n")
        for item in items:
            d: SectionDiff = item["diff"]
            lines.append(f"**{item['year_old']} → {item['year_new']}**  ")
            lines.append(f"Change magnitude: `{d.change_magnitude:.1f}/100`  ")
            lines.append(f"Distance: `{d.distance_score:.3f}` | "
                         f"Length Δ: `{d.length_delta:.2f}` | "
                         f"Category drift: `{'yes' if d.category_drift else 'no'}`\n")
            lines.append(f"Sentences: +{d.n_added} added, -{d.n_removed} removed, ~{d.n_rephrased} rephrased\n")

            # Sentence-level
            lines.append("<details><summary>Sentence diff</summary>\n")
            lines.append("```")
            for change in item["diff"].sentence_changes:
                prefix = _STATUS_LABEL[change.status]
                lines.append(f"{prefix} {change.text[:120]}")
            lines.append("```")
            lines.append("</details>\n")

    content = "\n".join(lines)
    Path(output_path).write_text(content, encoding="utf-8")
    print(f"Raport markdown zapisany: {output_path}")
    return content


def generate_html(diff_results: list[dict], output_path: str = "report.html") -> str:
    sorted_diffs = sorted(diff_results, key=lambda x: x["diff"].change_magnitude, reverse=True)

    # Heatmapa — zbieramy wszystkie lata i etykiety
    labels = list({item["label"][:50] for item in diff_results})
    years_set = sorted({item["year_old"] for item in diff_results} | {item["year_new"] for item in diff_results})

    # magnitude per (label, year_new)
    mag_map: dict[tuple, float] = {}
    for item in diff_results:
        mag_map[(item["label"][:50], item["year_new"])] = item["diff"].change_magnitude

    def _heatmap_cell(label: str, year: str) -> str:
        mag = mag_map.get((label, year))
        if mag is None:
            return '<td style="background:#eee">—</td>'
        r = int(255 - mag * 1.5)
        g = int(255 - mag * 0.5)
        b = 200
        return f'<td style="background:rgb({r},{g},{b});text-align:center">{mag:.0f}</td>'

    heatmap_rows = ""
    for lbl in sorted(labels):
        cells = "".join(_heatmap_cell(lbl, yr) for yr in years_set[1:])
        heatmap_rows += f"<tr><td><b>{lbl}</b></td>{cells}</tr>\n"

    year_headers = "".join(f"<th>{yr}</th>" for yr in years_set[1:])

    # Drill-down sections
    drill_sections = ""
    by_chain: dict[int, list[dict]] = {}
    for item in diff_results:
        by_chain.setdefault(item["chain_id"], []).append(item)

    for chain_id, items in sorted(by_chain.items()):
        label = items[0]["label"]
        drill_sections += f"<h3>{label}</h3>\n"
        for item in items:
            d: SectionDiff = item["diff"]
            drill_sections += f"<p><b>{item['year_old']} → {item['year_new']}</b> | "
            drill_sections += f"Change: <b>{d.change_magnitude:.1f}/100</b> | "
            drill_sections += f"Dist: {d.distance_score:.3f} | "
            drill_sections += f"Len Δ: {d.length_delta:.2f} | "
            drill_sections += f"Cat drift: {'yes' if d.category_drift else 'no'}</p>\n"
            drill_sections += "<div style='font-family:monospace;font-size:0.85em;margin-left:1em'>\n"
            for change in item["diff"].sentence_changes:
                color = _STATUS_COLOR[change.status]
                label_tag = _STATUS_LABEL[change.status]
                text = change.text.replace("<", "&lt;").replace(">", "&gt;")
                drill_sections += (
                    f'<div style="background:{color};padding:2px 6px;margin:1px 0">'
                    f'{label_tag} {text[:200]}</div>\n'
                )
            drill_sections += "</div>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Privacy Policy Change Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
    th {{ background: #333; color: white; }}
    h2 {{ border-bottom: 2px solid #333; }}
    details {{ margin: 4px 0; }}
  </style>
</head>
<body>
<h1>Privacy Policy Change Report</h1>

<h2>Change Heatmap</h2>
<p>Color: darker red = higher change magnitude (0–100). Rows = thematic sections, Columns = target year.</p>
<table>
  <tr><th>Section</th>{year_headers}</tr>
  {heatmap_rows}
</table>

<h2>Top 10 Largest Changes</h2>
<ol>
{"".join(f'<li><b>{item["label"][:60]}</b> ({item["year_old"]}→{item["year_new"]}): {item["diff"].change_magnitude:.1f}/100</li>' for item in sorted_diffs[:10])}
</ol>

<h2>Section Drill-Down</h2>
{drill_sections}
</body>
</html>
"""
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Raport HTML zapisany: {output_path}")
    return html


def generate_report(
    diff_results: list[dict],
    output_dir: str = "out",
    formats: list[str] = ("markdown", "html"),
) -> dict[str, str]:
    """Generuje raporty we wskazanych formatach. Zwraca ścieżki do plików."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    if "markdown" in formats:
        p = str(Path(output_dir) / "report.md")
        generate_markdown(diff_results, p)
        paths["markdown"] = p
    if "html" in formats:
        p = str(Path(output_dir) / "report.html")
        generate_html(diff_results, p)
        paths["html"] = p
    return paths
