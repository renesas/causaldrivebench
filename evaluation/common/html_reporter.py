"""
HTML report generation for the CausalDriveBench evaluation pipeline.

Produces self-contained HTML files (no external CDN) in cdb_reports/, mirroring
the cdb_outputs/ structure.  All links between pages are relative, making the
entire tree portable for file:// browsing or static hosting.  CSS is loaded from
``templates/report.css`` at import time and embedded inline in every page.

Report hierarchy in cdb_reports/
----------------------------------
    index.html                              ← Tier 4: all runs
    <run>/
        index.html                          ← Tier 3: run + dataset table
        <dataset>/
            index.html                      ← Tier 2: dataset + scene table
            <scene>/
                index.html                  ← Tier 2.5: scene + sample table
                <sample>/
                    index.html              ← Tier 1: Q&A accordion + trajectory

PDF Download
------------
Every page has a "Download PDF" button that calls ``window.print()``.
``@media print`` in report.css hides the button, expands all ``<details>``
elements, and removes scroll limits.

Entry points
------------
``generate_all_html(run_name, cdb_outputs_dir, cdb_reports_dir, root_aggregate,
                    bench_dirs=None)``
    Regenerates the full HTML tree for one run and updates the root index.
    ``bench_dirs`` is an optional ``{dataset_name: Path}`` mapping used to load
    ``graph_structure.json`` for each sample.

``generate_root_index(root_aggregate, cdb_reports_dir)``
    Writes/rewrites only the root index.html.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.common.report_generator import load_report

logger = logging.getLogger(__name__)

# CSS is maintained in templates/report.css and embedded inline in every page.
_CSS = (Path(__file__).parent / "templates" / "report.css").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

def _html_page(title: str, breadcrumbs: List[tuple], body: str) -> str:
    crumbs = ""
    for i, (label, href) in enumerate(breadcrumbs):
        crumbs += f'<a href="{href}">{label}</a>' if href else f"<strong>{label}</strong>"
        if i < len(breadcrumbs) - 1:
            crumbs += "<span>›</span>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — CausalDriveBench</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page-header">
  <div class="page-header-left">
    <h1>CausalDriveBench Evaluation Report</h1>
    <div class="breadcrumb">{crumbs}</div>
  </div>
  <button class="btn-pdf" onclick="window.print()">⬇ Download PDF</button>
</div>
<div class="container">
{body}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def _accuracy_color(acc: float) -> str:
    return "hi" if acc >= 0.70 else ("mid" if acc >= 0.45 else "lo")


def _acc_bar(label: str, acc: float, n: int, correct: int) -> str:
    pct = acc * 100
    return (
        f'<div class="acc-row">'
        f'<span class="acc-label" title="{label}">{label}</span>'
        f'<div class="acc-track">'
        f'<div class="acc-fill {_accuracy_color(acc)}" style="width:{pct:.1f}%"></div>'
        f'</div>'
        f'<span class="acc-value">{pct:.1f}% ({correct}/{n})</span>'
        f'</div>'
    )


def _svg_hbar(
    items: List[tuple],
    width: int = 520,
    bar_height: int = 22,
    label_width: int = 140,
    gap: int = 6,
    color: str = "#2563eb",
) -> str:
    if not items:
        return ""
    row_h = bar_height + gap
    height = row_h * len(items) + 30
    bar_area = width - label_width - 60

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'role="img" aria-label="Accuracy bar chart">'
    ]
    for i, (label, acc, n, correct) in enumerate(items):
        y = 20 + i * row_h
        bar_w = max(1, int(acc * bar_area))
        pct_str = f"{acc:.1%} ({correct}/{n})"
        safe_label = label[:22] + "…" if len(label) > 22 else label
        col = "#16a34a" if acc >= 0.70 else ("#d97706" if acc >= 0.45 else "#dc2626")
        lines += [
            f'  <text x="0" y="{y + bar_height - 5}" font-size="11" fill="#374151" '
            f'font-weight="600">{safe_label}</text>',
            f'  <rect x="{label_width}" y="{y}" width="{bar_area}" height="{bar_height}" '
            f'rx="4" fill="#e5e7eb"/>',
            f'  <rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" '
            f'rx="4" fill="{col}"/>',
            f'  <text x="{label_width + bar_area + 6}" y="{y + bar_height - 5}" '
            f'font-size="10" fill="#374151">{pct_str}</text>',
        ]
    lines.append("</svg>")
    return f'<div class="svg-chart-wrap">{"".join(lines)}</div>'


def _render_metadata_breakdown(title: str, breakdown: dict, width: int = 520) -> str:
    if not breakdown:
        return ""
    items = [
        (k, v.get("accuracy", 0.0), v.get("n", 0), v.get("correct", 0))
        for k, v in sorted(breakdown.items())
    ]
    svg = _svg_hbar(items, width=width)
    return (
        f'<details style="margin:8px 0">'
        f'<summary style="cursor:pointer;font-size:0.88em;font-weight:700;color:#374151;'
        f'padding:4px 0">{title}</summary>'
        f'<div style="margin-top:8px">{svg}</div>'
        f'</details>'
    )


def _render_binary_metrics(bm: dict) -> str:
    if not bm or not bm.get("n"):
        return ""
    return (
        f'<div class="binary-grid">'
        f'<div class="binary-cell"><div class="val">{bm.get("precision", 0.0):.2%}</div>'
        f'<div class="lbl">Precision</div></div>'
        f'<div class="binary-cell"><div class="val">{bm.get("recall", 0.0):.2%}</div>'
        f'<div class="lbl">Recall</div></div>'
        f'<div class="binary-cell"><div class="val">{bm.get("f1", 0.0):.2%}</div>'
        f'<div class="lbl">F1</div></div>'
        f'<div class="binary-cell"><div class="val">{bm.get("n", 0)}</div>'
        f'<div class="lbl">Binary Qs</div></div>'
        f'</div>'
    )


def _render_reasoning_score(rs: dict) -> str:
    if not rs or not rs.get("n"):
        return ""
    composite = rs.get("composite_mean", 0.0)
    n         = rs.get("n", 0)
    per_dim   = rs.get("per_dimension", {})

    color = "#16a34a" if composite >= 1.5 else ("#d97706" if composite >= 1.0 else "#dc2626")
    cells = (
        f'<div class="judge-dim judge-composite">'
        f'<div class="dim-val" style="color:{color}">{composite:.2f}</div>'
        f'<div class="dim-name">Composite (0–2)</div>'
        f'</div>'
    )
    for dim_name, info in sorted(per_dim.items()):
        mean = info.get("mean", 0.0)
        dc   = "#16a34a" if mean >= 1.5 else ("#d97706" if mean >= 1.0 else "#dc2626")
        cells += (
            f'<div class="judge-dim">'
            f'<div class="dim-val" style="color:{dc}">{mean:.2f}</div>'
            f'<div class="dim-name">{dim_name.replace("_", " ").title()}</div>'
            f'</div>'
        )
    return (
        f'<div style="margin-top:14px">'
        f'<div class="chart-section-title">LLM Judge — Reasoning Score'
        f' <span style="font-weight:400;color:#6b7280">(n={n} questions scored)</span></div>'
        f'<div class="judge-score-grid">{cells}</div>'
        f'</div>'
    )


def _render_legend() -> str:
    """Collapsible color/badge legend shown at the top of every report page."""
    return """<details class="legend-panel">
<summary>Legend — color and badge guide</summary>
<div class="legend-body">
  <div>
    <div class="legend-section-title">Accuracy</div>
    <div class="legend-row"><span class="legend-swatch" style="background:#16a34a"></span>≥ 70% — high</div>
    <div class="legend-row"><span class="legend-swatch" style="background:#d97706"></span>45–70% — mid</div>
    <div class="legend-row"><span class="legend-swatch" style="background:#dc2626"></span>&lt; 45% — low</div>
  </div>
  <div>
    <div class="legend-section-title">QA Type</div>
    <div class="legend-row"><span class="badge badge-ladder">ladder</span> causal-chain questions (Rungs 1–3)</div>
    <div class="legend-row"><span class="badge badge-dormant">dormant</span> dormant-edge questions</div>
    <div class="legend-row"><span class="badge badge-distractor">distractor</span> irrelevant-object questions</div>
  </div>
  <div>
    <div class="legend-section-title">Causal Rung (Pearl)</div>
    <div class="legend-row"><span class="badge badge-rung-0">Rung 0</span> observational (what is?)</div>
    <div class="legend-row"><span class="badge badge-rung-1">Rung 1</span> interventional (what if?)</div>
    <div class="legend-row"><span class="badge badge-rung-2">Rung 2</span> counterfactual — effect</div>
    <div class="legend-row"><span class="badge badge-rung-3">Rung 3</span> counterfactual — cause</div>
  </div>
  <div>
    <div class="legend-section-title">Trajectory Grade (ADE)</div>
    <div class="legend-row"><span class="traj-bte-card traj-grade-excellent" style="padding:1px 7px;font-size:0.78em">EXCELLENT</span> ADE &lt; 0.5 m</div>
    <div class="legend-row"><span class="traj-bte-card traj-grade-good" style="padding:1px 7px;font-size:0.78em">GOOD</span> ADE &lt; 1.0 m</div>
    <div class="legend-row"><span class="traj-bte-card traj-grade-fair" style="padding:1px 7px;font-size:0.78em">FAIR</span> ADE &lt; 2.0 m</div>
    <div class="legend-row"><span class="traj-bte-card traj-grade-poor" style="padding:1px 7px;font-size:0.78em">POOR</span> ADE ≥ 2.0 m</div>
  </div>
  <div>
    <div class="legend-section-title">CTE Flags</div>
    <div class="legend-row"><span class="traj-flag traj-flag-full">FULL_MATCH</span> correct VQA + distinct + ADE &lt; 0.75 m</div>
    <div class="legend-row"><span class="traj-flag traj-flag-good">GOOD_COUNTERFACTUAL</span> distinct trajectory + ADE &lt; 1.5 m</div>
    <div class="legend-row"><span class="traj-flag traj-flag-same">TRAJECTORY_SAME_AS_BTE</span> predicted same path as baseline</div>
    <div class="legend-row"><span class="traj-flag traj-flag-mismatch">ANSWER_MISMATCH</span> CTE VQA answer differs from original</div>
  </div>
</div>
</details>"""


def _render_metrics_section(metrics: dict) -> str:
    overall  = metrics.get("overall", {})
    per_type = metrics.get("per_qa_type", {})
    if not overall and not per_type:
        return "<p>No metrics computed.</p>"

    acc     = overall.get("accuracy", 0.0)
    n       = overall.get("n", 0)
    correct = overall.get("correct", 0)

    rs = metrics.get("reasoning_score", {})
    judge_card = ""
    if rs and rs.get("n"):
        composite = rs.get("composite_mean", 0.0)
        jc = "#16a34a" if composite >= 1.5 else ("#d97706" if composite >= 1.0 else "#dc2626")
        judge_card = (
            f'<div class="metric-card">'
            f'<div class="big" style="color:{jc}">{composite:.2f}</div>'
            f'<div class="sub">Judge Score (0–2)</div>'
            f'</div>'
        )

    cards = f"""<div class="metric-grid">
  <div class="metric-card"><div class="big">{acc:.1%}</div><div class="sub">Overall Accuracy</div></div>
  <div class="metric-card"><div class="big">{n}</div><div class="sub">Total Questions</div></div>
  <div class="metric-card"><div class="big">{correct}</div><div class="sub">Correct</div></div>
  <div class="metric-card"><div class="big">{n - correct}</div><div class="sub">Wrong</div></div>
  {judge_card}
</div>"""

    qa_type_items = []
    if per_type:
        qa_type_items = [
            (qt, v.get("accuracy", 0.0), v.get("n", 0), v.get("correct", 0))
            for qt, v in sorted(per_type.items())
        ]
    all_items = [("Overall", acc, n, correct)] + qa_type_items
    qa_type_svg = _svg_hbar(all_items)

    meta_html = ""
    meta_html += _render_metadata_breakdown("By Question Type",  metrics.get("per_question_type", {}))
    meta_html += _render_metadata_breakdown("By Graph Structure", metrics.get("per_graph_structure", {}))
    meta_html += _render_metadata_breakdown("By Causal Rung",    metrics.get("per_rung", {}))
    meta_html += _render_metadata_breakdown("By Causal Status",  metrics.get("per_causal_status", {}))

    binary_html = _render_binary_metrics(metrics.get("binary_metrics", {}))

    ans_stats = metrics.get("answer_stats", {})
    bias_val  = ans_stats.get("answer_bias", None)
    bias_html = ""
    if bias_val is not None:
        bias_color = "#dc2626" if bias_val > 15 else ("#d97706" if bias_val > 5 else "#16a34a")
        bias_html = (
            f'<p style="font-size:0.83em;margin-top:8px">'
            f'Answer bias: <strong style="color:{bias_color}">{bias_val:+.1f}%</strong>'
            f'<span style="color:#6b7280"> (max predicted % − uniform expected %)</span></p>'
        )

    inner_content = (
        f'<div class="chart-section">'
        f'<div class="chart-section-title">Accuracy by QA Type</div>'
        f'{qa_type_svg}</div>'
        + (f'<div class="meta-breakdown">{meta_html}</div>' if meta_html.strip() else "")
        + (f'<div style="margin-top:14px"><div class="chart-section-title">Binary Question Metrics (Yes/No)</div>{binary_html}</div>' if binary_html else "")
        + bias_html
        + _render_reasoning_score(rs)
    )

    return cards + f'<div class="card">{inner_content}</div>'


def _badge(text: str, css_class: str) -> str:
    return f'<span class="badge {css_class}">{text}</span>'


def _qa_type_badge(qa_type: str) -> str:
    cls = f"badge-{qa_type}" if qa_type in ("ladder", "dormant", "distractor") else "badge-unknown"
    return _badge(qa_type, cls)


def _rung_badge(rung: str) -> str:
    css = f"badge-rung-{rung}" if rung in ("0", "1", "2", "3") else "badge-unknown"
    label = f"Rung {rung}" if rung else "—"
    return _badge(label, css)


def _tag(text: str) -> str:
    return f'<span class="tag-chip">{text}</span>'


# ---------------------------------------------------------------------------
# QA Accordion
# ---------------------------------------------------------------------------

def _render_qa_accordion(qa_results: list) -> str:
    """
    Render questions as a collapsible accordion list.

    Each item's summary line shows: Q-ID · badges · truncated question text ·
    Pred → GT · result.  Expanding reveals all fields in a vertical key-value
    table, which is much easier to read and print than a wide table.
    """
    if not qa_results:
        return "<p><em>No QA results recorded.</em></p>"

    items_html = ""
    for r in qa_results:
        correct = r.get("correct", False)
        cls     = "correct" if correct else "wrong"
        result_badge = (
            _badge("✓ correct", "badge-correct") if correct
            else _badge("✗ wrong", "badge-wrong")
        )

        q_id      = r.get("question_id", "")
        qa_type   = r.get("qa_type", "unknown")
        rung      = str(r.get("rung", "")) if r.get("rung") is not None else ""
        q_type    = r.get("question_type", "")
        graph     = r.get("graph_structure", "") or ""
        q_text    = r.get("question_text", "") or ""
        q_short   = (q_text[:90] + "…") if len(q_text) > 90 else q_text
        pred      = r.get("predicted", "") or "—"
        gt        = r.get("ground_truth", "") or "—"

        # ── summary line ──────────────────────────────────────────────
        tags = (
            _qa_type_badge(qa_type)
            + (" " + _rung_badge(rung) if rung else "")
            + (" " + _tag(q_type)      if q_type else "")
            + (" " + _tag(graph)       if graph  else "")
        )
        summary = (
            f'<summary class="qa-summary">'
            f'<code class="qa-summary-qid">{q_id}</code>'
            f' {tags}'
            f' <span class="qa-summary-q">{q_short}</span>'
            f' <span class="qa-summary-answer">Pred: {pred} → GT: {gt}</span>'
            f' {result_badge}'
            f' <span class="qa-chevron">▶</span>'
            f'</summary>'
        )

        # ── expanded key-value rows ────────────────────────────────────
        def kv(key: str, val: str) -> str:
            return f'<tr><td class="qa-kv-key">{key}</td><td class="qa-kv-val">{val}</td></tr>'

        rows = kv("Question", q_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

        # MCQ options
        opts = r.get("options") or []
        if opts:
            gt_u   = gt.strip().upper()
            pred_u = pred.strip().upper()
            opts_html = '<ul class="qa-options">'
            for opt in opts:
                opt_str = str(opt)
                letter  = opt_str[0].upper() if opt_str else ""
                css = "qa-option"
                if letter == gt_u:
                    css += " qa-option-correct"
                elif letter == pred_u and not correct:
                    css += " qa-option-predicted-wrong"
                esc = opt_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                opts_html += f'<li class="{css}">{esc}</li>'
            opts_html += "</ul>"
            rows += kv("Options", opts_html)

        rows += kv("Predicted", f"<strong>{pred}</strong>")
        rows += kv("Ground Truth", gt)
        rows += kv("Result", result_badge)

        # Causal status
        causal_status = r.get("causal_status", "") or ""
        if causal_status:
            rows += kv("Causal Status", _tag(causal_status))

        # Inference time
        t_s = r.get("inference_time_s")
        if t_s is not None:
            rows += kv("Inference Time", f"{t_s:.2f} s")

        # Model output
        raw = r.get("raw_output_text", "") or ""
        if raw:
            esc = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            rows += kv("Model Output", f"<div class='output-text'>{esc}</div>")

        # Expected reasoning
        reasoning = r.get("reasoning") or ""
        if reasoning:
            esc_r = str(reasoning).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            rows += kv("Expected Reasoning", f"<div class='reasoning-text'>{esc_r}</div>")

        # LLM judge score
        js = r.get("llm_judge_score")
        if js and isinstance(js, dict):
            if "error" in js:
                rows += kv("Judge Score", '<em style="color:#dc2626">Error</em>')
            elif "scores" in js:
                composite = js.get("composite_score", 0.0)
                jc  = "#16a34a" if composite >= 1.5 else ("#d97706" if composite >= 1.0 else "#dc2626")
                dim_cells = "".join(
                    f'<div class="judge-dim">'
                    f'<div class="dim-val">{v}</div>'
                    f'<div class="dim-name">{d.replace("_", " ").title()}</div>'
                    f'</div>'
                    for d, v in js["scores"].items()
                )
                judge_reasoning = js.get("reasoning", "")
                jr_html = ""
                if judge_reasoning:
                    esc_jr = str(judge_reasoning).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    jr_html = f'<div class="reasoning-text" style="margin-top:6px">{esc_jr}</div>'
                judge_val = (
                    f'<div class="judge-composite" style="color:{jc};font-weight:800;margin-bottom:4px">'
                    f'{composite:.2f} / 2</div>'
                    f'<div class="judge-score-grid" style="margin-top:0">{dim_cells}</div>'
                    f'{jr_html}'
                )
                rows += kv("Judge Score", judge_val)

        detail = f'<div class="qa-detail"><table class="qa-kv-table"><tbody>{rows}</tbody></table></div>'

        items_html += f'<details class="qa-item {cls}">{summary}{detail}</details>\n'

    return f'<div class="qa-accordion">{items_html}</div>'


# ---------------------------------------------------------------------------
# Trajectory evaluation rendering
# ---------------------------------------------------------------------------

_GRADE_CSS = {
    "EXCELLENT": "traj-grade-excellent",
    "GOOD":      "traj-grade-good",
    "FAIR":      "traj-grade-fair",
    "POOR":      "traj-grade-poor",
}

_FLAG_CSS = {
    "TRAJECTORY_SAME_AS_BTE": "traj-flag-same",
    "ANSWER_MISMATCH":        "traj-flag-mismatch",
    "GOOD_COUNTERFACTUAL":    "traj-flag-good",
    "FULL_MATCH":             "traj-flag-full",
}


def _traj_grade_card(eval_type: str, ade: Optional[float], grade: Optional[str],
                     n_gt: int, n_pred: int, t_range: Optional[list]) -> str:
    if ade is None:
        css = "traj-grade-na"
        ade_str = "N/A"
        grade_str = "—"
    else:
        css = _GRADE_CSS.get(grade or "", "traj-grade-na")
        ade_str = f"{ade:.3f} m"
        grade_str = grade or "—"

    t_str = (f"t={t_range[0]}–{t_range[-1]}s" if t_range and len(t_range) >= 2 else "")
    sub = f"{n_gt} GT pts · {n_pred} pred pts" + (f" · {t_str}" if t_str else "")

    return (
        f'<div class="traj-bte-card {css}">'
        f'<div class="traj-type-label">{eval_type}</div>'
        f'<div class="traj-ade-val">{ade_str}</div>'
        f'<div class="traj-grade-label">{grade_str}</div>'
        f'<div class="traj-sub">{sub}</div>'
        f'</div>'
    )


def _flag_badges(flags: list) -> str:
    if not flags:
        return '<span style="color:#9ca3af;font-size:0.8em">no flags</span>'
    return " ".join(
        f'<span class="traj-flag {_FLAG_CSS.get(f, "")}">{f.replace("_", " ")}</span>'
        for f in flags
    )


def _render_trajectory_section(traj_report: dict) -> str:
    """Render BTE + CTE trajectory results for a single sample."""
    if not traj_report:
        return ""

    bte = traj_report.get("bte") or {}
    cte_items = traj_report.get("cte") or []

    # ── BTE card ──────────────────────────────────────────────────────
    if bte.get("parse_success"):
        analysis = bte.get("analysis", {})
        bte_card = _traj_grade_card(
            "BTE — Baseline",
            analysis.get("ade"),
            analysis.get("grade"),
            bte.get("n_waypoints_gt", 0),
            bte.get("n_waypoints_pred", 0),
            bte.get("gt_t_range"),
        )
        interp = _bte_interpretation(analysis.get("grade", ""))
    else:
        bte_card = (
            f'<div class="traj-bte-card traj-grade-na">'
            f'<div class="traj-type-label">BTE — Baseline</div>'
            f'<div class="traj-parse-fail">Could not parse waypoints</div>'
            f'</div>'
        )
        interp = "The model did not output valid waypoints in JSON format."

    # ── CTE items ─────────────────────────────────────────────────────
    cte_html = ""
    for c in cte_items:
        eval_type   = c.get("eval_type", "CTE")
        parent_q    = c.get("parent_question_id", "")
        analysis    = c.get("analysis", {})
        flags       = analysis.get("flags", [])
        grade       = analysis.get("grade", "")
        ade         = c.get("ade")
        ans_match   = analysis.get("answer_match")
        ans_cte     = c.get("vqa_answer_cte", "—")
        ans_orig    = c.get("vqa_answer_original", "—")
        n_gt        = c.get("n_waypoints_gt", 0)
        n_pred      = c.get("n_waypoints_pred", 0)
        t_range     = c.get("gt_t_range")

        if c.get("parse_success"):
            grade_css = _GRADE_CSS.get(grade, "")
            grade_badge = f'<span class="traj-bte-card {grade_css}" style="padding:1px 6px;font-size:0.78em">{grade}</span>'
            ade_str = f'{ade:.3f} m' if ade is not None else "—"
            t_str = (f"t={t_range[0]}–{t_range[-1]}s" if t_range and len(t_range) >= 2 else "")
            cte_body = (
                f'<div class="traj-cte-header">'
                f'<span class="traj-cte-type">{eval_type}</span>'
                f'<span class="traj-cte-ade">ADE: {ade_str}</span>'
                f' {grade_badge}'
                f'</div>'
                f'<div style="font-size:0.8em;color:#6b7280;margin-bottom:4px">'
                f'Parent: {parent_q} &nbsp;|&nbsp; {n_gt} GT pts · {n_pred} pred pts'
                + (f' · {t_str}' if t_str else '')
                + f'</div>'
                f'<div style="font-size:0.8em;margin-bottom:4px">'
                f'VQA answer: <strong>{ans_cte}</strong> (original: {ans_orig})'
                + (' &nbsp;<span style="color:#16a34a">✓ match</span>' if ans_match else
                   ' &nbsp;<span style="color:#dc2626">✗ mismatch</span>')
                + f'</div>'
                f'<div>Flags: {_flag_badges(flags)}</div>'
                f'<div style="margin-top:5px;font-size:0.78em;color:#6b7280">{_cte_interpretation(grade, flags)}</div>'
            )
        else:
            cte_body = (
                f'<div class="traj-cte-header"><span class="traj-cte-type">{eval_type}</span>'
                f'<span class="traj-parse-fail">Could not parse waypoints</span></div>'
                f'<div style="font-size:0.8em;color:#6b7280">Parent: {parent_q}</div>'
            )

        cte_html += f'<div class="traj-cte-item">{cte_body}</div>'

    cte_section = (
        f'<div class="traj-cte-list">{cte_html}</div>'
        if cte_html
        else '<p class="traj-parse-fail">No CTE questions for this sample.</p>'
    )

    return (
        f'<div class="card traj-section">'
        f'<div class="card-title">Trajectory Evaluation</div>'
        f'<div class="traj-layout">{bte_card}{cte_section}</div>'
        f'<div class="traj-interp">{interp}</div>'
        f'</div>'
    )


def _bte_interpretation(grade: str) -> str:
    return {
        "EXCELLENT": "BTE EXCELLENT (ADE &lt; 0.5 m): predicted trajectory closely matches ground truth ego path.",
        "GOOD":      "BTE GOOD (ADE &lt; 1.0 m): trajectory is reasonable but has moderate lateral or longitudinal error.",
        "FAIR":      "BTE FAIR (ADE &lt; 2.0 m): trajectory is recognizable but shows significant deviation from ground truth.",
        "POOR":      "BTE POOR (ADE ≥ 2.0 m): predicted trajectory diverges substantially from ground truth.",
    }.get(grade, "BTE result unavailable.")


def _cte_interpretation(grade: str, flags: list) -> str:
    if "FULL_MATCH" in flags:
        return "Full match: VQA answer consistent, trajectory clearly distinct and highly accurate."
    if "GOOD_COUNTERFACTUAL" in flags:
        return "Good counterfactual: trajectory is distinct from BTE and has low ADE."
    if "ANSWER_MISMATCH" in flags:
        return "VQA answer changed under counterfactual — may indicate causal awareness or instability."
    if "TRAJECTORY_SAME_AS_BTE" in flags:
        return "Trajectory nearly identical to BTE — model did not modify its path for the counterfactual."
    return {
        "INCONCLUSIVE": "Counterfactual trajectory distinct but ADE does not meet good/excellent threshold.",
        "FLAGGED":      "One or more consistency flags raised — inspect flags above.",
        "EXCELLENT":    "CTE EXCELLENT: highly accurate counterfactual with low ADE.",
        "GOOD":         "CTE GOOD: reasonable counterfactual trajectory.",
    }.get(grade, "")


def _render_trajectory_aggregate(traj_agg: dict) -> str:
    """Render trajectory aggregate stats for dataset or run-level pages."""
    if not traj_agg:
        return ""

    n_samples     = traj_agg.get("n_samples", 0)
    bte_stats     = traj_agg.get("bte_ade_stats", {})
    bte_grades    = traj_agg.get("bte_grade_counts", {})
    cte_ade_stats = traj_agg.get("cte_ade_stats", {})
    cte_grades    = traj_agg.get("cte_grade_counts", {})
    cte_flags     = traj_agg.get("cte_flag_counts", {})

    # BTE summary cards
    bte_cards = ""
    if bte_stats.get("n"):
        bte_mean = bte_stats.get("mean", 0.0)
        mc  = "#16a34a" if bte_mean < 1.0 else ("#d97706" if bte_mean < 2.0 else "#dc2626")
        bte_cards += (
            f'<div class="traj-agg-card">'
            f'<div class="traj-agg-val" style="color:{mc}">{bte_mean:.2f} m</div>'
            f'<div class="traj-agg-lbl">BTE Mean ADE (n={bte_stats["n"]})</div>'
            f'</div>'
        )
        bte_cards += (
            f'<div class="traj-agg-card">'
            f'<div class="traj-agg-val">{bte_stats.get("min", 0.0):.2f} – {bte_stats.get("max", 0.0):.2f} m</div>'
            f'<div class="traj-agg-lbl">BTE ADE range</div>'
            f'</div>'
        )
    for grade, cnt in sorted(bte_grades.items()):
        gcss = _GRADE_CSS.get(grade, "traj-grade-na")
        bte_cards += (
            f'<div class="traj-agg-card">'
            f'<div class="traj-agg-val">{cnt}</div>'
            f'<div class="traj-agg-lbl traj-bte-card {gcss}" style="padding:1px 5px;font-size:0.72em;display:inline-block;margin-top:3px">{grade}</div>'
            f'</div>'
        )

    # CTE summary
    cte_cards = ""
    for cte_type, stats in sorted(cte_ade_stats.items()):
        n_c = stats.get("n", 0)
        mean_c = stats.get("mean", 0.0)
        mc = "#16a34a" if mean_c < 1.0 else ("#d97706" if mean_c < 2.0 else "#dc2626")
        cte_cards += (
            f'<div class="traj-agg-card">'
            f'<div class="traj-agg-val" style="color:{mc}">{mean_c:.2f} m</div>'
            f'<div class="traj-agg-lbl">{cte_type} mean ADE (n={n_c})</div>'
            f'</div>'
        )

    # CTE flag counts
    flag_html = ""
    for flag, cnt in sorted(cte_flags.items()):
        fcss = _FLAG_CSS.get(flag, "")
        flag_html += (
            f'<span class="traj-flag {fcss}" style="margin:2px 4px;padding:2px 8px">'
            f'{flag.replace("_", " ")}: {cnt}</span>'
        )

    sections = ""
    if bte_cards:
        sections += (
            f'<div class="chart-section-title" style="margin-top:10px">BTE — Baseline Trajectory</div>'
            f'<div class="traj-aggregate-grid">{bte_cards}</div>'
        )
    if cte_cards:
        sections += (
            f'<div class="chart-section-title" style="margin-top:14px">CTE — Counterfactual Trajectory</div>'
            f'<div class="traj-aggregate-grid">{cte_cards}</div>'
        )
    if flag_html:
        sections += (
            f'<div class="chart-section-title" style="margin-top:14px">CTE Flags</div>'
            f'<div style="margin-top:6px">{flag_html}</div>'
        )

    if not sections:
        return ""

    return (
        f'<div class="card">'
        f'<div class="card-title">Trajectory Evaluation Summary'
        f' <span style="font-weight:400;font-size:0.85em;color:#6b7280">'
        f'({n_samples} samples with trajectory outputs)</span></div>'
        f'{sections}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Graph structure rendering
# ---------------------------------------------------------------------------

def _render_graph_structure(graph_struct: dict) -> str:
    """Render the topology summary from graph_structure.json."""
    if not graph_struct:
        return ""

    topo    = graph_struct.get("topology", {})
    sg_all  = graph_struct.get("subgraphs_all", {}).get("counts", {})
    sg_act  = graph_struct.get("subgraphs_active", {}).get("counts", {})

    chips = ""
    for key, label in [
        ("n_nodes", "nodes"), ("n_distractors", "distractors"),
        ("n_active_edges", "active edges"), ("n_dormant_edges", "dormant edges"),
        ("graph_depth", "depth"), ("richness", "richness"),
    ]:
        val = topo.get(key)
        if val is not None:
            chips += f'<span class="graph-topo-chip"><span>{val}</span> {label}</span>'

    sg_html = ""
    if sg_all or sg_act:
        for sg_type in ["direct", "chain", "collider", "confounding_observed", "confounding_unobserved"]:
            all_c = sg_all.get(sg_type, 0)
            act_c = sg_act.get(sg_type, 0)
            label = sg_type.replace("_", " ").title()
            cls   = "graph-subgraph-chip" + (" count-0" if all_c == 0 else "")
            sg_html += f'<span class="{cls}">{label}: {act_c}/{all_c}</span>'

    return (
        f'<details style="margin-bottom:16px">'
        f'<summary style="cursor:pointer;font-size:0.88em;font-weight:700;color:#374151;padding:4px 0">'
        f'Graph Structure</summary>'
        f'<div style="margin-top:8px">'
        f'<div class="graph-topo-grid">{chips}</div>'
        + (f'<div style="margin-top:8px;font-size:0.8em;color:#6b7280">Subgraph types (active/total): </div>'
           f'<div class="graph-subgraph-grid">{sg_html}</div>' if sg_html else "")
        + f'</div></details>'
    )


# ---------------------------------------------------------------------------
# Tier 1 — sample HTML
# ---------------------------------------------------------------------------

def write_sample_html(
    sample_report: dict,
    html_path: Path,
    bench_dir: Optional[Path] = None,
) -> None:
    """Write Tier 1 HTML for a single sample (SAMPLED_N)."""
    run_name  = sample_report.get("run_name", "")
    dataset   = sample_report.get("dataset", "")
    scene_id  = sample_report.get("scene_id", "")
    sample_id = sample_report.get("sample_id", "")
    generated = sample_report.get("generated_at", "")[:19].replace("T", " ")

    breadcrumbs = [
        ("All Runs", "../../../../index.html"),
        (run_name,   "../../../index.html"),
        (dataset,    "../../index.html"),
        (scene_id,   "../index.html"),
        (sample_id,  None),
    ]

    # Load graph_structure.json from bench_dir if available
    graph_html = ""
    if bench_dir:
        gs_path = bench_dir / scene_id / sample_id / "graph_structure.json"
        if gs_path.exists():
            try:
                graph_struct = json.loads(gs_path.read_text(encoding="utf-8"))
                graph_html   = _render_graph_structure(graph_struct)
            except Exception:
                pass

    # Load trajectory_report.json (lives beside report.json in cdb_outputs)
    traj_html = ""
    traj_path = html_path.parent.parent.parent.parent.parent / html_path.parts[-5] / html_path.parts[-4] / html_path.parts[-3] / html_path.parts[-2] / "trajectory_report.json"
    # Simpler: derive from sample report path structure via run_dir
    # The html_path is: cdb_reports/<run>/<dataset>/<scene>/<sample>/index.html
    # trajectory_report.json is: cdb_outputs/<run>/<dataset>/<scene>/<sample>/trajectory_report.json
    # We pass it separately, so look it up relative to the outputs dir stored in the report
    traj_report_path = _find_trajectory_report(html_path, sample_report)
    if traj_report_path and traj_report_path.exists():
        try:
            traj_report = json.loads(traj_report_path.read_text(encoding="utf-8"))
            traj_html   = _render_trajectory_section(traj_report)
        except Exception:
            pass

    n_questions = sample_report.get("n_questions", 0)
    body = f"""
{_render_legend()}
<div class="card">
  <div class="card-title">Sample: {scene_id} / {sample_id}</div>
  <p style="font-size:0.82em;color:#6b7280">Generated: {generated} &nbsp;|&nbsp;
     Run: {run_name} &nbsp;|&nbsp; Dataset: {dataset}</p>
  {graph_html}
</div>
{_render_metrics_section(sample_report.get("metrics", {}))}
{traj_html}
<div class="card">
  <div class="card-title">Question Results ({n_questions} questions)</div>
  {_render_qa_accordion(sample_report.get("qa_results", []))}
</div>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_page(f"{scene_id}/{sample_id}", breadcrumbs, body), encoding="utf-8")
    logger.debug("Sample HTML: %s", html_path)


def _find_trajectory_report(html_path: Path, sample_report: dict) -> Optional[Path]:
    """
    Derive the trajectory_report.json path from the sample HTML output path.
    html_path: cdb_reports/<run>/<dataset>/<scene>/<sample>/index.html
    target:    cdb_outputs/<run>/<dataset>/<scene>/<sample>/trajectory_report.json
    """
    # Walk up: index.html → sample/ → scene/ → dataset/ → run/ → cdb_reports/
    try:
        cdb_reports = html_path.parents[4]  # five levels up from index.html
        cdb_outputs = cdb_reports.parent / (cdb_reports.name.replace("reports", "outputs"))
        run_name    = html_path.parts[-5]
        dataset     = html_path.parts[-4]
        scene_id    = html_path.parts[-3]
        sample_id   = html_path.parts[-2]
        return cdb_outputs / run_name / dataset / scene_id / sample_id / "trajectory_report.json"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier 2.5 — scene HTML
# ---------------------------------------------------------------------------

def write_scene_html(
    scene_id: str,
    sample_reports: List[dict],
    html_path: Path,
    run_name: str,
    dataset_name: str,
) -> None:
    breadcrumbs = [
        ("All Runs",   "../../../index.html"),
        (run_name,     "../../index.html"),
        (dataset_name, "../index.html"),
        (scene_id,     None),
    ]

    all_qa: list = []
    for sr in sample_reports:
        all_qa.extend(sr.get("qa_results", []))
    scene_metrics = _aggregate_metrics(all_qa)

    rows = ""
    for sr in sorted(sample_reports, key=lambda s: s.get("sample_id", "")):
        sid     = sr.get("sample_id", "")
        ov      = sr.get("metrics", {}).get("overall", {})
        acc     = ov.get("accuracy", 0.0)
        pct     = acc * 100
        color   = _accuracy_color(acc)
        n_q     = sr.get("n_questions", 0)
        correct = ov.get("correct", 0)
        rows += f"""<tr>
  <td><a href="{sid}/index.html">{sid}</a></td>
  <td>{n_q}</td>
  <td>{correct}</td>
  <td>
    <div class="acc-track" style="max-width:180px;display:inline-block;vertical-align:middle">
      <div class="acc-fill {color}" style="width:{pct:.1f}%"></div>
    </div>
    &nbsp;<strong>{pct:.1f}%</strong>
  </td>
  <td><a href="{sid}/index.html">View →</a></td>
</tr>"""

    sample_table = (
        f"""<table>
<thead><tr><th>Sample</th><th>Questions</th><th>Correct</th>
<th>Accuracy</th><th>Details</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""
        if rows else "<p><em>No samples found.</em></p>"
    )

    # Print-only: accordion per sample (expands fully in print via CSS)
    print_sections = ""
    for sr in sorted(sample_reports, key=lambda s: s.get("sample_id", "")):
        sid  = sr.get("sample_id", "")
        ov   = sr.get("metrics", {}).get("overall", {})
        acc  = ov.get("accuracy", 0.0)
        n_q  = sr.get("n_questions", 0)
        corr = ov.get("correct", 0)
        print_sections += f"""
<div class="card" style="page-break-inside:avoid;margin-top:12px">
  <div class="card-title">Sample: {sid} &nbsp;—&nbsp; {acc:.1%} ({corr}/{n_q})</div>
  {_render_qa_accordion(sr.get("qa_results", []))}
</div>"""

    generated = (sample_reports[0].get("generated_at", "") if sample_reports else "")[:19].replace("T", " ")
    total_n   = scene_metrics.get("overall", {}).get("n", 0)

    body = f"""
{_render_legend()}
<div class="card">
  <div class="card-title">Scene: {scene_id}</div>
  <p style="font-size:0.82em;color:#6b7280">
    Run: {run_name} &nbsp;|&nbsp; Dataset: {dataset_name} &nbsp;|&nbsp;
    {len(sample_reports)} sample(s) &nbsp;|&nbsp; {total_n} questions &nbsp;|&nbsp;
    Generated: {generated}
  </p>
</div>
{_render_metrics_section(scene_metrics)}
<div class="card screen-only">
  <div class="card-title">Samples ({len(sample_reports)})</div>
  {sample_table}
</div>
<div class="print-only">
  <h2 style="margin:16px 0 8px;font-size:1em;font-weight:700;color:#374151">
    Full Q&amp;A — All Samples
  </h2>
  {print_sections}
</div>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_page(scene_id, breadcrumbs, body), encoding="utf-8")
    logger.debug("Scene HTML: %s", html_path)


# ---------------------------------------------------------------------------
# Tier 2 — dataset HTML
# ---------------------------------------------------------------------------

def write_dataset_html(
    dataset_report: dict,
    html_path: Path,
    traj_aggregate: Optional[dict] = None,
) -> None:
    """Write Tier 2 HTML: scene breakdown table + optional trajectory aggregate."""
    run_name  = dataset_report.get("run_name", "")
    dataset   = dataset_report.get("dataset", "")
    generated = dataset_report.get("generated_at", "")[:19].replace("T", " ")

    breadcrumbs = [
        ("All Runs", "../../index.html"),
        (run_name,   "../index.html"),
        (dataset,    None),
    ]

    summaries = dataset_report.get("sample_summaries", [])
    scenes: Dict[str, list] = defaultdict(list)
    for s in summaries:
        scenes[s.get("scene_id", "")].append(s)

    rows = ""
    for scene_id in sorted(scenes.keys()):
        sc   = scenes[scene_id]
        n_q  = sum(s.get("n_questions", 0) for s in sc)
        corr = sum(s.get("correct", 0) for s in sc)
        acc  = corr / n_q if n_q else 0.0
        pct  = acc * 100
        color = _accuracy_color(acc)
        rows += f"""<tr>
  <td><a href="{scene_id}/index.html">{scene_id}</a></td>
  <td>{len(sc)}</td>
  <td>{n_q}</td>
  <td>{corr}</td>
  <td>
    <div class="acc-track" style="max-width:180px;display:inline-block;vertical-align:middle">
      <div class="acc-fill {color}" style="width:{pct:.1f}%"></div>
    </div>
    &nbsp;<strong>{pct:.1f}%</strong>
  </td>
  <td><a href="{scene_id}/index.html">View →</a></td>
</tr>"""

    table_html = (
        f"""<table>
<thead><tr><th>Scene</th><th>Samples</th><th>Questions</th>
<th>Correct</th><th>Accuracy</th><th>Details</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""
        if rows else "<p><em>No scene reports found.</em></p>"
    )

    traj_html = _render_trajectory_aggregate(traj_aggregate or {})

    body = f"""
{_render_legend()}
<div class="card">
  <div class="card-title">Dataset: {dataset}</div>
  <p style="font-size:0.82em;color:#6b7280">Generated: {generated} &nbsp;|&nbsp;
     Run: {run_name} &nbsp;|&nbsp;
     {len(scenes)} scenes &nbsp;|&nbsp;
     {dataset_report.get("n_samples", 0)} samples</p>
</div>
{_render_metrics_section(dataset_report.get("metrics", {}))}
{traj_html}
<div class="card">
  <div class="card-title">Scene Breakdown ({len(scenes)} scenes)</div>
  {table_html}
</div>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_page(dataset, breadcrumbs, body), encoding="utf-8")
    logger.debug("Dataset HTML: %s", html_path)


# ---------------------------------------------------------------------------
# Tier 3 — run HTML
# ---------------------------------------------------------------------------

def write_run_html(
    run_report: dict,
    html_path: Path,
    traj_aggregates: Optional[Dict[str, dict]] = None,
) -> None:
    """Write Tier 3 HTML: dataset breakdown + trajectory aggregate per dataset."""
    run_name  = run_report.get("run_name", "")
    generated = run_report.get("generated_at", "")[:19].replace("T", " ")

    breadcrumbs = [
        ("All Runs", "../index.html"),
        (run_name,   None),
    ]

    rows = ""
    for s in run_report.get("dataset_summaries", []):
        ds   = s.get("dataset", "")
        acc  = s.get("accuracy", 0.0)
        pct  = acc * 100
        color = _accuracy_color(acc)
        rows += f"""<tr>
  <td><a href="{ds}/index.html">{ds}</a></td>
  <td>{s.get("n_samples", 0)}</td>
  <td>{s.get("n_questions", 0)}</td>
  <td>{s.get("correct", 0)}</td>
  <td>
    <div class="acc-track" style="max-width:180px;display:inline-block;vertical-align:middle">
      <div class="acc-fill {color}" style="width:{pct:.1f}%"></div>
    </div>
    &nbsp;<strong>{pct:.1f}%</strong>
  </td>
  <td><a href="{ds}/index.html">View →</a></td>
</tr>"""

    table_html = (
        f"""<table>
<thead><tr><th>Dataset</th><th>Samples</th><th>Questions</th>
<th>Correct</th><th>Accuracy</th><th>Details</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""
        if rows else "<p><em>No dataset reports found.</em></p>"
    )

    # Aggregate trajectory across all datasets
    combined_traj: dict = {}
    if traj_aggregates:
        for agg in traj_aggregates.values():
            _merge_traj_agg(combined_traj, agg)
    traj_html = _render_trajectory_aggregate(combined_traj)

    body = f"""
{_render_legend()}
<div class="card">
  <div class="card-title">Run: {run_name}</div>
  <p style="font-size:0.82em;color:#6b7280">Generated: {generated} &nbsp;|&nbsp;
     {run_report.get("n_questions", 0)} questions total</p>
</div>
{_render_metrics_section(run_report.get("metrics", {}))}
{traj_html}
<div class="card">
  <div class="card-title">Dataset Breakdown</div>
  {table_html}
</div>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_page(run_name, breadcrumbs, body), encoding="utf-8")
    logger.debug("Run HTML: %s", html_path)


def _merge_traj_agg(base: dict, extra: dict) -> None:
    """Merge extra trajectory aggregate into base in-place."""
    if not extra:
        return
    base["n_samples"] = base.get("n_samples", 0) + extra.get("n_samples", 0)

    for field in ("bte_ade_stats", "cte_grade_counts", "cte_flag_counts", "bte_grade_counts"):
        src = extra.get(field, {})
        dst = base.setdefault(field, {})
        for k, v in src.items():
            if isinstance(v, (int, float)):
                dst[k] = dst.get(k, 0) + v

    # cte_ade_stats requires weighted mean
    for cte_type, stats in extra.get("cte_ade_stats", {}).items():
        if cte_type not in base.setdefault("cte_ade_stats", {}):
            base["cte_ade_stats"][cte_type] = dict(stats)
        else:
            existing = base["cte_ade_stats"][cte_type]
            n1, n2 = existing.get("n", 0), stats.get("n", 0)
            if n1 + n2 > 0:
                existing["mean"] = (existing.get("mean", 0) * n1 + stats.get("mean", 0) * n2) / (n1 + n2)
            existing["n"] = n1 + n2
            existing["min"] = min(existing.get("min", float("inf")), stats.get("min", float("inf")))
            existing["max"] = max(existing.get("max", float("-inf")), stats.get("max", float("-inf")))

    # Also merge bte_ade_stats as a weighted mean
    if "bte_ade_stats" in extra and extra["bte_ade_stats"].get("n"):
        eb = extra["bte_ade_stats"]
        bb = base.setdefault("bte_ade_stats", {})
        n1, n2 = bb.get("n", 0), eb["n"]
        if n1 + n2 > 0:
            bb["mean"] = (bb.get("mean", 0) * n1 + eb.get("mean", 0) * n2) / (n1 + n2)
        bb["n"] = n1 + n2
        bb["min"] = min(bb.get("min", float("inf")), eb.get("min", float("inf")))
        bb["max"] = max(bb.get("max", float("-inf")), eb.get("max", float("-inf")))


# ---------------------------------------------------------------------------
# Tier 4 — root index.html
# ---------------------------------------------------------------------------

def _aggregate_metrics(qa_results: list) -> dict:
    if not qa_results:
        return {}
    n       = len(qa_results)
    correct = sum(1 for r in qa_results if r.get("correct"))
    by_type: Dict[str, list] = defaultdict(list)
    for r in qa_results:
        by_type[r.get("qa_type", "unknown")].append(r)
    per_type = {
        qt: {
            "n":        len(items),
            "correct":  sum(1 for r in items if r.get("correct")),
            "accuracy": sum(1 for r in items if r.get("correct")) / len(items),
        }
        for qt, items in by_type.items()
    }
    return {
        "overall":     {"accuracy": correct / n, "n": n, "correct": correct},
        "per_qa_type": per_type,
    }


def _render_model_comparison(per_model: dict, runs: dict) -> str:
    if not per_model:
        return ""

    svg_lines: List[str] = []
    bar_h, gap, label_w, bar_area, width = 24, 8, 160, 300, 560
    row_h = bar_h + gap
    chart_items = sorted(per_model.items())
    svg_h = row_h * len(chart_items) + 30
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}" '
        f'role="img" aria-label="Model comparison chart">'
    )
    for i, (mid, info) in enumerate(chart_items):
        acc = info.get("best_accuracy", 0.0)
        y = 20 + i * row_h
        bar_w = max(1, int(acc * bar_area))
        col = "#16a34a" if acc >= 0.70 else ("#d97706" if acc >= 0.45 else "#dc2626")
        safe_mid = mid[:24] + "…" if len(mid) > 24 else mid
        svg_lines += [
            f'  <text x="0" y="{y+bar_h-4}" font-size="12" fill="#374151" font-weight="600">{safe_mid}</text>',
            f'  <rect x="{label_w}" y="{y}" width="{bar_area}" height="{bar_h}" rx="4" fill="#e5e7eb"/>',
            f'  <rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{col}"/>',
            f'  <text x="{label_w+bar_area+6}" y="{y+bar_h-4}" font-size="11" fill="#374151">{acc:.1%}</text>',
        ]
    svg_lines.append("</svg>")
    chart_html = f'<div class="svg-chart-wrap">{"".join(svg_lines)}</div>'

    rows = ""
    for model_id, info in sorted(per_model.items()):
        rows += f"""<tr>
  <td><strong>{model_id}</strong></td>
  <td>{info.get("n_runs", 0)}</td>
  <td><strong>{info.get("best_accuracy", 0.0):.1%}</strong></td>
  <td>{info.get("avg_accuracy", 0.0):.1%}</td>
  <td style="font-size:0.82em">{", ".join(info.get("runs", []))}</td>
</tr>"""
    return f"""<div class="card model-section">
  <div class="card-title">Model Comparison</div>
  {chart_html}
  <table style="margin-top:14px">
  <thead><tr><th>Model</th><th>Runs</th><th>Best Accuracy</th>
  <th>Avg Accuracy</th><th>Run IDs</th></tr></thead>
  <tbody>{rows}</tbody>
  </table>
</div>"""


def generate_root_index(root_aggregate: dict, cdb_reports_dir: Path) -> None:
    """Write/rewrite ``cdb_reports/index.html`` from *root_aggregate*."""
    runs      = root_aggregate.get("runs", {})
    per_model = root_aggregate.get("per_model", {})
    generated = root_aggregate.get("generated_at", "")[:19].replace("T", " ")

    breadcrumbs = [("All Runs", None)]

    cards_html = ""
    for run_name, info in sorted(
        runs.items(), key=lambda kv: kv[1].get("generated_at", ""), reverse=True
    ):
        acc   = info.get("accuracy", 0.0)
        pct   = acc * 100
        color = _accuracy_color(acc)
        cards_html += f"""<div class="run-card">
  <h3><a href="{run_name}/index.html">{run_name}</a></h3>
  <div class="run-acc">{pct:.1f}%</div>
  <div class="acc-track" style="margin:8px 0">
    <div class="acc-fill {color}" style="width:{pct:.1f}%"></div>
  </div>
  <div class="run-meta">
    {info.get("n_questions", 0)} questions &nbsp;|&nbsp; {", ".join(info.get("datasets", []))}<br>
    {info.get("generated_at", "")[:10]}
  </div>
</div>"""

    body = f"""
{_render_legend()}
<div class="card">
  <div class="card-title">Benchmark Overview</div>
  <div class="metric-grid">
    <div class="metric-card"><div class="big">{len(runs)}</div><div class="sub">Total Runs</div></div>
    <div class="metric-card"><div class="big">{len(per_model)}</div><div class="sub">Models</div></div>
  </div>
  <p style="font-size:0.82em;color:#6b7280">Last updated: {generated}</p>
</div>
{_render_model_comparison(per_model, runs)}
<div class="card">
  <div class="card-title">All Evaluation Runs</div>
  <div class="run-grid">{cards_html}</div>
</div>"""

    html_path = cdb_reports_dir / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_page("All Runs", breadcrumbs, body), encoding="utf-8")
    logger.info("Root index: %s", html_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_html(
    run_name: str,
    cdb_outputs_dir: Path,
    cdb_reports_dir: Path,
    root_aggregate: dict,
    bench_dirs: Optional[Dict[str, Path]] = None,
) -> None:
    """
    Walk ``cdb_outputs/<run_name>/`` and generate HTML at all five tiers.

    Parameters
    ----------
    bench_dirs : optional mapping of dataset_name → bench_dir Path.
        When provided, graph_structure.json is loaded for each sample page.
    """
    run_dir = cdb_outputs_dir / run_name
    if not run_dir.exists():
        logger.warning("run_dir not found: %s", run_dir)
        return

    for dataset_dir in sorted(run_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        bench_dir    = (bench_dirs or {}).get(dataset_name)

        # Load trajectory aggregate for this dataset (if it exists)
        traj_agg_path = dataset_dir / "trajectory_aggregate.json"
        traj_agg: dict = {}
        if traj_agg_path.exists():
            try:
                traj_agg = json.loads(traj_agg_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Collect sample reports per scene as we generate Tier 1 pages
        scene_reports: Dict[str, List[dict]] = defaultdict(list)

        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name

            for sample_dir in sorted(scene_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sample_report = load_report(sample_dir / "report.json")
                if not sample_report or sample_report.get("level") != "sample":
                    continue

                # Tier 1: sample HTML
                sample_html = (
                    cdb_reports_dir / run_name / dataset_name
                    / scene_id / sample_dir.name / "index.html"
                )
                try:
                    write_sample_html(sample_report, sample_html, bench_dir=bench_dir)
                except Exception:
                    logger.exception("Failed sample HTML: %s", sample_html)

                scene_reports[scene_id].append(sample_report)

            # Tier 2.5: scene HTML
            if scene_reports.get(scene_id):
                scene_html = (
                    cdb_reports_dir / run_name / dataset_name / scene_id / "index.html"
                )
                try:
                    write_scene_html(
                        scene_id, scene_reports[scene_id],
                        scene_html, run_name, dataset_name,
                    )
                except Exception:
                    logger.exception("Failed scene HTML: %s", scene_html)

        # Tier 2: dataset HTML
        dataset_report = load_report(dataset_dir / "report.json")
        if dataset_report and dataset_report.get("level") == "dataset":
            dataset_html = cdb_reports_dir / run_name / dataset_name / "index.html"
            try:
                write_dataset_html(dataset_report, dataset_html, traj_aggregate=traj_agg)
            except Exception:
                logger.exception("Failed dataset HTML: %s", dataset_html)

    # Tier 3: run HTML — aggregate trajectory across all datasets
    run_report = load_report(run_dir / "report.json")
    if run_report and run_report.get("level") == "run":
        run_html = cdb_reports_dir / run_name / "index.html"
        # Collect all dataset-level trajectory aggregates
        traj_aggregates: Dict[str, dict] = {}
        for dataset_dir in run_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            tp = dataset_dir / "trajectory_aggregate.json"
            if tp.exists():
                try:
                    traj_aggregates[dataset_dir.name] = json.loads(tp.read_text(encoding="utf-8"))
                except Exception:
                    pass
        try:
            write_run_html(run_report, run_html, traj_aggregates=traj_aggregates)
        except Exception:
            logger.exception("Failed run HTML: %s", run_html)

    # Tier 4: root index
    try:
        generate_root_index(root_aggregate, cdb_reports_dir)
    except Exception:
        logger.exception("Failed root index HTML")
