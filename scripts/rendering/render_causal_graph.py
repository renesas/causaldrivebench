"""
Render clean causal graph PNGs from scene JSON files using Graphviz.
Supports both formats:
  - Hand-crafted (flat "edges" array)
  - VLM-generated ("ego_edges" + "inter_node_edges")

Also generates an HTML interactive visualization (D3.js force graph).

Usage:
    python render_causal_graph.py scene_0018.json scene_0093.json ...
    python render_causal_graph.py *.json
"""
import json
import os
import re
import sys
from pathlib import Path

_CSS_DIR = Path(__file__).parent / "css"

# Cache CSS files in memory — they're on local disk and never change during a run
_css_cache = {}

def _read_css(name):
    if name not in _css_cache:
        _css_cache[name] = (_CSS_DIR / name).read_text()
    return _css_cache[name]


def _normalize_edges(data):
    """
    Normalize both JSON formats into a unified edges list.
    Returns (ego_edges, inter_edges) where each is a list of dicts.
    """
    # New format: ego_edges + inter_node_edges
    if "ego_edges" in data:
        ego_edges = data.get("ego_edges", [])
        inter_edges = data.get("inter_node_edges", [])
        return ego_edges, inter_edges

    # Old format: flat "edges" array (all are ego edges, no inter-node)
    return data.get("edges", []), []


def _img_to_b64(img_path):
    """Read an image file and return base64 data-URI string."""
    import base64
    p = Path(img_path)
    if not p.exists():
        return ""
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, ext)
    with open(p, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"


def _rel_img(img_path, html_dir):
    """Return a relative URL from html_dir to img_path (no filesystem access)."""
    import os
    return os.path.relpath(str(img_path), str(html_dir))


def _load_d3_tag():
    """Return a <script> tag for D3.js using a relative path (served via HTTP)."""
    return '<script src="../d3.v7.min.js"></script>'


def _build_qa_section(qa_dir, html_dir, stem, scene_dir=None):
    """Build Q&A HTML section from qa results directory.

    Returns dict with keys: css, body_html, js, stats (counts for stats bar),
    and cf_image_urls (list of lists of {url, label}).

    All text fields are HTML-escaped to handle ?, ', \", <, >, & safely.
    """
    import html as _html

    def esc(text):
        """HTML-escape text for safe embedding."""
        return _html.escape(str(text)) if text else ""

    qa_dir = Path(qa_dir)
    if not qa_dir.exists():
        return {"body_html": "", "js": "", "stats": {}, "cf_image_urls": []}

    # ── Load data ──
    active_qs = []
    dl_qs = []
    cf_results = []

    def _extract_preamble(data, questions):
        """Extract preamble from top-level data or first question."""
        if isinstance(data, dict) and data.get("preamble"):
            return data["preamble"]
        for q in questions:
            if isinstance(q, dict) and q.get("preamble"):
                return q["preamble"]
        return ""

    lp = qa_dir / "active_qa.json"
    if lp.exists():
        ld = json.loads(lp.read_text())
        active_qs = ld.get("questions", ld) if isinstance(ld, dict) else ld
    active_preamble = _extract_preamble(ld if lp.exists() else {}, active_qs)

    dp = qa_dir / "dormant_qa.json"
    if dp.exists():
        dd = json.loads(dp.read_text())
        dl_qs = dd.get("questions", dd) if isinstance(dd, dict) else dd
    dl_preamble = _extract_preamble(dd if dp.exists() else {}, dl_qs)

    distractor_qs = []
    dist_p = qa_dir / "distractor_qa.json"
    if dist_p.exists():
        dist_d = json.loads(dist_p.read_text())
        distractor_qs = dist_d.get("questions", dist_d) if isinstance(dist_d, dict) else dist_d
    distractor_preamble = _extract_preamble(dist_d if dist_p.exists() else {}, distractor_qs)

    # Use first non-empty preamble as the scene-level preamble
    scene_preamble = active_preamble or dl_preamble or distractor_preamble

    cp = qa_dir / "rung3_cf_trajectories.json"
    if cp.exists():
        cf_results = json.loads(cp.read_text())

    ac_results = []
    ap = qa_dir / "ac_binary_trajectories.json"
    if ap.exists():
        ac_results = json.loads(ap.read_text())

    n_active = len(active_qs)
    n_dl = len(dl_qs)
    n_distractor = len(distractor_qs)
    n_cf = len(cf_results)
    n_ac = len(ac_results)

    # ── Helper badges ──
    def _rung_badge(rung):
        colors = {0: "#3498DB", 1: "#2ECC71", 2: "#F39C12", 3: "#E74C3C"}
        c = colors.get(rung, "#888")
        return f'<span style="background:{c}22;color:{c};border:1px solid {c}55;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700">Rung {rung}</span>'

    def _type_badge(qtype, color="#AB68FF"):
        return f'<span style="background:{color}22;color:{color};border:1px solid {color}55;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700">{qtype}</span>'

    # ── Build CF lookup by active question ID ──
    cf_by_qid = {}
    for entry in cf_results:
        lq = entry.get("active_question", {})
        qid = lq.get("id")
        if qid:
            cf_by_qid[qid] = entry.get("cf_pipeline_result", {})

    # ── Load original trajectory from state.json ──
    orig_trajectory = []
    if scene_dir:
        state_path = Path(scene_dir) / "Tp0p0" / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            for wp in state.get("ego", {}).get("future", []):
                orig_trajectory.append((wp["x"], wp["y"]))

    # ── Determine if a question type needs counterfactual intervention ──
    def _needs_cf_intervention(category, subtype, answer_format):
        """Return True if this question type is eligible for CF trajectory generation."""
        fmt = (answer_format or "").lower()
        cat = (category or "").upper()
        sub = (subtype or "").upper()
        # Active: CR, NC always binary; CDE/NIE/NDE binary only
        if cat in ("CR", "NC"):
            return True
        if cat in ("CDE", "NIE", "NDE") and fmt == "binary":
            return True
        # Dormant: AB binary only
        if cat == "AB" and fmt == "binary":
            return True
        # Distractor: WC, TWC
        if cat in ("WC", "TWC"):
            return True
        return False

    # ── Build QA cards ──
    def _qa_cards(questions, card_type="active"):
        html = ""
        for q in questions:
            if not isinstance(q, dict):
                continue
            if not q.get("question") and not q.get("id"):
                continue
            qid = q.get("id", "?")
            rung = q.get("rung", "?")
            # Uniform type extraction: category first, then question_type fallback
            category = q.get("category", q.get("question_type", "?"))
            subtype = q.get("subtype", "")
            graph_struct = q.get("graph_structure", "")
            question = q.get("question", "")
            causes = q.get("active_causes_used", q.get("nodes_used", []))
            meta = q.get("meta", {})

            # Color by card type
            if card_type == "active":
                cat_color = "#AB68FF"
            elif card_type == "dl":
                cat_color = "#3498DB"
            else:
                cat_color = "#F39C12"

            # Support both old (free-text "answer") and new (MCQ/binary) formats
            answer = q.get("answer", "")
            correct_answer = q.get("correct_answer", "")
            answer_format = q.get("answer_format", "")
            options = q.get("options", [])
            reasoning = q.get("reasoning", "")

            # CF intervention eligibility
            needs_cf = _needs_cf_intervention(category, subtype, answer_format)
            has_cf = bool(meta.get("counterfactual_intervention"))
            cf_badge = ""
            if needs_cf:
                if has_cf:
                    cf_badge = _type_badge("CF: generated", "#E74C3C")
                else:
                    cf_badge = _type_badge("CF: eligible", "#E67E22")

            html += f'''<div class="qa-card"><div class="qa-header">
                <span class="qa-id">{esc(qid)}</span>
                {_rung_badge(rung) if rung != "?" else ""}
                {_type_badge(esc(str(category)), cat_color)}
                {_type_badge(esc(subtype), "#8E44AD") if subtype else ""}
                {_type_badge(esc(graph_struct), "#1ABC9C") if graph_struct else ""}
                {_type_badge(esc(answer_format.upper()), "#16A085") if answer_format else ""}
                {cf_badge}
            </div>
            <div class="qa-question"><strong>Q:</strong> {esc(question)}</div>'''

            if options:
                html += '<div class="qa-options" style="margin:6px 0 6px 18px;font-size:13px">'
                for opt in options:
                    is_correct = correct_answer and str(opt).startswith(correct_answer + ")")
                    style = "color:#2ECC71;font-weight:700" if is_correct else "color:#ccc"
                    html += f'<div style="{style}">{esc(str(opt))}</div>'
                html += '</div>'

            if correct_answer:
                html += f'<div class="qa-answer"><strong>Answer:</strong> <span style="color:#2ECC71;font-weight:700">{esc(correct_answer)}</span></div>'
            elif answer:
                html += f'<div class="qa-answer"><strong>A:</strong> {esc(answer)}</div>'

            if reasoning:
                html += f'<div class="qa-reasoning" style="color:#999;font-size:12px;margin-top:4px"><strong>Reasoning:</strong> {esc(reasoning)}</div>'''

            if causes:
                causes_html = "".join(f'<span class="cause-tag">{esc(c)}</span>' for c in causes)
                html += f'<div class="qa-causes"><strong>Active causes:</strong> {causes_html}</div>'
            if meta:
                parts = []
                if meta.get("node_ids"):
                    parts.append("Nodes: " + ", ".join(meta["node_ids"]))
                if meta.get("edge_types"):
                    parts.append("Edges: " + ", ".join(meta["edge_types"]))
                if parts:
                    html += f'<div class="qa-meta">{" | ".join(parts)}</div>'

            # Show CF + original trajectory inline for Rung 3 active questions
            if card_type == "active" and rung == 3 and qid in cf_by_qid:
                cf_data = cf_by_qid[qid]
                cf_gen = cf_data.get("generator") or {}
                cf_traj = cf_gen.get("counterfactual_trajectory", [])
                if cf_traj:
                    cf_pts = ", ".join(f"({wp[0]:.2f}, {wp[1]:.2f})" for wp in cf_traj if isinstance(wp, list) and len(wp) >= 2)
                    html += f'<div style="margin-top:8px;font-size:12px;color:#E74C3C"><strong>CF Trajectory:</strong> [{cf_pts}]</div>'
                if orig_trajectory:
                    orig_pts = ", ".join(f"({wp[0]:.2f}, {wp[1]:.2f})" for wp in orig_trajectory)
                    html += f'<div style="margin-top:4px;font-size:12px;color:#2ECC71"><strong>Original Trajectory:</strong> [{orig_pts}]</div>'

            html += "</div>\n"
        return html

    def _find_traj_dir(traj_base, question_text):
        """Find trajectory folder by trying multiple slug lengths."""
        for max_len in [80, 60, 100]:
            slug = (question_text[:max_len]
                    .replace(" ", "_").replace("?", "").replace("/", "_")
                    .replace("'", "").replace('"', ""))
            candidate = traj_base / slug
            if candidate.exists():
                return candidate
            # Also try with .strip("_")
            candidate2 = traj_base / slug.strip("_")
            if candidate2.exists():
                return candidate2
        return None

    # ── Build CF cards ──
    cf_image_urls = []  # list of lists of {url, label}
    cf_html = ""
    cf_traj_dir = qa_dir / "cf_trajectories"

    for i, entry in enumerate(cf_results):
        lq = entry.get("active_question", {})
        cf = entry.get("cf_pipeline_result", {})
        if not cf:
            cf_image_urls.append([])
            continue

        gen = cf.get("generator") or {}

        question = lq.get("question", cf.get("question", ""))
        qtype = lq.get("question_type", "?")
        rung = lq.get("rung", 3)

        # Skip questions where LLM kept the ego trajectory unchanged
        if cf_traj_dir.exists():
            _q_dir = _find_traj_dir(cf_traj_dir, question)
            if _q_dir:
                _subdirs = [d for d in _q_dir.iterdir() if d.is_dir()]
                _scene_sub = _q_dir / stem if (_q_dir / stem).exists() else (_subdirs[0] if _subdirs else _q_dir)
                _llm_path = _scene_sub / "llm_ego_trajectory.json"
                if _llm_path.exists():
                    try:
                        if not json.loads(_llm_path.read_text()).get("trajectory_changed", True):
                            cf_image_urls.append([])
                            continue
                    except Exception:
                        pass

        cf_html += f'''<div class="cf-card"><div class="cf-header">
            <span class="qa-id">{esc(lq.get("id","CF-"+str(i+1)))}</span>
            {_rung_badge(rung)} {_type_badge(esc(qtype), "#AB68FF")}
            <span style="color:#E74C3C;font-weight:700;margin-left:auto">Counterfactual Trajectory</span>
        </div>
        <div class="qa-question"><strong>Q:</strong> {esc(question)}</div>'''

        # Reasoning (new format)
        reasoning = gen.get("reasoning", "")
        if reasoning:
            cf_html += f'''<div class="cf-intervention">
                <strong>Reasoning:</strong><br/>
                <div style="font-size:12px;color:#ccc;line-height:1.6;white-space:pre-wrap">{esc(reasoning)}</div>
            </div>'''

        # Trajectory waypoints
        cf_traj = gen.get("counterfactual_trajectory", [])
        if cf_traj:
            cf_html += '<table class="traj-table"><thead><tr><th>WP</th><th>x (m)</th><th>y (m)</th><th>heading (rad)</th><th>speed (m/s)</th></tr></thead><tbody>'
            for wi, wp in enumerate(cf_traj):
                if isinstance(wp, list) and len(wp) >= 4:
                    cf_html += f'<tr><td style="font-weight:700">{wi+1}</td><td>{wp[0]:.2f}</td><td>{wp[1]:.2f}</td><td>{wp[2]:.3f}</td><td>{wp[3]:.2f}</td></tr>'
            cf_html += '</tbody></table>'

        # CF images placeholder (populated by JS)
        cf_html += f'''<div style="margin:12px 0">
            <button class="cf-bev-btn" onclick="toggleCfImages({i})">Show CF Trajectory BEV</button>
            <div class="cf-img-grid" id="cf-img-grid-{i}" style="display:none"></div>
        </div>'''

        cf_html += '</div>\n'

        # Copy CF images — match by question slug, not index
        entry_imgs = []
        if cf_traj_dir.exists():
            q_dir = _find_traj_dir(cf_traj_dir, question)
            if q_dir:
                scene_subdir = q_dir / stem if (q_dir / stem).exists() else q_dir
                subdirs = [d for d in q_dir.iterdir() if d.is_dir()]
                if subdirs and not (q_dir / stem).exists():
                    scene_subdir = subdirs[0]
                # Auto-render BEV images if missing
                result_path = scene_subdir / "result.json"
                if result_path.exists() and scene_dir:
                    try:
                        import sys as _sys
                        _traj_dir = str(Path(__file__).resolve().parent.parent / "counterfactual_trajectory")
                        if _traj_dir not in _sys.path:
                            _sys.path.insert(0, _traj_dir)
                        from trajectory_utils import process_cf_result, render_llm_trajectory, render_original_trajectory
                        records_dir = Path(scene_dir).parent.parent
                        process_cf_result(result_path, records_dir)
                        render_original_trajectory(result_path, records_dir)
                        render_llm_trajectory(result_path, records_dir)
                    except Exception as _e:
                        pass  # rendering is best-effort; missing images just won't show
                for img_name, label in [("bev_original.png", "BEV Original (GREEN)"),
                                        ("bev_counterfactual.png", "BEV Counterfactual (RED)"),
                                        ("bev_intervention.png", "BEV Intervention"),
                                        ("bev_llm_trajectory.png", "BEV LLM Ego Trajectory"),
                                        ("bev_nuplan_trajectory.png", "BEV nuPlan Trajectory"),
                                        ("bev_overlay.png", "BEV Combined"),
                                        ("cam_original.png", "Camera Original (GREEN)"),
                                        ("cam_counterfactual.png", "Camera Counterfactual (RED)"),
                                        ("cam_proj.png", "Camera Combined")]:
                    img_path = scene_subdir / img_name
                    if img_path.exists():
                        url = _rel_img(str(img_path), html_dir)
                        entry_imgs.append({"url": url, "label": label})
                # Reference sim videos (nuPlan planner outputs)
                for vid_name, label in [
                    ("sim_pdm_original.mp4", "Original Simulation"),
                    ("sim_pdm_cf.mp4", "PDM Counterfactual"),
                    ("sim_diffusion_cf.mp4", "Diffusion Counterfactual"),
                    ("sim_auto_cf.mp4", "Auto Planner Counterfactual"),
                ]:
                    vid_path = scene_subdir / vid_name
                    if vid_path.exists():
                        url = _rel_img(str(vid_path), html_dir)
                        entry_imgs.append({"url": url, "label": label, "type": "video"})
        cf_image_urls.append(entry_imgs)

    # ── Build AC cards (same structure as CF but for AC-Binary activation trajectories) ──
    ac_image_urls = []
    ac_html = ""
    ac_traj_dir = qa_dir / "ac_trajectories"

    for i, entry in enumerate(ac_results):
        dq = entry.get("dl_question", {})
        ac = entry.get("ac_pipeline_result", {})
        if not ac:
            ac_image_urls.append([])
            continue

        gen = ac.get("generator") or {}

        question = dq.get("question", ac.get("question", ""))
        qtype = dq.get("subtype", dq.get("question_type", "AC-Binary"))

        ac_html += f'''<div class="cf-card" style="border-color:#F39C1244"><div class="cf-header">
            <span class="qa-id">{esc(dq.get("id","AC-"+str(i+1)))}</span>
            {_type_badge(esc(qtype), "#F39C12")}
            <span style="color:#F39C12;font-weight:700;margin-left:auto">Activation Trajectory</span>
        </div>
        <div class="qa-question"><strong>Q:</strong> {esc(question)}</div>'''

        reasoning = gen.get("reasoning", "")
        if reasoning:
            ac_html += f'''<div class="cf-intervention">
                <strong>Reasoning:</strong><br/>
                <div style="font-size:12px;color:#ccc;line-height:1.6;white-space:pre-wrap">{esc(reasoning)}</div>
            </div>'''

        ac_traj = gen.get("counterfactual_trajectory", [])
        if ac_traj:
            ac_html += '<table class="traj-table"><thead><tr><th>WP</th><th>x (m)</th><th>y (m)</th><th>heading (rad)</th><th>speed (m/s)</th></tr></thead><tbody>'
            for wi, wp in enumerate(ac_traj):
                if isinstance(wp, list) and len(wp) >= 4:
                    ac_html += f'<tr><td style="font-weight:700">{wi+1}</td><td>{wp[0]:.2f}</td><td>{wp[1]:.2f}</td><td>{wp[2]:.3f}</td><td>{wp[3]:.2f}</td></tr>'
            ac_html += '</tbody></table>'

        ac_idx = n_cf + i  # offset index to avoid colliding with CF image grid IDs
        ac_html += f'''<div style="margin:12px 0">
            <button class="cf-bev-btn" style="background:#F39C1222;color:#F39C12;border-color:#F39C1255" onclick="toggleCfImages({ac_idx})">Show AC Trajectory BEV</button>
            <div class="cf-img-grid" id="cf-img-grid-{ac_idx}" style="display:none"></div>
        </div>'''

        ac_html += '</div>\n'

        # Copy AC images
        entry_imgs = []
        if ac_traj_dir.exists():
            q_dir = _find_traj_dir(ac_traj_dir, question)
            if q_dir:
                scene_subdir = q_dir / stem if (q_dir / stem).exists() else q_dir
                subdirs = [d for d in q_dir.iterdir() if d.is_dir()]
                if subdirs and not (q_dir / stem).exists():
                    scene_subdir = subdirs[0]
                for img_name, label in [("bev_original.png", "BEV Original (GREEN)"),
                                        ("bev_counterfactual.png", "BEV Activation (RED)"),
                                        ("bev_intervention.png", "BEV Intervention"),
                                        ("bev_llm_trajectory.png", "BEV LLM Ego Trajectory"),
                                        ("bev_nuplan_trajectory.png", "BEV nuPlan Trajectory"),
                                        ("bev_overlay.png", "BEV Combined"),
                                        ("cam_original.png", "Camera Original (GREEN)"),
                                        ("cam_counterfactual.png", "Camera Activation (RED)"),
                                        ("cam_proj.png", "Camera Combined")]:
                    img_path = scene_subdir / img_name
                    if img_path.exists():
                        url = _rel_img(str(img_path), html_dir)
                        entry_imgs.append({"url": url, "label": label})
        ac_image_urls.append(entry_imgs)

    # ── CSS ──
    # CSS is now in css/qa_section.css and loaded by create_html_visualization

    # ── Body HTML ──
    active_html = _qa_cards(active_qs, "active")
    dl_html = _qa_cards(dl_qs, "dl")

    # Only show tabs that have content
    cf_tab = f'<div class="qa-tab" onclick="switchQaTab(\'cf\')">CF Trajectories <span class="qa-tab-badge">{n_cf}</span></div>' if n_cf > 0 else ""
    cf_content = f'<div class="qa-tab-content" id="qa-tab-cf">{cf_html}</div>' if n_cf > 0 else ""
    ac_tab = f'<div class="qa-tab" onclick="switchQaTab(\'ac\')">AC Trajectories <span class="qa-tab-badge">{n_ac}</span></div>' if n_ac > 0 else ""
    ac_content = f'<div class="qa-tab-content" id="qa-tab-ac">{ac_html}</div>' if n_ac > 0 else ""
    active_tab = f'<div class="qa-tab active" onclick="switchQaTab(\'active\')">Active QA <span class="qa-tab-badge">{n_active}</span></div>' if n_active > 0 else ""
    active_content = f'<div class="qa-tab-content active" id="qa-tab-active">{active_html}</div>' if n_active > 0 else ""

    # If no active tab, make dormant the active tab
    dl_active = " active" if n_active == 0 else ""
    dl_tab = f'<div class="qa-tab{dl_active}" onclick="switchQaTab(\'dl\')">Dormant Nodes <span class="qa-tab-badge">{n_dl}</span></div>' if n_dl > 0 else ""
    dl_content = f'<div class="qa-tab-content{dl_active}" id="qa-tab-dl">{dl_html}</div>' if n_dl > 0 else ""

    distractor_html = _qa_cards(distractor_qs, "distractor")
    distractor_tab = f'<div class="qa-tab" onclick="switchQaTab(\'distractor\')">Distractor Nodes <span class="qa-tab-badge">{n_distractor}</span></div>' if n_distractor > 0 else ""
    distractor_content = f'<div class="qa-tab-content" id="qa-tab-distractor">{distractor_html}</div>' if n_distractor > 0 else ""

    preamble_html = ""
    if scene_preamble:
        preamble_html = f'<div class="qa-preamble" style="margin:8px 0 12px 0;padding:10px 14px;background:#1a1a2e;border-left:3px solid #AB68FF;border-radius:4px;color:#ccc;font-size:13px"><strong>Preamble:</strong> {esc(scene_preamble)}</div>'

    body_html = f"""
    <!-- ── Q&A Section ── -->
    <div class="qa-section-title">Causal Question &amp; Answer Pairs</div>
    {preamble_html}
    <div class="qa-tabs">
        {active_tab}
        {dl_tab}
        {distractor_tab}
        {cf_tab}
        {ac_tab}
    </div>
    {active_content}
    {dl_content}
    {distractor_content}
    {cf_content}
    {ac_content}
    <div class="qa-lightbox" id="qa-lightbox" onclick="this.classList.remove('active')"><img id="qa-lightbox-img"/></div>"""

    # ── JS ──
    all_image_urls = cf_image_urls + ac_image_urls
    cf_img_json = json.dumps(all_image_urls)
    js = f"""
        /* ── Q&A Tab switching ── */
        function switchQaTab(name) {{
            document.querySelectorAll('.qa-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.qa-tab-content').forEach(t => t.classList.remove('active'));
            event.target.closest('.qa-tab').classList.add('active');
            document.getElementById('qa-tab-' + name).classList.add('active');
        }}
        function toggleCfImages(idx) {{
            const grid = document.getElementById('cf-img-grid-' + idx);
            grid.style.display = grid.style.display === 'none' ? 'grid' : 'none';
        }}
        /* ── CF trajectory images ── */
        const cfImageUrls = {cf_img_json};
        cfImageUrls.forEach((entry, i) => {{
            const grid = document.getElementById('cf-img-grid-' + i);
            if (!grid) return;
            entry.forEach(img => {{
                const div = document.createElement('div');
                if (img.type === 'video') {{
                    div.innerHTML = `<video src="${{img.url}}" controls autoplay loop muted style="width:100%;border-radius:6px;border:1px solid #333;"></video><div class="img-label">${{img.label}}</div>`;
                    div.style.gridColumn = 'span 2';
                }} else {{
                    div.innerHTML = `<img src="${{img.url}}" alt="${{img.label}}" onclick="document.getElementById('qa-lightbox-img').src=this.src;document.getElementById('qa-lightbox').classList.add('active');event.stopPropagation()"><div class="img-label">${{img.label}}</div>`;
                }}
                grid.appendChild(div);
            }});
        }});"""

    stats = {"n_active": n_active, "n_dl": n_dl, "n_cf": n_cf, "n_ac": n_ac}
    return {"body_html": body_html, "js": js, "stats": stats,
            "cf_image_urls": cf_image_urls, "ac_image_urls": ac_image_urls,
            "active_qs": active_qs, "dl_qs": dl_qs, "distractor_qs": distractor_qs}


_RETURN_HTML = object()  # sentinel: return HTML string instead of writing file

def create_html_visualization(json_path, output_file=None, scene_dir=None, model_label=None, qa_dir=None):
    """
    Create an interactive HTML visualization with:
      - Top: BEV (left) + 4-camera × N-timestep grid (right) showing temporal history
      - Bottom: D3 force-directed causal graph
    scene_dir: path to the scenario_records/<scene>/<maneuver>/ folder
    model_label: e.g. "Cosmos-Reason2-8B" or "GPT-5.2" — shown in header
    Images are copied to an assets/ subfolder alongside the HTML for fast loading.
    """
    import re
    from time import monotonic as _now
    _t0 = _now()
    _timings = {}
    def _mark(label):
        _timings[label] = _now() - _t0

    d3_script_tag = _load_d3_tag()
    json_path = Path(json_path)

    # Try to load Stage 1 raw graph and changes
    raw_graph_data = None
    pruned_graph_data = None
    changes_data = []
    stem_no_refined = json_path.stem
    _edited_re = re.compile(r"_[A-Za-z0-9]+_edited$")
    if _edited_re.search(stem_no_refined):
        stem_no_refined = _edited_re.sub("", stem_no_refined)
    else:
        for _sfx in ("_refined", "_pruned", "_verified"):
            if stem_no_refined.endswith(_sfx):
                stem_no_refined = stem_no_refined[:-len(_sfx)]
                break
    raw_json_path = json_path.parent / f"{stem_no_refined}.json"
    pruned_json_path = json_path.parent / f"{stem_no_refined}_pruned.json"
    changes_json_path = json_path.parent / f"{stem_no_refined}_changes.json"
    verification_json_path = json_path.parent / f"{stem_no_refined}_verification.json"
    if raw_json_path.exists() and raw_json_path != json_path:
        try:
            raw_graph_data = json.loads(raw_json_path.read_text())
        except Exception:
            pass
    if pruned_json_path.exists():
        try:
            pruned_graph_data = json.loads(pruned_json_path.read_text())
        except Exception:
            pass
    if changes_json_path.exists():
        try:
            cd = json.loads(changes_json_path.read_text())
            changes_data = cd if isinstance(cd, list) else cd.get("changes", [])
        except Exception:
            pass
    verification_data = None
    if verification_json_path.exists():
        try:
            verification_data = json.loads(verification_json_path.read_text())
        except Exception:
            pass

    _mark("load_companions")

    with open(json_path) as f:
        graph_data = json.load(f)

    # Normalize to new format
    if "ego_edges" not in graph_data and "edges" in graph_data:
        graph_data["ego_edges"] = graph_data.pop("edges")
        graph_data["inter_node_edges"] = []

    return_html = output_file is _RETURN_HTML
    # Always name HTML after the base scene_id (not _verified/_edited variant)
    base_html = json_path.parent / f"{stem_no_refined}.html"
    if output_file is None:
        output_file = str(base_html)
    elif return_html:
        output_file = str(base_html)

    file_stem = stem_no_refined
    scene_id = file_stem  # Use filename, not LLM-generated scene_id
    model_tag = f" [{model_label}]" if model_label else ""
    display_title = f"{scene_id}{model_tag}"

    html_dir = Path(scene_dir) if (return_html and scene_dir) else Path(output_file).parent

    _mark("load_graph")

    # ── Discover timestep folders ──────────────────────────────────────
    # Auto-detect cameras — AV2 has 5 (cam_back_right + cam_back_left), others have 4
    base_cams = ["cam_front", "cam_front_left", "cam_front_right"]
    rear_cams = ["cam_back_right", "cam_back_left", "cam_back"]
    tp0_check = Path(scene_dir) / "Tp0p0" if scene_dir else None
    CAMS = base_cams + [c for c in rear_cams if tp0_check and (tp0_check / f"{c}.jpg").exists()]
    CAM_LABELS = {"cam_front": "FRONT", "cam_front_left": "FRONT LEFT",
                  "cam_front_right": "FRONT RIGHT", "cam_back": "BACK",
                  "cam_back_right": "BACK RIGHT", "cam_back_left": "BACK LEFT"}
    img_bev = ""
    bev_temporal = []  # list of (label, img_url) for temporal BEV sequence
    # timesteps: list of (slug_label, {cam: img_url})
    timesteps = []

    if scene_dir:
        sd = Path(scene_dir)
        img_bev = _rel_img(sd / "Tp0p0" / "bev.png", html_dir)

        # Find all T* subdirs, sort by time
        def _slug_to_seconds(slug):
            s = slug.replace("Tp", "+").replace("Tm", "-").replace("p", ".")
            try:
                return float(s)
            except ValueError:
                return 999

        t_dirs = []
        for d in sorted(sd.iterdir()):
            if d.is_dir() and re.match(r"T[mp]\d", d.name):
                t_dirs.append(d)
        t_dirs.sort(key=lambda d: _slug_to_seconds(d.name))

        for td in t_dirs:
            # Human-readable label: Tm1p5 → T-1.5s, Tp0p0 → T=0.0s
            sec = _slug_to_seconds(td.name)
            label = f"T={sec:+.1f}s" if sec != 0 else "T=0 (now)"
            cam_imgs = {}
            for cam in CAMS:
                cam_imgs[cam] = _rel_img(td / f"{cam}.jpg", html_dir)
            timesteps.append((label, td.name, cam_imgs))
            # Collect temporal BEV images
            bev_img = _rel_img(td / "bev.png", html_dir)
            if bev_img:
                bev_temporal.append((label, bev_img))

    # Detect which timesteps have actual camera images
    cam_timesteps = [(label, slug, cam_imgs) for label, slug, cam_imgs in timesteps
                     if any(cam_imgs.get(c) for c in CAMS)]
    t0_only_cams = len(cam_timesteps) == 1 and cam_timesteps[0][1] == "Tp0p0"
    n_cols = len(cam_timesteps) if cam_timesteps else 1

    # Build camera grid HTML rows: one row per camera, cells are timesteps
    cam_grid_html = ""
    for cam in CAMS:
        for i, (label, slug, cam_imgs) in enumerate(cam_timesteps):
            is_t0 = "Tp0p0" in slug
            border = "border: 2px solid #3498DB;" if is_t0 else ""
            cam_grid_html += f'''            <div class="cam-cell" style="{border}">
                <img src="{cam_imgs.get(cam, '')}" alt="{cam} {label}">
                <div class="cam-label">{CAM_LABELS[cam]}<br/>{label}</div>
            </div>\n'''

    # Model label color mapping
    model_color = "#F39C12"  # default orange
    if model_label:
        if "cosmos" in model_label.lower():
            model_color = "#00D4AA"  # teal/green for Cosmos
        elif "gpt" in model_label.lower():
            model_color = "#AB68FF"  # purple for GPT

    # Data-driven grid layout (set as inline styles on the elements)
    scene_view_cols = '1fr' if len(bev_temporal) > 1 else 'minmax(0, 25%) 1fr'
    cam_grid_cols = f'repeat({len(CAMS) if t0_only_cams else n_cols}, 1fr)'
    cam_grid_rows = '1fr' if t0_only_cams else f'repeat({len(CAMS)}, 1fr)'

    json_pretty = json.dumps(graph_data, indent=2)
    # Escape backticks and backslashes for JS template literal
    json_for_js = json_pretty.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    _mark("discover_images")

    # ── Stage 1 / Stage 2 / Pruned toggle support ──
    stage1_label = "Original"
    pruned_label = "Pruned" if pruned_graph_data is not None else "Pruned — Not available"

    # Look for _verified.json
    verified_json_path = json_path.parent / f"{stem_no_refined}_verified.json"
    verified_graph_data = None
    if verified_json_path.exists():
        try:
            verified_graph_data = json.loads(verified_json_path.read_text())
        except Exception:
            pass
    has_verified = verified_graph_data is not None
    verified_label = "Verified" if has_verified else "Verified — Not available"

    # Look for any {stem}_<user>_edited.json (first match for determinism)
    edited_graph_data = None
    edited_user = None
    _edit_cands = sorted(json_path.parent.glob(f"{stem_no_refined}_*_edited.json"))
    if _edit_cands:
        try:
            edited_graph_data = json.loads(_edit_cands[0].read_text())
            _em = re.search(r"_([A-Za-z0-9]+)_edited$", _edit_cands[0].stem)
            edited_user = _em.group(1) if _em else "?"
        except Exception:
            pass
    has_edited = edited_graph_data is not None
    edited_label = f"Edited ({edited_user})" if has_edited else "Edited — Not available"

    # Look for _fixed.json companion
    fixed_json_path = json_path.parent / f"{stem_no_refined}_fixed.json"
    fixed_graph_data = None
    if fixed_json_path.exists():
        try:
            fixed_graph_data = json.loads(fixed_json_path.read_text())
        except Exception:
            pass
    has_fixed = fixed_graph_data is not None
    fixed_label = "Fixed" if has_fixed else "Fixed — Not available"
    # ── Load scene record metadata for download ──
    scene_record = {}
    if scene_dir:
        sd = Path(scene_dir)
        meta_path = sd / "meta.json"
        if meta_path.exists():
            try:
                scene_record.update(json.loads(meta_path.read_text()))
            except Exception:
                pass
        state_path = sd / "Tp0p0" / "state.json"
        if state_path.exists():
            try:
                scene_record["state"] = json.loads(state_path.read_text())
            except Exception:
                pass
        calib_path = sd / "Tp0p0" / "calib.json"
        if calib_path.exists():
            try:
                scene_record["calibration"] = json.loads(calib_path.read_text())
            except Exception:
                pass
    scene_record_json = json.dumps(scene_record)

    # Build changes summary text
    if changes_data:
        n_removed = sum(1 for c in changes_data if isinstance(c, dict) and c.get("action") in ("removed", "deleted", "remove", "delete"))
        n_added   = sum(1 for c in changes_data if isinstance(c, dict) and c.get("action") in ("added", "add"))
        n_modified = sum(1 for c in changes_data if isinstance(c, dict) and c.get("action") in ("modified", "updated", "modify", "update"))
        n_total = len(changes_data)
        parts = []
        if n_removed:  parts.append(f"{n_removed} removed")
        if n_added:    parts.append(f"{n_added} added")
        if n_modified: parts.append(f"{n_modified} modified")
        detail = f" ({', '.join(parts)})" if parts else ""
        changes_summary_text = f"{n_total} changes{detail}"
    elif pruned_graph_data is not None:
        # Show pruning stats
        raw_nodes = len([n for n in graph_data.get("nodes", []) if n.get("id") != "ego"])
        pruned_nodes = len([n for n in pruned_graph_data.get("nodes", []) if n.get("id") != "ego"])
        n_pruned = raw_nodes - pruned_nodes
        changes_summary_text = f"{n_pruned} node(s) pruned" if n_pruned > 0 else "No nodes pruned"
    else:
        changes_summary_text = ""
    raw_graph_json = json.dumps(raw_graph_data) if raw_graph_data is not None else "null"
    pruned_graph_json = json.dumps(pruned_graph_data) if pruned_graph_data is not None else "null"
    changes_json = json.dumps(changes_data)
    verification_json = json.dumps(verification_data) if verification_data is not None else "null"

    _mark("scene_metadata")

    # ── Q&A section (optional) ──
    qa = {"body_html": "", "js": "", "stats": {}}
    if qa_dir:
        qa_stem = Path(json_path).stem
        # Strip _refined suffix to match QA dir naming
        for sfx in ("_refined", "_raw"):
            if qa_stem.endswith(sfx):
                qa_stem = qa_stem[:-len(sfx)]
                break
        # Resolve per-scene QA directory: try <qa_dir>/<stem>/ first,
        # then <qa_dir>/qa_results/<stem>/, then <qa_dir> as-is.
        resolved_qa_dir = Path(qa_dir)
        scene_qa = Path(qa_dir) / qa_stem
        scene_qa_alt = Path(qa_dir) / "qa_results" / qa_stem
        if scene_qa.is_dir() and (scene_qa / "active_qa.json").exists():
            resolved_qa_dir = scene_qa
        elif scene_qa_alt.is_dir() and (scene_qa_alt / "active_qa.json").exists():
            resolved_qa_dir = scene_qa_alt
        elif scene_qa.is_dir() and (scene_qa / "dormant_qa.json").exists():
            resolved_qa_dir = scene_qa
        elif scene_qa_alt.is_dir() and (scene_qa_alt / "dormant_qa.json").exists():
            resolved_qa_dir = scene_qa_alt
        qa = _build_qa_section(str(resolved_qa_dir), str(html_dir), qa_stem, scene_dir=scene_dir)

    _mark("qa_section")

    qa_body = qa["body_html"]
    qa_js = qa["js"]
    qa_stats = qa["stats"]

    # Build combined QA download from already-loaded data (no re-reads)
    qa_result_json = "null"
    if qa_dir:
        combined_qa = {"scene_id": scene_id, "questions": []}
        for qs, source in [(qa.get("active_qs", []), "active_qa"),
                           (qa.get("dl_qs", []), "dormant_qa"),
                           (qa.get("distractor_qs", []), "distractor_qa")]:
            for q in qs:
                if isinstance(q, dict):
                    q["_source"] = source
                    combined_qa["questions"].append(q)
        if combined_qa["questions"]:
            qa_result_json = json.dumps(combined_qa).replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    # Load scene data for record.json download
    record_data = {"scene_id": scene_id}
    if scene_dir:
        sd = Path(scene_dir)
        tp0 = sd / "Tp0p0"
        for fname, key in [("state.json", "state"), ("meta.json", "meta"),
                           ("calib.json", "calib"), ("uuid_map.json", "uuid_map")]:
            fpath = tp0 / fname
            if not fpath.exists():
                fpath = sd / fname  # meta.json is at scene level
            if fpath.exists():
                record_data[key] = json.loads(fpath.read_text())
        # Raw original sensor image paths per timestep
        try:
            if not hasattr(create_html_visualization, '_raw_frames_fn'):
                import sys as _sys
                from pathlib import Path as _Path
                _common = str(_Path(__file__).parent.parent / "common")
                if _common not in _sys.path:
                    _sys.path.insert(0, _common)
                _scripts = str(_Path(__file__).parent.parent)
                if _scripts not in _sys.path:
                    _sys.path.insert(0, _scripts)
                from common.scene_utils import get_raw_frames
                from config_loader import load_config as _load_config
                create_html_visualization._raw_frames_fn = get_raw_frames
                create_html_visualization._cfg = _load_config() or {}
            record_data["frames"] = create_html_visualization._raw_frames_fn(
                sd, create_html_visualization._cfg)
        except Exception:
            record_data["frames"] = {}
        record_data["causal_graph"] = {
            "nodes": graph_data.get("nodes", []),
            "ego_edges": graph_data.get("ego_edges", []),
            "inter_node_edges": graph_data.get("inter_node_edges", []),
            "distractors": graph_data.get("distractors", []),
        }
    _mark("record_data")

    record_json_for_js = json.dumps(record_data).replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    def _js_safe(obj):
        return json.dumps(obj).replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    meta_json_for_js   = _js_safe(record_data.get("meta", {}))
    calib_json_for_js  = _js_safe(record_data.get("calib", {}))
    state_json_for_js  = _js_safe(record_data.get("state", {}))
    frames_json_for_js = _js_safe(record_data.get("frames", {}))
    graph_for_download = pruned_graph_data if pruned_graph_data is not None else graph_data
    graph_json_for_js  = _js_safe(graph_for_download)

    qa_stats_html = ""
    if qa_stats.get("n_active") or qa_stats.get("n_dl") or qa_stats.get("n_cf"):
        qa_stats_html = f'''<div class="stat"><div class="val">{qa_stats.get("n_active",0)}</div>Active Q&A</div>
        <div class="stat"><div class="val">{qa_stats.get("n_dl",0)}</div>Dormant Q&A</div>
        <div class="stat"><div class="val">{qa_stats.get("n_cf",0)}</div>CF Traj</div>'''

    def _tab_attrs(enabled: bool) -> str:
        if not enabled:
            return 'disabled style="padding:5px 14px;border-radius:4px;border:1px solid #444;background:transparent;color:#555;cursor:not-allowed;font-size:13px;"'
        return 'style="padding:5px 14px;border-radius:4px;border:1px solid #555;background:transparent;color:#aaa;cursor:pointer;font-size:13px;"'

    pruned_button_attrs = _tab_attrs(pruned_graph_data is not None)
    edited_button_attrs = _tab_attrs(has_edited)
    fixed_button_attrs = _tab_attrs(has_fixed)
    verified_button_attrs = _tab_attrs(has_verified)
    edited_graph_json = json.dumps(edited_graph_data) if edited_graph_data is not None else "null"
    fixed_graph_json = json.dumps(fixed_graph_data) if fixed_graph_data is not None else "null"
    verified_graph_json = json.dumps(verified_graph_data) if verified_graph_data is not None else "null"

    from scene_viewer_template import render_html as _render_html
    html_content = _render_html(
        display_title=display_title,
        d3_script_tag=d3_script_tag,
        model_color=model_color,
        scene_id=scene_id,
        model_label=model_label,
        n_nodes=len(graph_data.get("nodes", [])),
        n_ego_edges=len(graph_data.get("ego_edges", [])),
        n_inter_edges=len(graph_data.get("inter_node_edges", [])),
        n_cols=n_cols,
        qa_stats_html=qa_stats_html,
        bev_temporal=bev_temporal,
        bev_temporal_html="",
        scene_view_cols=scene_view_cols,
        img_bev=img_bev,
        cam_grid_cols=cam_grid_cols,
        cam_grid_rows=cam_grid_rows,
        cam_grid_html=cam_grid_html,
        stage1_label=stage1_label,
        pruned_button_attrs=pruned_button_attrs,
        pruned_label=pruned_label,
        edited_label=edited_label,
        edited_button_attrs=edited_button_attrs,
        has_edited="true" if has_edited else "false",
        edited_graph_json=edited_graph_json,
        fixed_label=fixed_label,
        fixed_button_attrs=fixed_button_attrs,
        has_fixed="true" if has_fixed else "false",
        fixed_graph_json=fixed_graph_json,
        verified_label=verified_label,
        verified_button_attrs=verified_button_attrs,
        has_verified="true" if has_verified else "false",
        verified_graph_json=verified_graph_json,
        changes_summary_text=changes_summary_text,
        qa_body=qa_body,
        graph_data_json=json.dumps(graph_data),
        raw_graph_json=raw_graph_json,
        pruned_graph_json=pruned_graph_json,
        changes_json=changes_json,
        scene_record_json=scene_record_json,
        json_for_js=json_for_js,
        meta_json_for_js=meta_json_for_js,
        frames_json_for_js=frames_json_for_js,
        calib_json_for_js=calib_json_for_js,
        state_json_for_js=state_json_for_js,
        graph_json_for_js=graph_json_for_js,
        qa_result_json=qa_result_json,
        qa_js=qa_js,
        verification_json=verification_json,
        read_css=_read_css,
    )

    _mark("template")

    if return_html:
        return html_content

    with open(output_file, "w") as f:
        f.write(html_content)
    _mark("write")

    # Print timing breakdown
    prev = 0
    parts = []
    for label, t in _timings.items():
        parts.append(f"{label}={t - prev:.2f}s")
        prev = t
    print(f"    timings: {' | '.join(parts)} | total={_now() - _t0:.2f}s", flush=True)

    return output_file


def create_index_html(html_dir, scene_files, model_label=None):
    """Create an index.html that links to all scene HTMLs."""
    links_html = ""
    for name, filename in sorted(scene_files):
        links_html += f'        <a href="{filename}">{name}</a>\n'

    model_tag = f" — {model_label}" if model_label else ""
    index = f"""<!DOCTYPE html>
<html>
<head>
    <title>STSBench — Causal Scene Graphs{model_tag}</title>
    <style>{_read_css("index.css")}</style>
</head>
<body>
    <h1>STSBench — Causal Scene Graphs{model_tag}</h1>
{links_html}
</body>
</html>"""
    idx_path = Path(html_dir) / "index.html"
    with open(idx_path, "w") as f:
        f.write(index)
    print(f"  Index → {idx_path}")
    return str(idx_path)


def create_unified_index(out_dir, cosmos_dir=None, gpt_dir=None):
    """Create a unified index.html that shows both Cosmos and GPT graphs side-by-side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect scenes from each model dir
    def _collect_scenes(d, prefix):
        scenes = []
        if d and Path(d).exists():
            for f in sorted(Path(d).glob("scene-*.html")):
                scene_name = f.stem  # e.g. scene-0018_left_turn
                scenes.append((scene_name, f"{prefix}/{f.name}"))
        return scenes

    cosmos_scenes = _collect_scenes(cosmos_dir, "cosmos")
    gpt_scenes = _collect_scenes(gpt_dir, "gpt")

    # Build a merged list of unique scene names
    all_scene_names = sorted(set(s[0] for s in cosmos_scenes) | set(s[0] for s in gpt_scenes))
    cosmos_map = {s[0]: s[1] for s in cosmos_scenes}
    gpt_map = {s[0]: s[1] for s in gpt_scenes}

    # Build table rows
    rows_html = ""
    for name in all_scene_names:
        # Parse scene info
        parts = name.split("_", 1)
        scene_id = parts[0] if parts else name
        maneuver = parts[1].replace("_", " ").title() if len(parts) > 1 else ""

        cosmos_link = f'<a href="{cosmos_map[name]}" class="btn cosmos">View</a>' if name in cosmos_map else '<span class="na">--</span>'
        gpt_link = f'<a href="{gpt_map[name]}" class="btn gpt">View</a>' if name in gpt_map else '<span class="na">--</span>'

        rows_html += f"""            <tr>
                <td class="scene-name">{scene_id}</td>
                <td class="maneuver">{maneuver}</td>
                <td>{cosmos_link}</td>
                <td>{gpt_link}</td>
            </tr>\n"""

    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>STSBench — Causal Scene Graphs</title>
    <style>{_read_css("unified_index.css")}</style>
</head>
<body>
    <div class="container">
        <h1>STSBench Causal Scene Graphs</h1>
        <div class="tagline">NuScenes scenario mining with causal graph generation</div>
        <div class="model-chips">
            <div class="chip cosmos">Cosmos-Reason2-8B</div>
            <div class="chip gpt">GPT-5.2</div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Scene</th>
                    <th>Maneuver</th>
                    <th>Cosmos-Reason2-8B</th>
                    <th>GPT-5.2</th>
                </tr>
            </thead>
            <tbody>
{rows_html}            </tbody>
        </table>
        <div class="footer">
            {len(all_scene_names)} scenes | {len(cosmos_scenes)} Cosmos | {len(gpt_scenes)} GPT
        </div>
    </div>
</body>
</html>"""
    idx_path = out_dir / "index.html"
    with open(idx_path, "w") as f:
        f.write(index_html)
    print(f"  Unified index → {idx_path}")
    return str(idx_path)


if __name__ == "__main__":
    import argparse
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent))
    from config_loader import load_config as _load_config

    parser = argparse.ArgumentParser(
        description="Render causal scene graph HTML visualizations.",
    )
    parser.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"], required=True,
                        help="Dataset to render (reads output/records paths from config.yaml)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: config[dataset].outputs)")
    parser.add_argument("--records-dir", default=None,
                        help="Override records directory (default: config[dataset].records)")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML viewer with BEV, cameras, and D3 graph")
    parser.add_argument("--only-with-qa", action="store_true",
                        help="Only render scenes that have QA results (active_qa.json, dormant_qa.json, or distractor_qa.json)")
    parser.add_argument("--index", action="store_true",
                        help="Generate index.html for the output directory")
    parser.add_argument("--root-index", action="store_true",
                        help="Regenerate the root index.html across all datasets")
    parser.add_argument("--serve", type=int, metavar="PORT", default=None,
                        help="Serve the output directory over HTTP on PORT after rendering")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if output HTML already exists")
    parser.add_argument("--model-label", default=None,
                        help="Model label shown in the HTML header")
    parser.add_argument("--qa-dir", default=None,
                        help="Directory containing QA results (auto-detected if omitted)")
    parser.add_argument("--unified-index", default=None,
                        help="Output dir for a unified comparison index (needs --cosmos-dir/--gpt-dir)")
    parser.add_argument("--cosmos-dir", default=None)
    parser.add_argument("--gpt-dir", default=None)
    args = parser.parse_args()

    _cfg = _load_config(args.dataset) or {}

    # Resolve output and records dirs
    _out_dir = _Path(args.output_dir or _cfg.get("outputs", ""))
    _records_dir = args.records_dir or _cfg.get("records", "")
    if not _out_dir or not _out_dir.exists():
        print(f"ERROR: output dir not found: {_out_dir}", file=_sys.stderr)
        _sys.exit(1)

    # Discover best graph per scene: verified > <user>_edited > pruned > raw
    # Only consider files matching the scene naming pattern (e.g. nuscenes-scene-XXXX_SAMPLED_N)
    import re as _re
    _SCENE_PATTERN = _re.compile(r'^.+_SAMPLED(?:_\d+)?$')
    _FIXED_COMPANIONS = ("_pruned", "_changes", "_verification", "_verified")
    _EDITED_RE = _re.compile(r"_[A-Za-z0-9]+_edited$")

    def _strip_variant_suffix(stem: str) -> str:
        if _EDITED_RE.search(stem):
            return _EDITED_RE.sub("", stem)
        for _sfx in ("_refined", "_pruned", "_verified", "_raw"):
            if stem.endswith(_sfx):
                return stem[:-len(_sfx)]
        return stem

    _raw_files = {}
    for p in _out_dir.glob("*.json"):
        if any(p.stem.endswith(s) for s in _FIXED_COMPANIONS):
            continue
        if _EDITED_RE.search(p.stem):
            continue
        if not _SCENE_PATTERN.match(p.stem):
            continue
        _raw_files[p.stem] = p

    # For each scene_id, pick the best available graph
    render_files = []
    for scene_id in sorted(_raw_files):
        best = None
        # 1. verified
        cand = _out_dir / f"{scene_id}_verified.json"
        if cand.exists():
            best = cand
        # 2. any <user>_edited.json (first alphabetically for determinism)
        if best is None:
            edits = sorted(_out_dir.glob(f"{scene_id}_*_edited.json"))
            if edits:
                best = edits[0]
        # 3. pruned
        if best is None:
            cand = _out_dir / f"{scene_id}_pruned.json"
            if cand.exists():
                best = cand
        # 4. raw
        if best is None:
            best = _raw_files[scene_id]
        render_files.append(str(best))

    # Filter to only scenes with QA results if requested
    if args.only_with_qa:
        _QA_FILES = ("active_qa.json", "dormant_qa.json", "distractor_qa.json")
        _before = len(render_files)
        _filtered = []
        for fp in render_files:
            _fp_stem = _strip_variant_suffix(_Path(fp).stem)
            # Check qa_results/<scene_id>/ and <scene_id>/ dirs
            has_qa = False
            for _qa_parent in [_out_dir / "qa_results" / _fp_stem, _out_dir / _fp_stem]:
                if _qa_parent.is_dir() and any((_qa_parent / f).exists() for f in _QA_FILES):
                    has_qa = True
                    break
            if has_qa:
                _filtered.append(fp)
        render_files = _filtered
        print(f"  --only-with-qa: {_before} → {len(render_files)} scenes with QA results")

    scene_html_files = []
    n_ok = 0
    n_skip = 0
    n_cached = 0

    _n_total = len(render_files)
    print(f"Rendering {_n_total} files ...")

    import re as _render_re
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _render_one(fp, force, records_dir, out_dir, model_label, explicit_qa_dir):
        """Render a single scene — designed to run in a thread."""
        fp_path = _Path(fp)
        base_stem = _strip_variant_suffix(fp_path.stem)

        if not force:
            expected_html = fp_path.parent / f"{base_stem}.html"
            if expected_html.exists():
                return ("cached", base_stem, expected_html.name)

        scene_dir = None
        if records_dir:
            m = _render_re.match(r'^(.+)_(SAMPLED(?:_\d+)?)$', base_stem)
            if m:
                candidate = _Path(records_dir) / m.group(1) / m.group(2)
                if candidate.exists():
                    scene_dir = str(candidate)

        qa_dir = explicit_qa_dir
        if not qa_dir:
            for candidate_qa in [out_dir / "qa_results" / base_stem, out_dir / base_stem]:
                if candidate_qa.is_dir() and any(
                    (candidate_qa / f).exists()
                    for f in ("active_qa.json", "dormant_qa.json", "distractor_qa.json")
                ):
                    qa_dir = str(candidate_qa)
                    break

        html_path = create_html_visualization(fp, scene_dir=scene_dir,
                                              model_label=model_label, qa_dir=qa_dir)
        return ("ok", base_stem, _Path(html_path).name)

    _N_WORKERS = 4
    _futures = {}
    with ThreadPoolExecutor(max_workers=_N_WORKERS) as _pool:
        for fp in render_files:
            fut = _pool.submit(_render_one, fp, args.force, _records_dir,
                               _out_dir, args.model_label, args.qa_dir)
            _futures[fut] = fp

        _done = 0
        for fut in as_completed(_futures):
            _done += 1
            try:
                status, base_stem, html_name = fut.result()
                if status == "cached":
                    scene_html_files.append((base_stem, html_name))
                    n_cached += 1
                else:
                    scene_html_files.append((base_stem, html_name))
                    n_ok += 1
                print(f"  [{_done}/{_n_total}] {base_stem} ... {status}", flush=True)
            except Exception as e:
                fp = _futures[fut]
                print(f"  [{_done}/{_n_total}] {_Path(fp).stem} ... SKIP ({e})", flush=True)
                n_skip += 1

    cached_msg = f", {n_cached} already up-to-date" if n_cached else ""
    print(f"Done: {n_ok} rendered{f', {n_skip} skipped' if n_skip else ''}{cached_msg}.")

    # Generate index.html
    if args.index:
        _rendering_dir = str(_Path(__file__).parent)
        if _rendering_dir not in _sys.path:
            _sys.path.insert(0, _rendering_dir)
        from generate_index import generate_index as _generate_index
        _generate_index(_out_dir)

    # Unified comparison index
    if args.unified_index:
        create_unified_index(args.unified_index,
                             cosmos_dir=args.cosmos_dir, gpt_dir=args.gpt_dir)

    # Root index across all datasets
    if args.root_index:
        _rendering_dir = str(_Path(__file__).parent)
        if _rendering_dir not in _sys.path:
            _sys.path.insert(0, _rendering_dir)
        from gen_root_index import generate_index as _gen_root_index
        _top_cfg = _load_config() or {}
        _outputs_base = _Path(_top_cfg.get("outputs_base", ""))
        if _outputs_base.exists():
            _gen_root_index(_outputs_base)
        else:
            print(f"  WARNING: outputs_base not found: {_outputs_base}")

    # HTTP server
    if args.serve:
        import http.server
        import socketserver
        # Serve from parent of outputs/ so records/ is reachable alongside outputs/
        _outputs_dir = _out_dir.parent       # e.g. Neurips/outputs/
        serve_dir = _outputs_dir.parent      # e.g. Neurips/
        serve_url_path = f"{_outputs_dir.name}/{_out_dir.name}"
        port = args.serve

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(serve_dir), **kw)
            def log_message(self, fmt, *a):
                pass

        with socketserver.TCPServer(("", port), _Handler) as httpd:
            print(f"Serving {serve_dir} at http://localhost:{port}/{serve_url_path}/  (Ctrl+C to stop)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")


