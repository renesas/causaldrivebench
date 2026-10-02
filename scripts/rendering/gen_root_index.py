#!/usr/bin/env python3
"""
Generate root index.html — a styled dashboard of all output directories.

Each output directory is shown as a card with its scene/sample count,
linking to its own index.html for per-scene browsing.

Usage:
  python scripts/rendering/gen_root_index.py
  python scripts/rendering/gen_root_index.py --base /path/to/outputs
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config as _load_config


def _default_outputs_base() -> Path:
    cfg = _load_config() or {}
    val = cfg.get("outputs_base", "")
    return Path(val) if val else Path()


DATASET_COLORS = {
    "nuscenes":   "#7b8cf0",
    "argoverse2": "#2ECC71",
    "openscene":  "#F39C12",
}

DATASET_LABELS = {
    "nuscenes":   "nuScenes",
    "argoverse2": "Argoverse 2",
    "openscene":  "OpenScene",
}


def _dir_color(dir_name: str) -> str:
    for ds_key, color in DATASET_COLORS.items():
        if dir_name.startswith(ds_key):
            return color
    return "#8b9cf7"


def _dir_label(dir_name: str) -> str:
    for ds_key, label in DATASET_LABELS.items():
        if dir_name.startswith(ds_key):
            suffix = dir_name[len(ds_key):].lstrip("_").replace("_", " ")
            return f"{label} — {suffix}" if suffix else label
    return dir_name.replace("_", " ").title()


def _scan_dir_stats(ds_dir: Path):
    """Return (n_scenes, n_samples) by counting SAMPLED HTML files, or None if empty."""
    htmls = [
        f for f in ds_dir.glob("*_SAMPLED*.html")
        if f.name != "index.html" and "_changes" not in f.name
    ]
    if not htmls:
        # Also try _refined suffix
        htmls = list(ds_dir.glob("*_SAMPLED*_refined.html"))
    if not htmls:
        return None

    scene_ids = set()
    for f in htmls:
        m = re.match(r'^(.+?)_(SAMPLED(?:_\d+)?)(?:_refined)?\.html$', f.name)
        if m:
            scene_ids.add(m.group(1))

    return len(scene_ids), len(htmls)


def generate_index(base: Path):
    candidate_dirs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    sections = []
    for ds_dir in candidate_dirs:
        stats = _scan_dir_stats(ds_dir)
        if not stats:
            continue
        n_scenes, n_samples = stats
        name = ds_dir.name
        has_index = (ds_dir / "index.html").exists()
        sections.append({
            "name": name,
            "label": _dir_label(name),
            "color": _dir_color(name),
            "n_scenes": n_scenes,
            "n_samples": n_samples,
            "has_index": has_index,
        })

    if not sections:
        print(f"No rendered HTMLs found in {base}")
        return

    print(f"Root index: {len(sections)} output dirs in {base}:")
    for s in sections:
        idx = "✓" if s["has_index"] else "–"
        print(f"  [{idx}] {s['label']:<35} {s['n_scenes']:>4} scenes  {s['n_samples']:>5} samples")

    # Build cards HTML
    cards_html = ""
    for s in sections:
        color = s["color"]
        href = f"{s['name']}/index.html" if s["has_index"] else s["name"] + "/"
        tag = "a" if s["has_index"] else "div"
        href_attr = f' href="{href}"' if s["has_index"] else ""
        dim_style = "" if s["has_index"] else "opacity:0.45;cursor:default;"
        no_index_note = "" if s["has_index"] else '<div class="no-index">no index.html</div>'
        cards_html += f"""
        <{tag}{href_attr} class="card" style="--c:{color};{dim_style}">
            <div class="card-bar" style="background:{color}"></div>
            <div class="card-body">
                <div class="card-label">{s['label']}</div>
                <div class="card-name">{s['name']}</div>
                <div class="card-stats">
                    <span class="stat-pill">{s['n_scenes']} scenes</span>
                    <span class="stat-pill">{s['n_samples']} samples</span>
                </div>
                {no_index_note}
            </div>
            <div class="card-arrow">{'→' if s['has_index'] else ''}</div>
        </{tag}>"""

    ts = sum(s["n_scenes"] for s in sections)
    tp = sum(s["n_samples"] for s in sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NeurIPS — Causal Scene Graphs</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0a0a12; color: #c8cad0;
    padding: 40px;
    min-height: 100vh;
}}
.header {{
    border-bottom: 1px solid #1e2030;
    padding-bottom: 20px;
    margin-bottom: 32px;
}}
h1 {{
    font-size: 24px; font-weight: 600; color: #e0e2e8;
    letter-spacing: -0.5px; margin-bottom: 6px;
}}
.subtitle {{
    font-size: 13px; color: #6b7080;
}}
.subtitle span {{ color: #8b9cf7; font-weight: 600; }}
.subtitle a {{
    color: #58a6ff; text-decoration: none;
    border-bottom: 1px dashed #58a6ff66; margin-left: 16px;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
}}
.card {{
    display: flex; align-items: stretch;
    background: #12121e; border: 1px solid #1e2030;
    border-radius: 10px; overflow: hidden;
    text-decoration: none; color: inherit;
    transition: border-color 0.2s, transform 0.15s, box-shadow 0.2s;
    position: relative;
}}
a.card:hover {{
    border-color: var(--c);
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
}}
.card-bar {{
    width: 4px; flex-shrink: 0;
}}
.card-body {{
    flex: 1; padding: 16px 14px;
}}
.card-label {{
    font-size: 15px; font-weight: 600; color: #e0e2e8;
    margin-bottom: 3px;
}}
.card-name {{
    font-size: 11px; color: #4a4d5e;
    font-family: monospace; margin-bottom: 10px;
}}
.card-stats {{
    display: flex; gap: 6px; flex-wrap: wrap;
}}
.stat-pill {{
    background: #1c1c2e; border: 1px solid #2a2a40;
    border-radius: 20px; padding: 2px 10px;
    font-size: 12px; color: #8a8d9e;
}}
.no-index {{
    margin-top: 8px; font-size: 11px; color: #4a4d5e; font-style: italic;
}}
.card-arrow {{
    display: flex; align-items: center;
    padding: 0 14px; font-size: 18px;
    color: var(--c); opacity: 0.4;
    transition: opacity 0.2s;
}}
a.card:hover .card-arrow {{ opacity: 1; }}
</style>
</head>
<body>
<div class="header">
    <h1>NeurIPS — Causal Scene Graphs</h1>
    <div class="subtitle">
        <span>{ts}</span> scenes &middot; <span>{tp}</span> samples &middot;
        {len(sections)} output dirs
        <a href="downloads/">Downloads</a>
    </div>
</div>
<div class="grid">
{cards_html}
</div>
</body>
</html>"""

    out = base / "index.html"
    out.write_text(html)
    print(f"Root index → {out}  ({ts} scenes, {tp} samples, {len(sections)} dirs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate root index.html")
    parser.add_argument("--base", type=Path,
                        default=_default_outputs_base(),
                        help="Base outputs directory")
    args = parser.parse_args()
    generate_index(args.base)
