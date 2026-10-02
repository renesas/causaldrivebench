#!/usr/bin/env python3
"""
Generate index.html for any output directory.

Usage:
    python generate_index.py /path/to/outputs/nuscenes_v1
    python generate_index.py --all   # generates for all output dirs
    python generate_index.py --dataset nuscenes  # uses config paths, builds downloads
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config as _load_config

def _default_outputs_base() -> str:
    cfg = _load_config() or {}
    return cfg.get("outputs_base", "")


# ── Download packaging helpers ────────────────────────────────────────────────

def _collect_sample_files(scene: str, sample: str, out_dir: Path, records_dir: Path | None) -> dict:
    """Return {filename: bytes} for all available download files for one sample."""
    stem = f"{scene}_SAMPLED_{sample}"
    files = {}

    # graph.json — pruned preferred, fall back to raw
    for candidate in [f"{stem}_pruned.json", f"{stem}.json"]:
        p = out_dir / candidate
        if p.exists():
            files["graph.json"] = p.read_bytes()
            break

    # qa.json — merge active + dormant + distractor
    for qa_base in [out_dir / "qa_results" / stem, out_dir / stem]:
        if qa_base.is_dir():
            combined = {"scene_id": stem, "questions": []}
            for qa_file in ["active_qa.json", "dormant_qa.json", "distractor_qa.json"]:
                qf = qa_base / qa_file
                if qf.exists():
                    try:
                        qdata = json.loads(qf.read_text())
                        for q in qdata.get("questions", []):
                            q["_source"] = qa_file.replace(".json", "")
                            combined["questions"].append(q)
                    except Exception:
                        pass
            if combined["questions"]:
                files["qa.json"] = json.dumps(combined, indent=2).encode()
            break

    if records_dir:
        sd = records_dir / scene / f"SAMPLED_{sample}"
        tp0 = sd / "Tp0p0"

        # meta.json (sample level first, then scene level)
        for meta_p in [sd / "meta.json", sd.parent / "meta.json"]:
            if meta_p.exists():
                files["meta.json"] = meta_p.read_bytes()
                break

        # state.json, calib.json
        for fname in ["state.json", "calib.json"]:
            p = tp0 / fname
            if p.exists():
                files[fname] = p.read_bytes()

        # frames.json — build from image paths
        if sd.exists():
            frames = {}
            for tdir in sorted(sd.iterdir()):
                if tdir.is_dir() and tdir.name.startswith("T"):
                    frame = {}
                    for img in sorted(list(tdir.glob("*.jpg")) + list(tdir.glob("*.png"))):
                        frame[img.stem] = str(img.resolve())
                    if frame:
                        frames[tdir.name] = frame
            if frames:
                files["frames.json"] = json.dumps(frames, indent=2).encode()

    return files


def _build_downloads(scenes: dict, out_dir: Path, records_dir: Path | None, dir_name: str) -> tuple:
    """
    Build per-scene zip files and a full dataset zip in {out_dir}/downloads/.
    Returns (scene_zip_map, full_zip_rel) where scene_zip_map is {scene: rel_path}.
    """
    downloads_dir = out_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    scene_zip_map = {}
    full_buf = io.BytesIO()

    with zipfile.ZipFile(full_buf, "w", zipfile.ZIP_DEFLATED) as full_zf:
        for scene, samples in sorted(scenes.items()):
            scene_buf = io.BytesIO()
            any_data = False
            with zipfile.ZipFile(scene_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for sample, _ in sorted(samples):
                    files = _collect_sample_files(scene, sample, out_dir, records_dir)
                    for fname, content in files.items():
                        arc_path = f"SAMPLED_{sample}/{fname}"
                        zf.writestr(arc_path, content)
                        full_zf.writestr(f"{scene}/SAMPLED_{sample}/{fname}", content)
                        any_data = True

            if any_data:
                zip_rel = f"downloads/{scene}.zip"
                (downloads_dir / f"{scene}.zip").write_bytes(scene_buf.getvalue())
                scene_zip_map[scene] = zip_rel

    full_zip_rel = f"downloads/{dir_name}_full.zip"
    (downloads_dir / f"{dir_name}_full.zip").write_bytes(full_buf.getvalue())

    return scene_zip_map, full_zip_rel


# ── Main index generation ─────────────────────────────────────────────────────

def generate_index(out_dir: Path, title: str = None, records_dir: Path = None):
    """Generate index.html for an output directory."""
    import re as _re
    _SCENE_HTML = _re.compile(r".+_SAMPLED(?:_\d+)?\.html$")
    html_files = sorted(f for f in out_dir.glob("*_refined.html") if _SCENE_HTML.match(f.name))
    if not html_files:
        html_files = sorted(f for f in out_dir.glob("*.html") if _SCENE_HTML.match(f.name))
    if not html_files:
        print(f"  No HTML files in {out_dir.name}")
        return

    if not title:
        name = out_dir.name
        title_map = {
            "nuscenes": "NuScenes — GPT-5.2",
            "nuscenes_claude": "NuScenes — Claude (old records)",
            "nuscenes_gemini": "NuScenes — Gemini (old records)",
            "nuscenes_v1": "NuScenes V1 — Claude (filtered, 16 cams + 1 BEV)",
            "nuscenes_v2": "NuScenes V2 — Claude (filtered, 4 BEVs + 4 cams)",
            "openscene": "OpenScene — GPT-5.2",
            "argoverse2": "Argoverse V2 — GPT-5.2",
        }
        title = title_map.get(name, name)

    # Load verification report if present
    flagged_samples = set()
    verification_report_path = out_dir / "verification_report.json"
    if verification_report_path.exists():
        try:
            vr = json.loads(verification_report_path.read_text())
            for entry in vr.get("flagged_scenes", []):
                flagged_samples.add(entry["scene_id"])
        except Exception:
            pass

    # Group by scene
    scenes = {}
    for f in html_files:
        stem = f.stem.replace("_refined", "").replace("_pruned", "")
        if stem.endswith("_verification"):
            continue
        parts = stem.rsplit("_SAMPLED_", 1)
        if len(parts) == 2:
            scene, sample = parts
        else:
            scene = stem.replace("_SAMPLED", "")
            sample = "0"
        scenes.setdefault(scene, []).append((sample, f.name))

    # Build download zips
    print(f"  Building download packages for {len(scenes)} scenes ...")
    scene_zip_map, full_zip_rel = _build_downloads(scenes, out_dir, records_dir, out_dir.name)
    n_zipped = len(scene_zip_map)
    print(f"  Packaged {n_zipped}/{len(scenes)} scenes → {out_dir / 'downloads'}")

    def _count_qa_types(stem: str) -> dict:
        """Return {active, dormant, distractor} QA counts for one sample."""
        out = {"active": 0, "dormant": 0, "distractor": 0}
        for qa_base in [out_dir / "qa_results" / stem, out_dir / stem]:
            if not qa_base.is_dir():
                continue
            for key, fname in [("active", "active_qa.json"),
                               ("dormant", "dormant_qa.json"),
                               ("distractor", "distractor_qa.json")]:
                qf = qa_base / fname
                if qf.exists():
                    try:
                        out[key] += len(json.loads(qf.read_text()).get("questions", []))
                    except Exception:
                        pass
            break
        return out

    def _node_count(stem: str) -> int:
        for candidate in [f"{stem}_verified.json", f"{stem}_pruned.json", f"{stem}.json"]:
            cpath = out_dir / candidate
            if cpath.exists():
                try:
                    g = json.loads(cpath.read_text())
                    if "nodes" in g:
                        return len([n for n in g["nodes"] if n.get("id") != "ego"])
                except Exception:
                    pass
        return 0

    rows = ""
    totals = {"active": 0, "dormant": 0, "distractor": 0, "nodes": 0}
    for scene_idx, (scene, samples) in enumerate(sorted(scenes.items()), 1):
        short = scene.replace("nuscenes-", "").replace("openscene-", "").replace("argoverse2-", "")

        # Aggregate counts across ALL samples in this scene
        scene_counts = {"active": 0, "dormant": 0, "distractor": 0, "nodes": 0}
        sample_has_qa = {}
        for sample, _name in sorted(samples):
            stem = f"{scene}_SAMPLED_{sample}"
            per = _count_qa_types(stem)
            for k, v in per.items():
                scene_counts[k] += v
            scene_counts["nodes"] += _node_count(stem)
            sample_has_qa[sample] = sum(per.values()) > 0
        for k, v in scene_counts.items():
            totals[k] += v

        links = "  ".join(
            f'<a class="sample-link{" has-qa" if sample_has_qa.get(s) else ""}" href="{name}">S{s}</a>'
            for s, name in sorted(samples)
        )

        def _badge(n, cls):
            return f'<span class="badge {cls}">{n}</span>' if n else '<span class="badge empty">0</span>'

        if scene in scene_zip_map:
            dl_cell = f'<a class="dl-btn" href="{scene_zip_map[scene]}" download title="Download all {len(samples)} samples">📥</a>'
        else:
            dl_cell = '<span style="color:#444">—</span>'

        rows += f"""<tr>
            <td>{scene_idx}</td>
            <td>{short}</td>
            <td>{scene_counts['nodes']}</td>
            <td>{_badge(scene_counts['active'], 'active')}</td>
            <td>{_badge(scene_counts['dormant'], 'dormant')}</td>
            <td>{_badge(scene_counts['distractor'], 'distractor')}</td>
            <td>{links}</td>
            <td>{dl_cell}</td>
        </tr>\n"""

    full_dl_btn = f'<a class="full-dl-btn" href="{full_zip_rel}" download>📥 Download {out_dir.name}_full.zip</a>'

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ background: #0a0e27; color: #e0e0e0; font-family: system-ui; margin: 20px; }}
h1 {{ color: #3498DB; margin-bottom: 4px; }}
.subtitle {{ color: #888; margin-bottom: 16px; font-size: 14px; }}
.stats {{ display: flex; gap: 20px; margin-bottom: 16px; }}
.stat {{ background: #1a1e3e; padding: 8px 16px; border-radius: 6px; }}
.stat .num {{ font-size: 20px; font-weight: 700; color: #3498DB; }}
.stat .lbl {{ font-size: 11px; color: #888; }}
table {{ border-collapse: collapse; width: 100%; max-width: 1100px; }}
th {{ background: #1a1e3e; padding: 8px 12px; text-align: left; border-bottom: 2px solid #3498DB; font-size: 13px; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #222; font-size: 13px; }}
a {{ color: #F39C12; text-decoration: none; margin-right: 6px; }}
a:hover {{ color: #fff; text-decoration: underline; }}
/* flagging handled by editor server */
tr:hover {{ background: #1a1e3e; }}
.badge {{ background: #2ecc7133; color: #2ecc71; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; }}
.badge.empty {{ background: #33333355; color: #666; }}
.badge.active {{ background: #2ecc7133; color: #2ecc71; }}
.badge.dormant {{ background: #9b59b633; color: #b57edc; }}
.badge.distractor {{ background: #3498db33; color: #5dade2; }}
.sample-link {{ padding: 1px 6px; border-radius: 3px; }}
.sample-link.has-qa {{ background: #2ecc7122; color: #2ecc71 !important; font-weight: 600; }}
.sample-link.has-qa:hover {{ background: #2ecc7155; color: #fff !important; }}
.badge.cf {{ background: #e74c3c33; color: #e74c3c; }}
.badge.ac {{ background: #f39c1233; color: #f39c12; }}
.nav {{ margin-bottom: 16px; }}
.nav a {{ color: #3498DB; margin-right: 16px; font-size: 13px; }}
.dl-btn {{
    display: inline-block; padding: 2px 8px;
    background: #238636; border: 1px solid #2ea04388;
    border-radius: 4px; color: white !important;
    font-size: 12px; margin: 0; text-decoration: none;
}}
.dl-btn:hover {{ background: #2ea043; color: white !important; text-decoration: none; }}
.full-dl-btn {{
    display: inline-block; padding: 8px 18px;
    background: #238636; border: 1px solid #2ea043;
    border-radius: 6px; color: white !important;
    font-size: 14px; font-weight: 600; text-decoration: none;
    margin-bottom: 20px;
}}
.full-dl-btn:hover {{ background: #2ea043; color: white !important; text-decoration: none; }}
</style>
</head><body>
<h1>{title}</h1>
<div class="subtitle">{len(html_files)} samples across {len(scenes)} scenes</div>

{full_dl_btn}

<div class="stats">
    <div class="stat"><div class="num">{len(scenes)}</div><div class="lbl">Scenes</div></div>
    <div class="stat"><div class="num">{len(html_files)}</div><div class="lbl">Samples</div></div>
    <div class="stat"><div class="num">{totals['nodes']}</div><div class="lbl">Nodes</div></div>
    <div class="stat"><div class="num">{totals['active']}</div><div class="lbl">Active QA</div></div>
    <div class="stat"><div class="num">{totals['dormant']}</div><div class="lbl">Dormant QA</div></div>
    <div class="stat"><div class="num">{totals['distractor']}</div><div class="lbl">Distractor QA</div></div>
</div>

<div class="nav">
    <a href="../">← All Datasets</a>
</div>

<table>
<tr><th>#</th><th>Scene</th><th title="Total causal nodes across samples (excluding ego)">Nodes</th><th title="Active causal nodes QA (Stage 3a)">Active</th><th title="Dormant causal nodes QA (Stage 3b)">Dormant</th><th title="Distractor nodes QA (Stage 3c)">Distractor</th><th>View</th><th>Download</th></tr>
{rows}
</table>
</body></html>"""

    (out_dir / "index.html").write_text(html)
    print(f"  {out_dir.name}: {len(scenes)} scenes, {len(html_files)} samples → index.html")


def main():
    # Pre-parse --dataset so config defaults are available
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"], default=None)
    _known, _ = _pre.parse_known_args()
    _cfg = _load_config(_known.dataset) if _known.dataset else {}

    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", help="Output directories")
    parser.add_argument("--all", action="store_true", help="All output dirs")
    parser.add_argument("--title", help="Custom title")
    parser.add_argument("--base", default=_default_outputs_base())
    parser.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"],
                        default=None, help="Dataset name (resolves --records from config)")
    parser.add_argument("--records", default=_cfg.get("records"),
                        help="Records directory (adds meta/state/calib/frames to downloads)")
    args = parser.parse_args()

    records_dir = Path(args.records) if args.records else None

    if args.all:
        base = Path(args.base)
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            has_html = any(d.glob("*_refined.html")) or any(
                f for f in d.glob("*.html")
                if f.name != "index.html" and "_changes" not in f.name
            )
            if not has_html:
                continue
            # Auto-resolve records from config for each dataset directory
            dir_records = records_dir  # use explicit --records if given
            if dir_records is None:
                for ds in ("nuscenes", "openscene", "argoverse2"):
                    if d.name == ds or d.name.startswith(ds + "_"):
                        ds_cfg = _load_config(ds) or {}
                        if ds_cfg.get("records"):
                            dir_records = Path(ds_cfg["records"])
                        break
            generate_index(d, records_dir=dir_records)
    elif args.dirs:
        for d in args.dirs:
            generate_index(Path(d), title=args.title, records_dir=records_dir)
    elif _known.dataset and _cfg.get("outputs"):
        generate_index(Path(_cfg["outputs"]), records_dir=records_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
