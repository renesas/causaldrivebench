#!/usr/bin/env python3
"""
Causal Graph Editor Server

Serves existing rendered HTML scene files with authentication,
graph edit saving, and per-user bookmarking/flagging.

Usage:
    python editor_server.py add-user navvrat
    python editor_server.py serve --outputs-dir /path/to/outputs --edits-dir /path/to/edits --port 8555
"""
import argparse
import getpass
import json
import secrets
import sys
from pathlib import Path

import bcrypt
from flask import (
    Flask, abort, jsonify, redirect, render_template_string, request,
    send_from_directory, session, url_for,
)

SCRIPT_DIR = Path(__file__).resolve().parent
USERS_FILE = SCRIPT_DIR / "users.json"

app = Flask(__name__)

# Set from CLI
BASE_DIR: Path = Path(".")      # Common root for all relative paths
OUTPUTS_DIR: Path = Path(".")   # Subdir with HTML scenes + index
OUTPUTS_PREFIX: str = ""        # URL prefix for outputs subdir
EDITS_DIR: Path = Path(".")


# ═══════════════════════════════════════════════════════════════════
# User management
# ═══════════════════════════════════════════════════════════════════

def load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}

def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))

def add_user(username: str):
    users = load_users()
    if username in users:
        print(f"User '{username}' already exists. Resetting password.")
    pw = getpass.getpass(f"Password for {username}: ")
    pw2 = getpass.getpass("Confirm password: ")
    if pw != pw2:
        print("Passwords do not match.")
        sys.exit(1)
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    users[username] = {"password_hash": hashed}
    save_users(users)
    print(f"User '{username}' saved.")

def verify_password(username: str, password: str) -> bool:
    users = load_users()
    entry = users.get(username)
    if not entry:
        return False
    return bcrypt.checkpw(password.encode(), entry["password_hash"].encode())


# ═══════════════════════════════════════════════════════════════════
# Auth
# ═══════════════════════════════════════════════════════════════════

def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper


LOGIN_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Graph Editor — Login</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f0f1a;color:#eee;
     display:flex;justify-content:center;align-items:center;min-height:100vh}
.box{background:#16213e;border:1px solid #333;border-radius:12px;padding:40px;width:360px}
.box h2{text-align:center;margin-bottom:24px;font-size:20px}
.box input{width:100%;padding:10px 14px;margin-bottom:14px;border-radius:6px;
           border:1px solid #444;background:#1a1a2e;color:#eee;font-size:14px}
.box input:focus{outline:none;border-color:#E67E22}
.box button{width:100%;padding:10px;border-radius:6px;border:none;background:#E67E22;
            color:#fff;font-size:14px;cursor:pointer;font-weight:bold}
.box button:hover{background:#d35400}
.err{color:#E74C3C;font-size:13px;text-align:center;margin-bottom:10px}
</style></head><body>
<div class="box"><h2>Causal Graph Editor</h2>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<form method="POST" action="/login">
<input name="username" placeholder="Username" required autofocus value="{{ request.form.get('username','') }}">
<input type="password" name="password" placeholder="Password" required>
<button type="submit">Log In</button></form></div>
</body></html>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if verify_password(username, password):
            session["username"] = username
            return redirect(request.args.get("next", "/"))
        error = "Invalid username or password"
    return render_template_string(LOGIN_PAGE, error=error, request=request)


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect("/login")


# ═══════════════════════════════════════════════════════════════════
# Serve existing HTML files — index + scene pages
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    """Redirect to the outputs subdirectory index."""
    return redirect(f"/{OUTPUTS_PREFIX}/index.html")


@app.route("/outputs/")
@app.route("/outputs")
@login_required
def outputs_index():
    """Serve the root outputs index if it exists."""
    root_index = BASE_DIR / "outputs" / "index.html"
    if root_index.exists():
        html = root_index.read_text()
        html = _inject_user_bar(html)
        return html
    return redirect("/")


@app.route("/<path:filename>")
@login_required
def serve_file(filename):
    """Serve any file from the base directory (HTML, PNG, CSS, JS, images, etc.)."""
    filepath = BASE_DIR / filename
    if not filepath.exists():
        abort(404)

    # Prevent path traversal
    try:
        filepath.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        abort(403)

    # For HTML files in the outputs dir, inject user bar + server mode + flag buttons
    if filepath.suffix == ".html" and str(filepath).startswith(str(OUTPUTS_DIR)):
        html = filepath.read_text()
        scene_id = filepath.stem
        html = _inject_user_bar(html, scene_id=scene_id)
        html = _inject_server_mode(html, scene_id=scene_id)
        return html

    # Everything else served directly (images, JS, JSON, CSS, etc.)
    return send_from_directory(str(filepath.parent), filepath.name)


FLAG_COLORS = {"needs_review": "#E74C3C", "good": "#2ECC71"}
FLAG_LABELS = {"needs_review": "Needs Review", "good": "Good"}


def _inject_user_bar(html: str, scene_id: str = "") -> str:
    """Inject top bar with auth, flags, and index badge JS."""
    username = session.get("username", "")
    is_scene = scene_id and scene_id != "index"

    # Flag buttons on scene pages
    flag_html = ""
    if is_scene:
        flag_html = (
            '<span style="display:flex;gap:6px;align-items:center">'
            '<button onclick="flagScene(\'needs_review\')" id="flag-needs_review" '
            'style="padding:5px 14px;border-radius:4px;border:1px solid #E74C3C55;'
            'background:transparent;color:#E74C3C;cursor:pointer;font-size:12px;'
            'font-weight:normal;transition:all 0.15s">Needs Review</button>'
            '<button onclick="flagScene(\'good\')" id="flag-good" '
            'style="padding:5px 14px;border-radius:4px;border:1px solid #2ECC7155;'
            'background:transparent;color:#2ECC71;cursor:pointer;font-size:12px;'
            'font-weight:normal;transition:all 0.15s">Good</button>'
            '<button onclick="flagScene(\'none\')" '
            'style="padding:5px 8px;border-radius:4px;border:1px solid #55555555;'
            'background:transparent;color:#666;cursor:pointer;font-size:12px"'
            ' title="Clear flag">&times;</button>'
            '</span>'
        )

    bar = (
        f'<div id="editor-user-bar" style="position:fixed;top:0;left:0;right:0;z-index:9999;'
        f'background:#1a1a2e;border-bottom:1px solid #333;padding:8px 20px;display:flex;'
        f'align-items:center;gap:16px;font-family:Segoe UI,Arial,sans-serif;font-size:13px">'
        f'<span style="color:#E67E22;font-weight:bold">Graph Editor</span>'
        f'<a href="/" style="color:#3498DB;text-decoration:none">Scenes</a>'
        f'<a href="/bookmarks" style="color:#3498DB;text-decoration:none">Bookmarks</a>'
        f'{flag_html}'
        f'<span style="margin-left:auto;color:#888">{username}</span>'
        f'<a href="/logout" style="color:#888;text-decoration:none">Logout</a>'
        f'</div>'
        f'<div style="height:42px"></div>'
    )

    inject_js = ""
    if is_scene:
        inject_js = f"""<script>
const _SCENE_ID = {json.dumps(scene_id)};
function flagScene(flag) {{
  fetch("/api/bookmark", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{scene_id: _SCENE_ID, flag: flag}})
  }}).then(r => r.json()).then(() => {{
    ["needs_review","good"].forEach(k => {{
      const b = document.getElementById("flag-" + k);
      if (!b) return;
      const c = k === "needs_review" ? "#E74C3C" : "#2ECC71";
      if (k === flag) {{
        b.style.background = c + "33";
        b.style.fontWeight = "bold";
        b.style.borderColor = c;
      }} else {{
        b.style.background = "transparent";
        b.style.fontWeight = "normal";
        b.style.borderColor = c + "55";
      }}
    }});
  }});
}}
(function() {{
  const bm = {json.dumps(_load_bookmarks(username))};
  const info = bm[_SCENE_ID];
  if (info && info.flag) flagScene(info.flag);
}})();
</script>"""
    else:
        # Index page: inject bookmark + edited data server-side for instant highlights
        username = session.get("username", "")
        bm_data = json.dumps(_load_bookmarks(username)) if username else "{}"
        edited_list = [s for s in _build_scene_list()
                       if (OUTPUTS_DIR / f"{s}_{username}_edited.json").exists()]
        edited_data = json.dumps(edited_list)
        inject_js = f"""<script>
(function() {{
  const bm = {bm_data};
  const editedSet = new Set({edited_data});
  const flagColors = {{"needs_review": "#E74C3C", "good": "#2ECC71"}};
  document.querySelectorAll('a[href$=".html"]').forEach(a => {{
    const sid = a.getAttribute("href").replace(/\\.html$/, "");
    const info = bm[sid];
    if (info && info.flag && flagColors[info.flag]) {{
      a.style.background = flagColors[info.flag] + "33";
      a.style.color = flagColors[info.flag];
      a.style.padding = "1px 6px";
      a.style.borderRadius = "3px";
      a.style.fontWeight = "bold";
    }} else if (editedSet.has(sid)) {{
      a.style.background = "#F1C40F33";
      a.style.color = "#F1C40F";
      a.style.padding = "1px 6px";
      a.style.borderRadius = "3px";
      a.style.fontWeight = "bold";
    }}
  }});
}})();
</script>"""

    html = html.replace("<body>", f"<body>{bar}", 1)
    html = html.replace("</body>", f"{inject_js}</body>", 1)
    return html


def _replace_js_var(html: str, var_name: str, new_json: str) -> str:
    """Replace a JS variable assignment like `const/let varName = {...};` with new JSON."""
    import re
    pattern = re.compile(r'(?:const|let|var)\s+' + re.escape(var_name) + r'\s*=\s*')
    m = pattern.search(html)
    if not m:
        return html
    # Use json.JSONDecoder to find exact end of the JSON object
    start = html.index('{', m.end() - 1)
    decoder = json.JSONDecoder()
    try:
        _, end_idx = decoder.raw_decode(html, start)
        return html[:start] + new_json + html[end_idx:]
    except json.JSONDecodeError:
        return html  # don't break the HTML if parsing fails


def _inject_server_mode(html: str, scene_id: str = "") -> str:
    """Inject SERVER_MODE, nav, edited data (server-side), and overrides into scene HTML."""
    username = session.get("username", "")

    # Build prev/next from index order
    scenes = _build_scene_list()
    prev_id = next_id = ""
    if scene_id in scenes:
        idx = scenes.index(scene_id)
        if idx > 0:
            prev_id = scenes[idx - 1]
        if idx < len(scenes) - 1:
            next_id = scenes[idx + 1]

    inject = (
        f"<script>"
        f"window.SERVER_MODE = true;"
        f"window.CURRENT_USER = '{username}';"
        f"window.SCENE_ID_FROM_SERVER = document.title.replace('Scene: ','');"
        f"window.PREV_SCENE = '{prev_id}';"
        f"window.NEXT_SCENE = '{next_id}';"
        f"window.OUTPUTS_PREFIX = '/{OUTPUTS_PREFIX}';"
        f"</script>"
    )
    html = html.replace("</head>", f"{inject}</head>", 1)

    # ── Server-side graph data replacement (no flash, no async) ──
    edited_path = OUTPUTS_DIR / f"{scene_id}_{username}_edited.json" if scene_id else None
    has_edit = edited_path and edited_path.exists()
    if has_edit:
        edited_json = edited_path.read_text().strip()
        html = _replace_js_var(html, 'graphData', edited_json)
        html = _replace_js_var(html, 'prunedGraphData', edited_json)
        # Inject "Edited Graph" tab in JSON viewer
        html = html.replace(
            'onclick="switchJsonTab(\'verification\'',
            'data-tab="edited" onclick="switchJsonTab(\'edited\', this)">Edited Graph</button>'
            '<button class="json-tab" onclick="switchJsonTab(\'verification\'',
            1
        )
        # Store editedGraphData for JS access
        edited_js_var = f"<script>window.__EDITED_DATA__ = {edited_json};</script>"
        html = html.replace("</head>", f"{edited_js_var}</head>", 1)

    # Inject nav buttons + overrides before </body>
    nav_js = """<script>
(function() {
  // ── Prev / Next nav buttons ──
  const bar = document.getElementById('editor-user-bar');
  if (bar) {
    const navSpan = document.createElement('span');
    navSpan.style.cssText = 'display:flex;gap:6px;align-items:center;margin-left:8px';
    if (window.PREV_SCENE) {
      const pb = document.createElement('a');
      pb.href = window.OUTPUTS_PREFIX + '/' + window.PREV_SCENE + '.html';
      pb.textContent = '\\u25C0 Prev';
      pb.style.cssText = 'color:#3498DB;text-decoration:none;padding:4px 10px;border:1px solid #3498DB44;border-radius:4px;font-size:12px';
      pb.onmouseover = () => pb.style.background = '#3498DB22';
      pb.onmouseout = () => pb.style.background = 'transparent';
      navSpan.appendChild(pb);
    }
    if (window.NEXT_SCENE) {
      const nb = document.createElement('a');
      nb.href = window.OUTPUTS_PREFIX + '/' + window.NEXT_SCENE + '.html';
      nb.textContent = 'Next \\u25B6';
      nb.style.cssText = 'color:#3498DB;text-decoration:none;padding:4px 10px;border:1px solid #3498DB44;border-radius:4px;font-size:12px';
      nb.onmouseover = () => nb.style.background = '#3498DB22';
      nb.onmouseout = () => nb.style.background = 'transparent';
      navSpan.appendChild(nb);
    }
    // Insert after the Bookmarks link
    const bookmarkLink = bar.querySelector('a[href="/bookmarks"]');
    if (bookmarkLink) bookmarkLink.after(navSpan);
    else bar.insertBefore(navSpan, bar.querySelector('span[style*="margin-left:auto"]'));
  }

  // ── Autocomplete options for node fields ──
  window._LABEL_OPTIONS = {
    OTHER_AGENT: ['car','truck','bus','motorcycle','bicycle','pedestrian','trailer','construction vehicle','police officer','construction worker'],
    TRAFFIC_CONTROL: ['red traffic light','green traffic light','yellow traffic light','pedestrian signal','stop sign','yield sign','speed limit sign','crosswalk marking','stop line marking'],
    ROAD_OBSTACLE: ['traffic cone','debris','pothole','standing water','barrier','construction zone'],
    ENVIRONMENT: ['wet road surface','fog','heavy rain','nighttime','snow','glare']
  };
  window._STATE_OPTIONS = {
    OTHER_AGENT: ['stationary','moving slowly','moderate speed','high speed','decelerating','accelerating','stopped at signal','parked','turning left','turning right','changing lanes','crossing road','waiting to cross'],
    TRAFFIC_CONTROL: ['red','green','yellow','flashing','active','visible','partially occluded','unoccupied','occupied'],
    ROAD_OBSTACLE: ['stationary','partially blocking lane','fully blocking lane','on road edge','in ego path'],
    ENVIRONMENT: ['active','worsening','improving','mild','severe','intermittent']
  };
  window._POSITION_OPTIONS = [
    'ahead same lane','ahead left lane','ahead right lane','ahead-left','ahead-right',
    'behind same lane','behind-left','behind-right',
    'left adjacent lane','right adjacent lane','far left','far right',
    'oncoming lane','at intersection','in crosswalk','on sidewalk','on road shoulder'
  ];
  window._VISIBLE_OPTIONS = [
    'CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK',
    'CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_F0','CAM_L0','CAM_R0','CAM_B0',
    'BEV','STATE','CAM_FRONT (all timesteps)','CAM_FRONT, BEV',
    'all cameras, BEV','all cameras, BEV, STATE'
  ];

  // ── Custom autocomplete dropdown ──
  (function() {
    const s = document.createElement('style');
    s.textContent = '.ac-wrap{position:relative;display:block}'
      +'.ac-drop{display:none;position:absolute;top:100%;left:0;right:0;z-index:100;'
      +'background:#1a1a2e;border:1px solid #444;border-radius:4px;max-height:110px;'
      +'overflow-y:auto;margin-top:2px;box-shadow:0 4px 12px rgba(0,0,0,.5)}'
      +'.ac-drop.open{display:block}'
      +'.ac-opt{padding:3px 8px;font-size:11px;color:#aaa;cursor:pointer;border-bottom:1px solid #262640}'
      +'.ac-opt:last-child{border-bottom:none}'
      +'.ac-opt:hover,.ac-opt.sel{background:#2a2a4a;color:#eee}';
    document.head.appendChild(s);

    function attachAC(inp, opts) {
      if (!inp || inp._ac) return;
      inp._ac = true;
      const wrap = document.createElement('span');
      wrap.className = 'ac-wrap';
      inp.parentNode.insertBefore(wrap, inp);
      wrap.appendChild(inp);
      const drop = document.createElement('div');
      drop.className = 'ac-drop';
      wrap.appendChild(drop);
      let si = -1;
      function render() {
        const q = inp.value.toLowerCase();
        const f = q ? opts.filter(o => o.toLowerCase().includes(q)) : opts;
        drop.innerHTML = ''; si = -1;
        f.slice(0, 8).forEach(o => {
          const d = document.createElement('div');
          d.className = 'ac-opt'; d.textContent = o;
          d.onmousedown = e => { e.preventDefault(); inp.value = o; drop.classList.remove('open'); };
          drop.appendChild(d);
        });
        drop.classList.toggle('open', f.length > 0);
      }
      inp.addEventListener('focus', render);
      inp.addEventListener('input', render);
      inp.addEventListener('blur', () => setTimeout(() => drop.classList.remove('open'), 120));
      inp.addEventListener('keydown', e => {
        const items = drop.querySelectorAll('.ac-opt');
        if (!items.length) return;
        if (e.key === 'ArrowDown') { e.preventDefault(); si = Math.min(si + 1, items.length - 1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); si = Math.max(si - 1, 0); }
        else if (e.key === 'Enter' && si >= 0) { e.preventDefault(); inp.value = items[si].textContent; drop.classList.remove('open'); return; }
        else if (e.key === 'Escape') { drop.classList.remove('open'); return; }
        else return;
        items.forEach((it, i) => it.classList.toggle('sel', i === si));
        if (items[si]) items[si].scrollIntoView({ block: 'nearest' });
      });
    }

    new MutationObserver(function(muts) {
      muts.forEach(m => m.addedNodes.forEach(n => {
        if (!n.querySelectorAll) return;
        n.querySelectorAll('input[id*="-state"]').forEach(inp => {
          const sel = (inp.closest('.edit-form') || document).querySelector('select[id*="-type"]');
          attachAC(inp, window._STATE_OPTIONS[(sel && sel.value) || 'OTHER_AGENT'] || []);
        });
        n.querySelectorAll('input[id*="-pos"], input[id*="-position"]').forEach(inp => {
          attachAC(inp, window._POSITION_OPTIONS);
        });
        n.querySelectorAll('input[id*="-label"]').forEach(inp => {
          const sel = (inp.closest('.edit-form') || document).querySelector('select[id*="-type"]');
          attachAC(inp, window._LABEL_OPTIONS[(sel && sel.value) || 'OTHER_AGENT'] || []);
        });
        // Multi-select checkboxes for visible_in
        n.querySelectorAll('[data-vis-multi]').forEach(wrap => {
          if (wrap._visDone) return;
          wrap._visDone = true;
          const input = wrap.querySelector('input');
          if (!input) return;
          const current = (input.value || '').split(',').map(s => s.trim()).filter(Boolean);
          const cams = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','BEV'];
          const box = document.createElement('div');
          box.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;margin-top:4px';
          cams.forEach(c => {
            const lbl = document.createElement('label');
            lbl.style.cssText = 'font-size:10px;color:#aaa;display:flex;align-items:center;gap:2px;cursor:pointer';
            const cb = document.createElement('input');
            cb.type = 'checkbox'; cb.value = c;
            cb.checked = current.includes(c);
            cb.style.cssText = 'width:12px;height:12px;accent-color:#3498DB';
            cb.onchange = () => {
              const checked = box.querySelectorAll('input:checked');
              input.value = Array.from(checked).map(x => x.value).join(', ');
            };
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(c.replace('CAM_','')));
            box.appendChild(lbl);
          });
          wrap.appendChild(box);
        });
      }));
    }).observe(document.getElementById('edit-panel') || document.body, { childList: true, subtree: true });
  })();

  // ── Inject sub-tab CSS ──
  (function() {
    const st = document.createElement('style');
    st.textContent = '.edit-subtabs{display:flex;gap:0;border-bottom:1px solid #333;margin:6px 0 0}'
      +'.edit-subtab{padding:5px 14px;font-size:11px;color:#888;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}'
      +'.edit-subtab:hover{color:#ccc}.edit-subtab.active{color:#eee;border-bottom-color:#3498DB}'
      +'.node-subtab-content{display:none}.node-subtab-content.active{display:block}';
    document.head.appendChild(st);
  })();

  // ── Override renderNodeList with Active/Distractors sub-tabs ──
  window.switchNodeSubTab = function(tab, el) {
    document.querySelectorAll('.edit-subtab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    document.querySelectorAll('.node-subtab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('node-subtab-' + tab).classList.add('active');
  };

  window.renderNodeList = function() {
    const el = document.getElementById('edit-nodes');
    if (!el || typeof editableGraph === 'undefined' || !editableGraph) return;
    const nodes = editableGraph.nodes || [];
    const distractors = editableGraph.distractors || [];
    const allNodes = [];
    nodes.forEach((n, i) => { allNodes.push({node:n, idx:i}); });

    let html = '';
    // Sub-tabs
    html += '<div class="edit-subtabs">'
      +'<div class="edit-subtab active" onclick="switchNodeSubTab(\\\'active\\\',this)">Active/Dormant ('+allNodes.length+')</div>'
      +'<div class="edit-subtab" onclick="switchNodeSubTab(\\\'distractor\\\',this)">Distractors ('+distractors.length+')</div>'
      +'</div>';
    // Active
    html += '<div id="node-subtab-active" class="node-subtab-content active"><div class="edit-scroll-section">';
    allNodes.forEach(item => {
      const isEgo = item.node.id === 'ego';
      html += '<div class="edit-row" data-node-idx="'+item.idx+'">'
        +'<span class="row-id">'+item.node.id+'</span>'
        +'<span class="row-type">'+(item.node.type||'')+'</span>'
        +'<span class="row-label">'+(item.node.label||item.node.category||'')+'</span>'
        +'<span class="row-actions">'
        +'<button onclick="showEditNodeForm('+item.idx+')">Edit</button>'
        +(isEgo ? '' : '<button class="danger" onclick="removeNode('+item.idx+')">Remove</button>')
        +'</span></div>';
    });
    if (!allNodes.length) html += '<div style="padding:6px 12px;color:#666;font-size:12px">None</div>';
    html += '</div></div>';
    // Distractors
    html += '<div id="node-subtab-distractor" class="node-subtab-content"><div class="edit-scroll-section">';
    distractors.forEach((d, i) => {
      html += '<div class="edit-row">'
        +'<span class="row-id">'+d.id+'</span>'
        +'<span class="row-type">'+(d.type||'')+'</span>'
        +'<span class="row-label">'+(d.label||d.category||d.id)+'</span>'
        +'<span class="row-actions"><button class="promote" onclick="promoteDistractor('+i+')">Promote</button><button class="danger" onclick="deleteDistractor('+i+')">Delete</button></span>'
        +'</div>';
    });
    if (!distractors.length) html += '<div style="padding:6px 12px;color:#666;font-size:12px">None</div>';
    html += '</div></div>';
    html += '<button class="edit-add-btn" onclick="showAddNodeForm()">+ Add Node</button>';
    html += '<div id="node-form-container"></div>';
    el.innerHTML = html;
  };

  // ── Override renderEditPanel + refreshAll to use our renderNodeList ──
  window.renderEditPanel = function() {
    window.renderNodeList();
    if (typeof renderEdgeList === 'function') renderEdgeList();
  };
  window.refreshAll = function() {
    if (typeof rebuildGraph === 'function') rebuildGraph(editableGraph);
    window.renderEditPanel();
    if (typeof updateUndoRedo === 'function') updateUndoRedo();
  };

  // ── Override removeNode to skip prompt popup ──
  window.removeNode = function(idx) {
    if (typeof editableGraph === 'undefined' || !editableGraph) return;
    const n = editableGraph.nodes[idx];
    if (!n || n.id === 'ego') return;
    editableGraph.distractors = editableGraph.distractors || [];
    const dist = Object.assign({}, n);
    dist.rejection_reason = dist.rejection_reason || 'Removed by annotator';
    editableGraph.distractors.push(dist);
    editableGraph.nodes.splice(idx, 1);
    editableGraph.ego_edges = (editableGraph.ego_edges || []).filter(e => e.source !== n.id && e.target !== n.id);
    editableGraph.inter_node_edges = (editableGraph.inter_node_edges || []).filter(e => e.source !== n.id && e.target !== n.id);
    if (typeof autoDistractOrphans === 'function') autoDistractOrphans();
    if (typeof pushEditState === 'function') pushEditState();
    if (typeof refreshAll === 'function') refreshAll();
  };

  // ── Override removeEgoEdge / removeInterEdge to auto-distract orphans ──
  window.removeEgoEdge = function(idx) {
    if (typeof editableGraph === 'undefined' || !editableGraph) return;
    editableGraph.ego_edges.splice(idx, 1);
    if (typeof autoDistractOrphans === 'function') autoDistractOrphans();
    if (typeof pushEditState === 'function') pushEditState();
    if (typeof refreshAll === 'function') refreshAll();
  };
  window.removeInterEdge = function(idx) {
    if (typeof editableGraph === 'undefined' || !editableGraph) return;
    editableGraph.inter_node_edges.splice(idx, 1);
    if (typeof autoDistractOrphans === 'function') autoDistractOrphans();
    if (typeof pushEditState === 'function') pushEditState();
    if (typeof refreshAll === 'function') refreshAll();
  };

  // ── Override promoteDistractor to preserve all fields ──
  window.promoteDistractor = function(idx) {
    if (typeof editableGraph === 'undefined' || !editableGraph) return;
    const d = editableGraph.distractors[idx];
    if (!d) return;
    const node = Object.assign({}, d);
    node.type = node.type || 'OTHER_AGENT';
    node.category = node.category || '';
    node.label = node.label || node.id.replace('_', ' ');
    node.state = node.state || '';
    node.position_relative_to_ego = node.position_relative_to_ego || '';
    node.visible_in = node.visible_in || '';
    node.motion = node.motion || {};
    delete node.rejection_reason;
    editableGraph.nodes.push(node);
    editableGraph.distractors.splice(idx, 1);
    if (typeof pushEditState === 'function') pushEditState();
    if (typeof refreshAll === 'function') refreshAll();
    const tab = document.querySelector('.edit-tab');
    if (tab && typeof switchEditTab === 'function') switchEditTab('nodes', tab);
  };
  window.deleteDistractor = function(idx) {
    if (typeof editableGraph === 'undefined' || !editableGraph || !editableGraph.distractors) return;
    editableGraph.distractors.splice(idx, 1);
    if (typeof pushEditState === 'function') pushEditState();
    if (typeof refreshAll === 'function') refreshAll();
  };

  // ── Override saveEditedGraph to also show Edited Graph tab + update editedGraphData ──
  const _origSave = window.saveEditedGraph;
  window.saveEditedGraph = function() {
    if (typeof editableGraph === 'undefined' || !editableGraph) return;
    const output = JSON.parse(JSON.stringify(editableGraph));
    output.nodes = (output.nodes || []).map(n => {
      const clean = {};
      ['id','type','category','label','state','position_relative_to_ego','visible_in','motion'].forEach(k => { if (n[k] !== undefined) clean[k] = n[k]; });
      return clean;
    });

    if (window.SERVER_MODE) {
      const saveBtn = document.getElementById('btn-save');
      saveBtn.textContent = 'Saving...';
      saveBtn.disabled = true;
      const SCENE_ID = window.SCENE_ID_FROM_SERVER || document.title.replace('Scene: ','');
      fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene_id: SCENE_ID, graph: output })
      }).then(r => {
        if (r.redirected || r.status === 401 || r.status === 302) {
          window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
          throw new Error('Session expired — redirecting to login');
        }
        if (!r.ok) throw new Error('Save failed: ' + r.status);
        return r.json();
      }).then(data => {
        saveBtn.textContent = 'Saved!';
        saveBtn.style.borderColor = '#2ECC71';
        saveBtn.style.color = '#2ECC71';
        // Update editedGraphData
        window.editedGraphData = JSON.parse(JSON.stringify(output));
        if (typeof editedGraphData !== 'undefined') editedGraphData = window.editedGraphData;
        // Update graph data in-place
        Object.keys(output).forEach(k => { if (k !== '_usage' && typeof graphData !== 'undefined') graphData[k] = output[k]; });
        if (typeof prunedGraphData !== 'undefined' && prunedGraphData) {
          Object.keys(output).forEach(k => { if (k !== '_usage') prunedGraphData[k] = output[k]; });
        }
        // Ensure Edited Graph tab exists in JSON viewer
        const jsonTabs = document.querySelector('.json-tabs');
        if (jsonTabs && !jsonTabs.querySelector('[data-tab="edited"]')) {
          const btn = document.createElement('button');
          btn.className = 'json-tab';
          btn.setAttribute('data-tab', 'edited');
          btn.textContent = 'Edited Graph';
          btn.onclick = function() { switchJsonTab('edited', btn); };
          const verTab = jsonTabs.querySelector('[onclick*="verification"]') || jsonTabs.lastElementChild;
          jsonTabs.insertBefore(btn, verTab);
        }
        setTimeout(() => {
          saveBtn.textContent = 'Save JSON';
          saveBtn.disabled = false;
          saveBtn.style.borderColor = '';
          saveBtn.style.color = '';
        }, 2000);
      }).catch(err => {
        alert('Save failed: ' + err.message);
        saveBtn.textContent = 'Save JSON';
        saveBtn.disabled = false;
      });
    }
  };

  // ── Override switchJsonTab to handle 'edited' tab on load ──
  if (typeof switchJsonTab === 'function') {
    const _origSwitchTab = switchJsonTab;
    window.switchJsonTab = function(tab, b) {
      if (tab === 'edited' && window.editedGraphData) {
        if (typeof activeJsonTab !== 'undefined') activeJsonTab = tab;
        document.querySelectorAll('.json-tab').forEach(x => x.classList.remove('active'));
        if (b) b.classList.add('active');
        document.getElementById('json-viewer').innerHTML = (typeof syntaxHighlight === 'function')
          ? syntaxHighlight(JSON.stringify(window.editedGraphData, null, 2))
          : '<pre>' + JSON.stringify(window.editedGraphData, null, 2) + '</pre>';
        return;
      }
      _origSwitchTab(tab, b);
    };
  }

  // ── Auto-enter edit mode if edited data was loaded server-side ──
  if (window.__EDITED_DATA__) {
    if (typeof editedGraphData !== 'undefined') editedGraphData = window.__EDITED_DATA__;
    if (typeof enterEditMode === 'function') enterEditMode();
  }
})();
</script>"""
    html = html.replace("</body>", f"{nav_js}</body>", 1)
    return html


import re as _re
_SAFE_ID = _re.compile(r'^[a-zA-Z0-9_\-]+$')


def _validate_id(value: str) -> bool:
    """Check that a scene_id or username is safe for use in file paths."""
    return bool(value) and bool(_SAFE_ID.match(value)) and '..' not in value


# ═══════════════════════════════════════════════════════════════════
# API: Save edited graph
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/save", methods=["POST"])
@login_required
def save_graph():
    data = request.get_json()
    if not data or "scene_id" not in data or "graph" not in data:
        return jsonify({"error": "Missing scene_id or graph"}), 400

    username = session["username"]
    scene_id = data["scene_id"]
    if not _validate_id(scene_id) or not _validate_id(username):
        return jsonify({"error": "Invalid scene_id or username"}), 400
    out_path = OUTPUTS_DIR / f"{scene_id}_{username}_edited.json"
    out_path.write_text(json.dumps(data["graph"], indent=2))
    return jsonify({"status": "ok", "path": str(out_path)})


@app.route("/api/edited/<scene_id>", methods=["GET"])
@login_required
def get_edited(scene_id):
    """Return edited JSON for current user's scene, or 404 if none exists."""
    username = session["username"]
    if not _validate_id(scene_id) or not _validate_id(username):
        abort(400)
    edited_path = OUTPUTS_DIR / f"{scene_id}_{username}_edited.json"
    if not edited_path.exists():
        return jsonify(None), 404
    return jsonify(json.loads(edited_path.read_text()))


def _build_scene_list():
    """Parse index.html to get ordered list of scene IDs (preserves index order)."""
    import re
    index_path = OUTPUTS_DIR / "index.html"
    if not index_path.exists():
        return []
    html = index_path.read_text()
    hrefs = re.findall(r'href="([^"]+\.html)"', html)
    return [h.replace(".html", "") for h in hrefs
            if h != "index.html" and not h.startswith("verification")]


@app.route("/api/scene_list", methods=["GET"])
@login_required
def scene_list():
    """Return ordered list of all scene IDs + which ones have edits."""
    scenes = _build_scene_list()
    username = session["username"]
    edited = [s for s in scenes if (OUTPUTS_DIR / f"{s}_{username}_edited.json").exists()]
    return jsonify({"scenes": scenes, "edited": edited})


# ═══════════════════════════════════════════════════════════════════
# API: Bookmarks / Flags
# ═══════════════════════════════════════════════════════════════════

def _bookmarks_path(username: str) -> Path:
    p = EDITS_DIR / username / "bookmarks.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _load_bookmarks(username: str) -> dict:
    p = _bookmarks_path(username)
    if p.exists():
        return json.loads(p.read_text())
    return {}

def _save_bookmarks(username: str, bookmarks: dict):
    _bookmarks_path(username).write_text(json.dumps(bookmarks, indent=2))


@app.route("/api/bookmark", methods=["POST"])
@login_required
def toggle_bookmark():
    """Toggle bookmark for a scene. Body: {"scene_id": "...", "flag": "review|done|issue|none"}"""
    data = request.get_json()
    if not data or "scene_id" not in data:
        return jsonify({"error": "Missing scene_id"}), 400

    username = session["username"]
    scene_id = data["scene_id"]
    if not _validate_id(scene_id):
        return jsonify({"error": "Invalid scene_id"}), 400
    flag = data.get("flag", "review")  # review, done, issue, none
    bookmarks = _load_bookmarks(username)

    if flag == "none":
        bookmarks.pop(scene_id, None)
    else:
        bookmarks[scene_id] = {"flag": flag, "note": data.get("note", "")}

    _save_bookmarks(username, bookmarks)
    return jsonify({"status": "ok", "bookmarks": bookmarks})


@app.route("/api/bookmarks", methods=["GET"])
@login_required
def get_bookmarks():
    return jsonify(_load_bookmarks(session["username"]))


@app.route("/bookmarks")
@login_required
def bookmarks_page():
    username = session["username"]
    bookmarks = _load_bookmarks(username)
    return render_template_string(BOOKMARKS_PAGE, bookmarks=bookmarks, username=username)


BOOKMARKS_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Bookmarks</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f0f1a;color:#eee;padding:20px}
.header{display:flex;align-items:center;justify-content:space-between;max-width:1000px;margin:0 auto 20px}
.header h1{font-size:22px}
.header a{color:#E67E22;text-decoration:none;margin-left:12px}
.header .user{color:#888;font-size:13px}
table{max-width:1000px;margin:0 auto;width:100%;border-collapse:collapse}
th{text-align:left;padding:10px 12px;border-bottom:2px solid #333;color:#888;font-size:12px;text-transform:uppercase}
td{padding:8px 12px;border-bottom:1px solid #222;font-size:13px}
tr:hover{background:#ffffff08}
a.link{color:#3498DB;text-decoration:none}
a.link:hover{text-decoration:underline}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px}
.badge-needs_review{background:#E67E2222;color:#E67E22}
.badge-good{background:#2ECC7122;color:#2ECC71}
.note{color:#888;font-size:12px}
.empty{text-align:center;color:#666;padding:40px}
</style></head><body>
<div class="header">
    <h1>Bookmarks</h1>
    <div class="user"><a href="/">Back to Scenes</a> | {{ username }} | <a href="/logout">Logout</a></div>
</div>
{% if bookmarks %}
<table>
<thead><tr><th>Scene</th><th>Flag</th><th>Note</th></tr></thead>
<tbody>
{% for scene_id, info in bookmarks.items() %}
<tr>
    <td><a class="link" href="/{{ scene_id }}.html">{{ scene_id }}</a></td>
    <td><span class="badge badge-{{ info.flag }}">{{ info.flag }}</span></td>
    <td class="note">{{ info.get('note', '') | e }}</td>
</tr>
{% endfor %}
</tbody></table>
{% else %}
<div class="empty">No bookmarks yet. Flag scenes from the scene viewer.</div>
{% endif %}
</body></html>"""


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Causal Graph Editor Server")
    sub = parser.add_subparsers(dest="command")

    add_cmd = sub.add_parser("add-user", help="Add or update an annotator account")
    add_cmd.add_argument("username")

    serve_cmd = sub.add_parser("serve", help="Start the editor server")
    serve_cmd.add_argument("--outputs-dir", required=True,
                           help="Directory with rendered HTML files + index.html")
    serve_cmd.add_argument("--edits-dir", required=True,
                           help="Directory to store edits and bookmarks per user")
    serve_cmd.add_argument("--port", type=int, default=8555)
    serve_cmd.add_argument("--host", default="0.0.0.0")
    serve_cmd.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.command == "add-user":
        add_user(args.username)
    elif args.command == "serve":
        global BASE_DIR, OUTPUTS_DIR, OUTPUTS_PREFIX, EDITS_DIR
        OUTPUTS_DIR = Path(args.outputs_dir).resolve()
        EDITS_DIR = Path(args.edits_dir).resolve()
        EDITS_DIR.mkdir(parents=True, exist_ok=True)

        if not OUTPUTS_DIR.exists():
            print(f"Outputs directory not found: {OUTPUTS_DIR}")
            sys.exit(1)

        # Base dir = 2 levels up from outputs (to resolve ../../records/... paths)
        BASE_DIR = OUTPUTS_DIR.parent.parent
        OUTPUTS_PREFIX = f"{OUTPUTS_DIR.parent.name}/{OUTPUTS_DIR.name}"

        if not (OUTPUTS_DIR / "index.html").exists():
            print(f"Warning: No index.html in {OUTPUTS_DIR}")

        users = load_users()
        if not users:
            print(f"No users. Run: python {__file__} add-user <username>")
            sys.exit(1)

        # Persist secret key so sessions survive server restarts
        secret_file = SCRIPT_DIR / ".flask_secret"
        if secret_file.exists():
            app.secret_key = secret_file.read_text().strip()
        else:
            app.secret_key = secrets.token_hex(32)
            secret_file.write_text(app.secret_key)
            secret_file.chmod(0o600)
        print(f"Editor server on {args.host}:{args.port}")
        print(f"  Base:    {BASE_DIR}")
        print(f"  Outputs: {OUTPUTS_DIR}")
        print(f"  Edits:   {EDITS_DIR}")
        print(f"  Users:   {', '.join(users.keys())}")
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
