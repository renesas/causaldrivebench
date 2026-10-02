# Causal Graph Editor Server

Interactive editor for reviewing and correcting LLM-generated causal graphs. Serves pre-rendered HTML scene files with authentication, inline graph editing, and bookmarking.

## Prerequisites

```bash
pip install flask bcrypt
```

## Quick Start

### 1. Render HTML scene files (if not already done)

```bash
python scripts/rendering/render_causal_graph.py \
    --dataset nuscenes \
    --html --force --index \
    --output-dir /path/to/outputs/nuscenes_full_v2
```

### 2. Create user accounts

```bash
python scripts/rendering/editor_server.py add-user navvrat
python scripts/rendering/editor_server.py add-user naren
```

You will be prompted to set a password for each user.

### 3. Start the server

```bash
python scripts/rendering/editor_server.py serve \
    --outputs-dir /path/to/outputs/nuscenes_full_v2 \
    --edits-dir /path/to/edits \
    --port 8555
```

**Arguments:**

| Argument | Description |
|---|---|
| `--outputs-dir` | Directory containing rendered `.html`, `.json`, `.png` files and `index.html` |
| `--edits-dir` | Directory for per-user bookmarks |
| `--port` | Port to serve on (default: 8555) |
| `--host` | Host to bind to (default: 0.0.0.0) |

**Important:** The server resolves relative paths in the HTML files (`../../records/`, `../d3.v7.min.js`) by setting `BASE_DIR` to 2 levels above the outputs directory. Make sure your directory structure looks like:

```
<BASE_DIR>/
  outputs/
    nuscenes_full_v2/       <-- --outputs-dir points here
      index.html
      *.html, *.json, *.png
  records/
    nuscenes_full/
      nuscenes-scene-XXXX/  <-- camera images, BEV, state data
  d3.v7.min.js
```

### 4. Open in browser

```
http://<your-ip>:8555
```

Log in with the credentials you created in step 2.

## Features

### Edit Mode
- Toggle via **Edit Mode** button in the graph stage bar
- **Nodes tab** with two sub-tabs:
  - **Active**: All nodes in the causal graph (including ego). Edit type, category, label, state, position with autocomplete suggestions. Remove non-ego nodes (auto-moves to distractors, preserving all fields).
  - **Distractors**: Nodes removed from the graph. Promote back to active (preserves all fields).
- **Edges tab**: Add/edit/delete ego edges and inter-node edges
- **Add Node**: From scene agents (state.json) or manually (traffic control, obstacle, environment)
- **Undo/Redo** support
- **Save**: Writes `{scene_id}_edited.json` in outputs dir. No reload needed.

### How edits work
- Edits saved as `{scene_id}_edited.json` in the outputs directory
- One file per scene, overwritten on re-save
- On page load (with server), edited data replaces original graph data
- Each new edit builds on the previous edit
- Auto-enters edit mode if a saved edit exists

### Navigation
- **Prev/Next** buttons navigate sample-by-sample (order from index.html)
- **Bookmarks**: Flag scenes as "Needs Review" (red) or "Good" (green)
- **Index page**: Edited scenes highlighted orange, flagged scenes red/green (instant, no loading delay)

### JSON Viewer (bottom of scene page)
- **Causal Graph**: Original LLM output
- **Edited Graph**: Appears after saving an edit
- **Verification**: If verification data exists

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/save` | POST | Save edited graph `{scene_id, graph}` |
| `/api/edited/<scene_id>` | GET | Get edited JSON for a scene (404 if none) |
| `/api/scene_list` | GET | Ordered scene list + which have edits |
| `/api/bookmark` | POST | Toggle bookmark `{scene_id, flag}` |
| `/api/bookmarks` | GET | Get current user's bookmarks |

## Re-rendering after template changes

If you modify `scene_viewer_template.py` or `scene_viewer.css`, re-render all HTML files:

```bash
python scripts/rendering/render_causal_graph.py \
    --dataset nuscenes \
    --html --force --index \
    --output-dir /path/to/outputs/nuscenes_full_v2
```

This regenerates all HTML files with the updated template. Existing `_edited.json` files are preserved and excluded from rendering.

## Security notes

- `users.json` contains bcrypt password hashes -- do NOT commit this file
- `.flask_secret` stores the session key -- do NOT commit this file
- Both are in `.gitignore`
- Sessions persist across server restarts (secret key saved to `.flask_secret`)
- Expired sessions redirect to login gracefully

## Multi-user behavior

- **Edits are shared**: All annotators edit the same `{scene_id}_edited.json`
- **Bookmarks are per-user**: Stored in `edits/{username}/bookmarks.json`
- No merge conflicts: last save wins (single file overwrite)
