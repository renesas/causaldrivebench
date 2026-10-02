"""
HTML template for the per-scene causal graph viewer.

All rendering logic lives in render_causal_graph.py.
This module only owns the markup/script structure.
"""


def render_html(
    *,
    display_title,
    d3_script_tag,
    model_color,
    scene_id,
    model_label,
    n_nodes,
    n_ego_edges,
    n_inter_edges,
    n_cols,
    qa_stats_html,
    bev_temporal_html,
    scene_view_cols,
    img_bev,
    bev_temporal,
    cam_grid_cols,
    cam_grid_rows,
    cam_grid_html,
    stage1_label,
    pruned_button_attrs,
    pruned_label,
    edited_label,
    edited_button_attrs,
    has_edited,
    edited_graph_json,
    fixed_label,
    fixed_button_attrs,
    has_fixed,
    fixed_graph_json,
    verified_label,
    verified_button_attrs,
    has_verified,
    verified_graph_json,
    changes_summary_text,
    qa_body,
    graph_data_json,
    raw_graph_json,
    pruned_graph_json,
    changes_json,
    scene_record_json,
    json_for_js,
    meta_json_for_js,
    frames_json_for_js,
    calib_json_for_js,
    state_json_for_js,
    graph_json_for_js,
    qa_result_json,
    qa_js,
    verification_json,
    read_css,
):
    bev_section = (
        f'<div class="bev-temporal">'
        + "".join(
            f'<div class="bev-temporal-cell">'
            f'<img src="{url}" alt="{lbl}">'
            f'<div class="bev-temporal-label{" t0" if "T=0" in lbl else ""}">{lbl}</div>'
            f"</div>"
            for lbl, url in bev_temporal
        )
        + "</div>"
        if len(bev_temporal) > 1
        else ""
    )

    bev_panel = (
        "<!-- BEV shown in temporal strip above -->"
        if len(bev_temporal) > 1
        else f'<div class="bev-panel" style="position:relative"><img src="{img_bev}" alt="BEV"><div class="bev-title">BEV @ T=0</div></div>'
    )

    model_badge = f'<span class="model-badge">{model_label}</span>' if model_label else ""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Scene: {display_title}</title>
    {d3_script_tag}
    <style>{read_css("scene_viewer.css")}{read_css("qa_section.css")}</style>
    <style>:root {{ --model-color: {model_color}; --model-color-bg: {model_color}22; --model-color-border: {model_color}55; }}</style>
</head>
<body>
    <h1>{scene_id} {model_badge}</h1>
    <div style="text-align: center; margin-bottom: 20px;">
        <button onclick="downloadFile('meta.json', metaData)" style="padding:8px 14px;margin:0 4px;background:#238636;color:white;border:1px solid #2ea043;border-radius:6px;cursor:pointer;font-size:13px;">&#128229; meta.json</button>
        <button onclick="downloadFile('frames.json', framesData)" style="padding:8px 14px;margin:0 4px;background:#238636;color:white;border:1px solid #2ea043;border-radius:6px;cursor:pointer;font-size:13px;">&#128229; frames.json</button>
        <button onclick="downloadFile('calib.json', calibData)" style="padding:8px 14px;margin:0 4px;background:#238636;color:white;border:1px solid #2ea043;border-radius:6px;cursor:pointer;font-size:13px;">&#128229; calib.json</button>
        <button onclick="downloadFile('state.json', stateData)" style="padding:8px 14px;margin:0 4px;background:#238636;color:white;border:1px solid #2ea043;border-radius:6px;cursor:pointer;font-size:13px;">&#128229; state.json</button>
        <button onclick="downloadFile('graph.json', prunedGraphData || graphData)" style="padding:8px 14px;margin:0 4px;background:#1f6feb;color:white;border:1px solid #388bfd;border-radius:6px;cursor:pointer;font-size:13px;">&#128229; graph.json</button>
        <button onclick="downloadFile('qa.json', qaData || {{}})" style="padding:8px 14px;margin:0 4px;background:#1f6feb;color:white;border:1px solid #388bfd;border-radius:6px;cursor:pointer;font-size:13px;">&#128229; qa.json</button>
    </div>
    <div class="stats-bar">
        <div class="stat"><div class="val">{n_nodes}</div>Nodes</div>
        <div class="stat"><div class="val">{n_ego_edges}</div>Ego Edges</div>
        <div class="stat"><div class="val">{n_inter_edges}</div>Inter Edges</div>
        <div class="stat"><div class="val">{n_cols}</div>Timesteps</div>
        {qa_stats_html}
    </div>

    <!-- ── Temporal BEV Sequence ── -->
    {bev_section}

    <!-- ── Scene View: BEV + Camera Grid ── -->
    <div class="scene-view" style="grid-template-columns: {scene_view_cols}">
        {bev_panel}
        <div class="cam-grid" style="grid-template-columns: {cam_grid_cols}; grid-template-rows: {cam_grid_rows}">
{cam_grid_html}        </div>
    </div>

    <!-- ── Causal Graph ── -->
    <div id="graph-stage-bar" style="display:flex;align-items:center;gap:10px;padding:8px 16px;background:#1a1a1a;border-bottom:1px solid #333;max-width:1600px;margin:0 auto 0 auto;border-radius:8px 8px 0 0;">
        <button id="btn-stage1" onclick="switchGraph('stage1')" style="padding:5px 14px;border-radius:4px;border:1px solid #3498DB;background:#3498DB22;color:#3498DB;cursor:pointer;font-size:13px;">{stage1_label}</button>
        <button id="btn-pruned" onclick="switchGraph('pruned')" {pruned_button_attrs}>{pruned_label}</button>
        <button id="btn-edited" onclick="switchGraph('edited')" {edited_button_attrs}>{edited_label}</button>
        <button id="btn-fixed" onclick="switchGraph('fixed')" {fixed_button_attrs}>{fixed_label}</button>
        <button id="btn-verified" onclick="switchGraph('verified')" {verified_button_attrs}>{verified_label}</button>
        <span style="flex:1"></span>
        <button class="undo-btn" id="btn-undo" onclick="editorUndo()" disabled title="Undo">&#8617;</button>
        <button class="redo-btn" id="btn-redo" onclick="editorRedo()" disabled title="Redo">&#8618;</button>
        <button class="edit-btn" id="btn-edit" onclick="toggleEditMode()">Edit Mode</button>
        <button class="save-btn" id="btn-save" onclick="saveEditedGraph()">Save JSON</button>
        <span id="changes-summary" style="font-size:12px;color:#888;">{changes_summary_text}</span>
    </div>
    <div id="edit-panel">
        <div class="edit-tabs">
            <div class="edit-tab active" onclick="switchEditTab('nodes', this)">Nodes</div>
            <div class="edit-tab" onclick="switchEditTab('edges', this)">Edges</div>
        </div>
        <div id="edit-nodes" class="edit-tab-content active"></div>
        <div id="edit-edges" class="edit-tab-content"></div>
    </div>
    <div id="graph-container">
        <div id="legend">
            <div class="legend-item"><div class="legend-color" style="background:#E74C3C"></div>Ego</div>
            <div class="legend-item"><div class="legend-color" style="background:#9B59B6"></div>Traffic Control</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498DB"></div>Other Agent</div>
            <div class="legend-item"><div class="legend-color" style="background:#7F8C8D"></div>Environment</div>
            <div class="legend-item"><div class="legend-color" style="background:#E67E22"></div>Road Obstacle</div>
            <hr style="border-color:#444; margin:5px 0">
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#F39C12" stroke-width="2"/></svg>&nbsp;Constrains Long.</div>
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#1ABC9C" stroke-width="2"/></svg>&nbsp;Constrains Lat.</div>
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#E74C3C" stroke-width="2"/></svg>&nbsp;Inhibits Proceed</div>
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#2ECC71" stroke-width="2"/></svg>&nbsp;Grants Proceed</div>
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#95A5A6" stroke-width="2"/></svg>&nbsp;Degrades</div>
            <hr style="border-color:#444; margin:5px 0">
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#3498DB" stroke-width="2" stroke-dasharray="5,5"/></svg>&nbsp;Affects</div>
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#9B59B6" stroke-width="2" stroke-dasharray="5,5"/></svg>&nbsp;Confounded By</div>
        </div>
        <div id="tooltip"></div>
    </div>

    {qa_body}

    <!-- ── JSON Viewer ── -->
    <div class="json-panel">
        <div class="json-toggle" onclick="toggleJson(this)">
            <span class="label">JSON</span>
            <span class="arrow">&#9660;</span>
        </div>
        <div class="json-content" id="json-content">
            <div class="json-tabs">
                <button class="json-tab active" onclick="switchJsonTab('graph', this)">Causal Graph</button>
                <button class="json-tab" onclick="switchJsonTab('edited', this)" id="tab-edited" style="display:none">Edited Graph</button>
                <button class="json-tab" onclick="switchJsonTab('verification', this)" id="tab-verification" {'style="display:none"' if verification_json == 'null' else ''}>Verification</button>
            </div>
            <div class="json-actions">
                <button onclick="copyJson(this)">Copy JSON</button>
                <button onclick="downloadJson()">Download .json</button>
            </div>
            <div id="json-viewer"></div>
        </div>
    </div>

    <script>
        let graphData = {graph_data_json};
        const rawGraphData = {raw_graph_json};
        let prunedGraphData = {pruned_graph_json};
        const userEditedGraphData = {edited_graph_json};
        const fixedGraphData = {fixed_graph_json};
        const verifiedGraphData = {verified_graph_json};
        const changesData = {changes_json};
        const sceneRecord = {scene_record_json};
        let editedGraphData = null;  // loaded from server if exists

        // ── Shared tooltip state ──
        const tooltip = d3.select("#tooltip");
        let pinned = false;
        function closeTooltip() {{
            pinned = false;
            tooltip.classed("pinned", false).style("opacity", 0).html("");
        }}

        // ── Tooltip drag-to-move + drag-to-resize ──
        (function() {{
            const tip = document.getElementById('tooltip');
            let mode = null, startX, startY, startL, startT, startW, startH;
            tip.addEventListener('click', function(e) {{
                if (e.target.classList.contains('close-btn')) {{ closeTooltip(); e.stopPropagation(); }}
            }});
            tip.addEventListener('mousedown', function(e) {{
                if (e.target.classList.contains('resize-handle')) {{
                    mode = 'resize';
                }} else if (e.target.closest('.tip-header') && !e.target.classList.contains('close-btn')) {{
                    mode = 'drag';
                }} else {{ return; }}
                e.preventDefault(); e.stopPropagation();
                startX = e.clientX; startY = e.clientY;
                startW = tip.offsetWidth; startH = tip.offsetHeight;
                startL = tip.offsetLeft; startT = tip.offsetTop;
            }});
            document.addEventListener('mousemove', function(e) {{
                if (!mode) return;
                if (mode === 'resize') {{
                    tip.style.width  = Math.max(220, startW + (e.clientX - startX)) + 'px';
                    tip.style.height = Math.max(120, startH + (e.clientY - startY)) + 'px';
                }} else if (mode === 'drag') {{
                    tip.style.left = (startL + e.clientX - startX) + 'px';
                    tip.style.top  = (startT + e.clientY - startY) + 'px';
                }}
            }});
            document.addEventListener('mouseup', function() {{ mode = null; }});
        }})();

        function drag(sim) {{
            return d3.drag()
                .on("start", (ev) => {{ if(!ev.active) sim.alphaTarget(0.3).restart(); ev.subject.fx=ev.subject.x; ev.subject.fy=ev.subject.y; }})
                .on("drag", (ev) => {{ ev.subject.fx=ev.x; ev.subject.fy=ev.y; }})
                .on("end", (ev) => {{ if(!ev.active) sim.alphaTarget(0); ev.subject.fx=null; ev.subject.fy=null; }});
        }}

        function rebuildGraph(data) {{
            closeTooltip();
            const container = d3.select("#graph-container");
            container.selectAll("svg").remove();
            const rect = container.node().getBoundingClientRect();
            const width = rect.width || 1600, height = rect.height || 650;
            const svg = container.append("svg").attr("width", "100%").attr("height", "100%")
                .attr("viewBox", `0 0 ${{width}} ${{height}}`);
            const gEl = svg.append("g");
            svg.call(d3.zoom().extent([[0,0],[width,height]]).scaleExtent([0.1,10])
                .filter(ev => !ev.ctrlKey && !ev.button)
                .on("zoom", ev => gEl.attr("transform", ev.transform)));
            svg.append("defs").selectAll("marker").data(["arrow-ego","arrow-inter"])
                .enter().append("marker").attr("id", d=>d).attr("viewBox","0 -5 10 10")
                .attr("refX",28).attr("refY",0).attr("markerWidth",6).attr("markerHeight",6)
                .attr("orient","auto").append("path").attr("d","M0,-5L10,0L0,5")
                .attr("fill", d => d==="arrow-ego" ? "#E74C3C" : "#3498DB");

            const nodes = (data.nodes || []).map((n, i) => ({{ ...n, index: i }}));
            const nodeMap = {{}};
            nodes.forEach(n => {{ nodeMap[n.id] = n; }});
            const links = [];
            (data.ego_edges || []).forEach(e => {{
                if (nodeMap[e.source] && nodeMap[e.target])
                    links.push({{ source: nodeMap[e.source], target: nodeMap[e.target], type:'ego',
                        effect: e.effect_type, active: e.active, priority: e.priority_level,
                        justification: e.justification }});
            }});
            (data.inter_node_edges || []).forEach(e => {{
                if (nodeMap[e.source] && nodeMap[e.target])
                    links.push({{ source: nodeMap[e.source], target: nodeMap[e.target], type:'inter',
                        relation: e.relation, justification: e.justification }});
            }});

            const sim = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d=>d.id).distance(160))
                .force("charge", d3.forceManyBody().strength(-700))
                .force("center", d3.forceCenter(width/2, height/2))
                .force("collision", d3.forceCollide().radius(45));
            const link = gEl.append("g").selectAll("line").data(links).enter().append("line")
                .attr("class", d => `link-${{d.type}} link-${{(d.active===true||d.active==='true')?'active':'latent'}}`)
                .attr("stroke-width", d => d.type==='ego' ? 3 : 2)
                .attr("marker-end", d => `url(#arrow-${{d.type}})`);
            const edgeLbl = gEl.append("g").selectAll("text").data(links).enter().append("text")
                .attr("class","edge-label").text(d => d.effect||d.relation||"");
            const node = gEl.append("g").selectAll("circle").data(nodes).enter().append("circle")
                .attr("r", d => d.id==="ego"?28:20)
                .attr("class", d => `node-${{d.type||'OTHER_AGENT'}}`)
                .attr("stroke-width", 2).call(drag(sim));
            const nodeLbl = gEl.append("g").selectAll("text").data(nodes).enter().append("text")
                .attr("class","node-label").text(d => d.id);

            function buildTooltipHTML(d) {{
                const outgoing = links.filter(l => l.source.id===d.id);
                const incoming = links.filter(l => l.target.id===d.id);
                function edgeRow(l, dir) {{
                    const other = dir==='out' ? l.target.id : l.source.id;
                    const arrow = dir==='out' ? '&rarr;' : '&larr;';
                    const color = dir==='out' ? '#E74C3C' : '#2ECC71';
                    return `<div style="margin:4px 0;padding:6px 0;border-bottom:1px solid #333">`
                        +`<span style="color:${{color}};font-weight:bold">${{arrow}} ${{other}}</span> `
                        +`<span style="color:#aaa">[${{l.effect||l.relation}}]</span><br/>`
                        +`<span style="color:#ccc">${{l.justification||''}}</span></div>`;
                }}
                let outHTML = outgoing.map(l => edgeRow(l, 'out')).join('');
                let inHTML  = incoming.map(l => edgeRow(l, 'in')).join('');
                let edgesHTML = '';
                if (outHTML) edgesHTML += `<div style="margin-top:6px"><b style="color:#E74C3C">Outgoing (${{outgoing.length}})</b></div>` + outHTML;
                if (inHTML)  edgesHTML += `<div style="margin-top:6px"><b style="color:#2ECC71">Incoming (${{incoming.length}})</b></div>` + inHTML;
                return `<div class="tip-header"><span class="title">${{d.label}}</span>`
                    +`<span class="close-btn">&times;</span></div>`
                    +`<div class="tip-body">`
                    +`Type: ${{d.type}}<br/>`
                    +`${{d.state?'State: '+d.state+'<br/>':''}}`
                    +`Pos: ${{d.position_relative_to_ego||d.position||''}}`
                    +`${{edgesHTML?'<hr style="border-color:#555;margin:6px 0">'+edgesHTML:''}}`
                    +`</div><div class="resize-handle"></div>`;
            }}
            let focusedNode = null;
            function focusNode(d) {{
                focusedNode = d.id;
                svg.classed("focus-active", true);
                const connected = new Set([d.id]);
                links.forEach(l => {{
                    const sid = typeof l.source === 'object' ? l.source.id : l.source;
                    const tid = typeof l.target === 'object' ? l.target.id : l.target;
                    if (sid === d.id || tid === d.id) {{ connected.add(sid); connected.add(tid); }}
                }});
                link.classed("dimmed", l => {{
                    const sid = typeof l.source === 'object' ? l.source.id : l.source;
                    const tid = typeof l.target === 'object' ? l.target.id : l.target;
                    return sid !== d.id && tid !== d.id;
                }});
                edgeLbl.classed("dimmed", l => {{
                    const sid = typeof l.source === 'object' ? l.source.id : l.source;
                    const tid = typeof l.target === 'object' ? l.target.id : l.target;
                    return sid !== d.id && tid !== d.id;
                }});
                node.classed("dimmed", n => !connected.has(n.id));
                nodeLbl.classed("dimmed", n => !connected.has(n.id));
            }}
            function clearFocus() {{
                focusedNode = null;
                svg.classed("focus-active", false);
                link.classed("dimmed", false);
                edgeLbl.classed("dimmed", false);
                node.classed("dimmed", false);
                nodeLbl.classed("dimmed", false);
            }}
            node.on("mouseenter", function(ev,d) {{
                if (pinned) return;
                tooltip.style("opacity",1).style("left",(ev.offsetX+15)+"px").style("top",(ev.offsetY-10)+"px")
                    .html(buildTooltipHTML(d));
            }}).on("mousemove", function(ev) {{
                if (pinned) return;
                tooltip.style("left",(ev.offsetX+15)+"px").style("top",(ev.offsetY-10)+"px");
            }}).on("mouseleave", () => {{
                if (!pinned) tooltip.style("opacity",0);
            }}).on("click", function(ev,d) {{
                ev.stopPropagation();
                pinned = true;
                tooltip.classed("pinned", true).style("opacity",1)
                    .style("left",(ev.offsetX+15)+"px").style("top",(ev.offsetY-10)+"px")
                    .html(buildTooltipHTML(d));
                if (focusedNode === d.id) {{ clearFocus(); }}
                else {{ focusNode(d); }}
            }});
            svg.on("click", () => {{ if (pinned) closeTooltip(); clearFocus(); }});
            sim.on("tick", () => {{
                link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
                    .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
                edgeLbl.attr("x",d=>(d.source.x+d.target.x)/2).attr("y",d=>(d.source.y+d.target.y)/2);
                node.attr("cx",d=>d.x).attr("cy",d=>d.y);
                nodeLbl.attr("x",d=>d.x).attr("y",d=>d.y+4);
            }});
        }}

        // ── Stage toggle ──
        let currentStage = 'stage1';
        function switchGraph(stage) {{
            currentStage = stage;
            let data;
            if (stage === 'verified' && verifiedGraphData) data = verifiedGraphData;
            else if (stage === 'fixed' && fixedGraphData) data = fixedGraphData;
            else if (stage === 'edited' && userEditedGraphData) data = userEditedGraphData;
            else if (stage === 'pruned' && prunedGraphData) data = prunedGraphData;
            else if (stage === 'stage1' && rawGraphData) data = rawGraphData;
            else data = graphData;
            ['stage1', 'pruned', 'edited', 'fixed', 'verified'].forEach(s => {{
                const btn = document.getElementById('btn-' + s);
                if (!btn || btn.disabled) return;
                const colors = {{ stage1: '#3498DB', pruned: '#2ECC71', edited: '#F1C40F', fixed: '#E91E63', verified: '#16A085' }};
                const c = colors[s];
                btn.style.borderColor = stage === s ? c : '#555';
                btn.style.color = stage === s ? c : '#aaa';
                btn.style.background = stage === s ? c + '22' : 'transparent';
            }});
            rebuildGraph(data);
        }}

        // ── Initial render ── pick the last-available tab:
        // verified > fixed > edited > pruned > stage1
        rebuildGraph(graphData);
        if (verifiedGraphData) switchGraph('verified');
        else if (fixedGraphData) switchGraph('fixed');
        else if (userEditedGraphData) switchGraph('edited');
        else if (prunedGraphData) switchGraph('pruned');
        else switchGraph('stage1');

        // ── Re-render graph on back/forward navigation (bfcache restore) ──
        window.addEventListener('pageshow', function(e) {{
            if (e.persisted) {{
                const data = (typeof prunedGraphData !== 'undefined' && prunedGraphData) ? prunedGraphData : graphData;
                rebuildGraph(data);
            }}
        }});

        // ── Load edited graph from server on page load ──
        (function() {{
            if (!window.SERVER_MODE) return;
            const sid = '{scene_id}';
            fetch('/api/edited/' + sid).then(r => {{
                if (!r.ok) return null;
                return r.json();
            }}).then(data => {{
                if (!data) return;
                editedGraphData = data;
                // Replace active graph data so the visualization shows the edit
                Object.keys(data).forEach(k => {{ if (k !== '_usage') graphData[k] = data[k]; }});
                if (prunedGraphData) Object.keys(data).forEach(k => {{ if (k !== '_usage') prunedGraphData[k] = data[k]; }});
                rebuildGraph(prunedGraphData || graphData);
                // Show the Edited Graph tab in JSON viewer
                const editedTab = document.getElementById('tab-edited');
                if (editedTab) editedTab.style.display = '';
                // Auto-enter edit mode with the saved data
                if (typeof enterEditMode === 'function') enterEditMode();
            }}).catch(() => {{}});
        }})();

        // ── JSON Viewer ──
        const rawJson = `{json_for_js}`;
        const verificationData = {verification_json};
        let activeJsonTab = 'graph';
        function switchJsonTab(tab, btn) {{
            activeJsonTab = tab;
            document.querySelectorAll('.json-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            let content;
            if (tab === 'verification' && verificationData) content = JSON.stringify(verificationData, null, 2);
            else if (tab === 'edited' && editedGraphData) content = JSON.stringify(editedGraphData, null, 2);
            else content = rawJson;
            document.getElementById('json-viewer').innerHTML = syntaxHighlight(content);
        }}
        function syntaxHighlight(json) {{
            return json.replace(/("(\\\\u[a-zA-Z0-9]{{4}}|\\\\[^u]|[^\\\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(\.\d+)?([eE][+-]?\d+)?)/g,
                function(match) {{
                    let cls = 'json-number';
                    if (/^"/.test(match)) {{
                        cls = /:$/.test(match) ? 'json-key' : 'json-string';
                    }} else if (/true|false/.test(match)) {{
                        cls = 'json-bool';
                    }} else if (/null/.test(match)) {{
                        cls = 'json-null';
                    }}
                    return '<span class="' + cls + '">' + match + '</span>';
                }});
        }}
        document.getElementById('json-viewer').innerHTML = syntaxHighlight(rawJson);

        function toggleJson(el) {{
            el.classList.toggle('open');
            document.getElementById('json-content').classList.toggle('open');
        }}
        function copyJson(btn) {{
            const data = currentStage === 'stage1' && rawGraphData ? rawGraphData : graphData;
            navigator.clipboard.writeText(JSON.stringify(data, null, 2)).then(() => {{
                btn.classList.add('copied');
                btn.textContent = 'Copied!';
                setTimeout(() => {{ btn.classList.remove('copied'); btn.textContent = 'Copy JSON'; }}, 2000);
            }});
        }}
        function downloadJson() {{
            let content, filename;
            if (activeJsonTab === 'verification' && verificationData) {{
                content = JSON.stringify(verificationData, null, 2);
                filename = '{scene_id}_verification.json';
            }} else if (activeJsonTab === 'edited' && editedGraphData) {{
                content = JSON.stringify(editedGraphData, null, 2);
                filename = '{scene_id}_edited.json';
            }} else {{
                content = rawJson;
                filename = '{scene_id}.json';
            }}
            const blob = new Blob([content], {{type: 'application/json'}});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            a.click();
        }}

        const metaData   = JSON.parse(`{meta_json_for_js}`);
        const framesData = JSON.parse(`{frames_json_for_js}`);
        const calibData  = JSON.parse(`{calib_json_for_js}`);
        const stateData  = JSON.parse(`{state_json_for_js}`);
        const qaData     = JSON.parse(`{qa_result_json}`);

        function downloadFile(filename, data) {{
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            a.click();
        }}
        {qa_js}

        // ══════════════════════════════════════════════════════════════
        // ██  GRAPH EDITOR                                           ██
        // ══════════════════════════════════════════════════════════════

        // ── Taxonomy constants ──
        const NODE_TYPES = ['EGO','OTHER_AGENT','TRAFFIC_CONTROL','ROAD_OBSTACLE','ENVIRONMENT'];
        const VALID_EFFECTS = {{
            TRAFFIC_CONTROL: ['INHIBITS_PROCEED','GRANTS_PROCEED'],
            ROAD_OBSTACLE:   ['CONSTRAINS_LONGITUDINAL','CONSTRAINS_LATERAL','INHIBITS_PROCEED'],
            OTHER_AGENT:     ['CONSTRAINS_LONGITUDINAL','CONSTRAINS_LATERAL','INHIBITS_PROCEED'],
            ENVIRONMENT:     ['DEGRADES']
        }};
        const AFFECTED_MAP = {{
            CONSTRAINS_LONGITUDINAL: ['speed','acceleration'],
            CONSTRAINS_LATERAL:      ['heading'],
            INHIBITS_PROCEED:        ['speed','heading','acceleration'],
            GRANTS_PROCEED:          ['speed','heading','acceleration'],
            DEGRADES:                ['speed','heading','acceleration']
        }};
        const CAUSAL_STATUSES = ['active_sustaining','active_triggering','dormant'];
        const INTER_RELATIONS = ['AFFECTS','CONFOUNDED_BY'];

        // Category suggestions per node type
        const CATEGORY_OPTIONS = {{
            OTHER_AGENT: [
                'vehicle.car','vehicle.truck','vehicle.bus.rigid','vehicle.bus.bendy',
                'vehicle.motorcycle','vehicle.bicycle','vehicle.trailer','vehicle.construction',
                'human.pedestrian.adult','human.pedestrian.child',
                'human.pedestrian.construction_worker','human.pedestrian.police_officer',
                'human.pedestrian.personal_mobility','human.pedestrian.stroller',
                'REGULAR_VEHICLE','LARGE_VEHICLE','BOX_TRUCK','TRUCK','TRUCK_CAB',
                'BUS','SCHOOL_BUS','BICYCLE','BICYCLIST','MOTORCYCLIST',
                'PEDESTRIAN','VEHICULAR_TRAILER','WHEELCHAIR','WHEELED_RIDER'
            ],
            TRAFFIC_CONTROL: [
                'signal.red','signal.green','signal.yellow','signal.pedestrian',
                'sign.stop','sign.yield','sign.speed_limit','sign.no_entry','sign.regulatory',
                'marking.crosswalk','marking.stop_line'
            ],
            ROAD_OBSTACLE: [
                'cone','debris','pothole','standing_water','rock','fallen_branch','barrier'
            ],
            ENVIRONMENT: [
                'wet_road','fog','rain','night','snow','hail','glare'
            ]
        }};

        // ── Edit state ──
        let editMode = false;
        let editableGraph = null;
        let editHistory = [];
        let editHistoryIdx = -1;
        let activeEditTab = 'nodes';

        function deepClone(o) {{ return JSON.parse(JSON.stringify(o)); }}

        function pushEditState() {{
            editHistory = editHistory.slice(0, editHistoryIdx + 1);
            editHistory.push(deepClone(editableGraph));
            editHistoryIdx = editHistory.length - 1;
            updateUndoRedo();
        }}
        function editorUndo() {{
            if (editHistoryIdx <= 0) return;
            editHistoryIdx--;
            editableGraph = deepClone(editHistory[editHistoryIdx]);
            refreshAll();
        }}
        function editorRedo() {{
            if (editHistoryIdx >= editHistory.length - 1) return;
            editHistoryIdx++;
            editableGraph = deepClone(editHistory[editHistoryIdx]);
            refreshAll();
        }}
        function updateUndoRedo() {{
            const u = document.getElementById('btn-undo');
            const r = document.getElementById('btn-redo');
            if (u) u.disabled = editHistoryIdx <= 0;
            if (r) r.disabled = editHistoryIdx >= editHistory.length - 1;
        }}
        function refreshAll() {{
            rebuildGraph(editableGraph);
            renderEditPanel();
            updateUndoRedo();
        }}

        // ── Enter / Exit edit mode ──
        function toggleEditMode() {{
            if (editMode) exitEditMode();
            else enterEditMode();
        }}
        function enterEditMode() {{
            editMode = true;
            const activeData = (currentStage === 'pruned' && prunedGraphData) ? prunedGraphData
                : (currentStage === 'stage1' && rawGraphData) ? rawGraphData : graphData;
            editableGraph = deepClone(activeData);
            if (!editableGraph.distractors) editableGraph.distractors = [];
            editHistory = [deepClone(editableGraph)];
            editHistoryIdx = 0;
            document.getElementById('graph-stage-bar').classList.add('edit-active');
            document.getElementById('btn-edit').classList.add('active');
            document.getElementById('btn-edit').textContent = 'Exit Edit';
            document.getElementById('btn-save').style.display = 'inline-block';
            document.getElementById('btn-undo').style.display = 'inline-block';
            document.getElementById('btn-redo').style.display = 'inline-block';
            document.getElementById('btn-stage1').disabled = true;
            document.getElementById('btn-stage1').style.opacity = '0.3';
            const bp = document.getElementById('btn-pruned');
            if (bp) {{ bp.disabled = true; bp.style.opacity = '0.3'; }}
            document.getElementById('edit-panel').classList.add('open');
            rebuildGraph(editableGraph);
            renderEditPanel();
            updateUndoRedo();
        }}
        function exitEditMode() {{
            editMode = false;
            editableGraph = null;
            editHistory = [];
            editHistoryIdx = -1;
            document.getElementById('graph-stage-bar').classList.remove('edit-active');
            document.getElementById('btn-edit').classList.remove('active');
            document.getElementById('btn-edit').textContent = 'Edit Mode';
            document.getElementById('btn-save').style.display = 'none';
            document.getElementById('btn-undo').style.display = 'none';
            document.getElementById('btn-redo').style.display = 'none';
            document.getElementById('btn-stage1').disabled = false;
            document.getElementById('btn-stage1').style.opacity = '1';
            const bp = document.getElementById('btn-pruned');
            if (bp) {{ bp.disabled = false; bp.style.opacity = '1'; }}
            document.getElementById('edit-panel').classList.remove('open');
            switchGraph(currentStage);
        }}

        // ── Tab switching ──
        function switchEditTab(tab, el) {{
            activeEditTab = tab;
            document.querySelectorAll('.edit-tab').forEach(t => t.classList.remove('active'));
            el.classList.add('active');
            document.querySelectorAll('.edit-tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById('edit-' + tab).classList.add('active');
        }}
        function switchNodeSubTab(tab, el) {{
            document.querySelectorAll('.edit-subtab').forEach(t => t.classList.remove('active'));
            el.classList.add('active');
            document.querySelectorAll('.node-subtab-content').forEach(c => c.classList.remove('active'));
            document.getElementById('node-subtab-' + tab).classList.add('active');
        }}

        function nextNodeId() {{
            const ids = (editableGraph.nodes || []).map(n => n.id)
                .concat((editableGraph.distractors || []).map(d => d.id));
            let i = 1;
            while (ids.includes('n' + i)) i++;
            return 'n' + i;
        }}

        // ── Autocomplete options ──
        const LABEL_OPTIONS = {{
            OTHER_AGENT: ['car','truck','bus','motorcycle','bicycle','pedestrian','trailer','construction vehicle','police officer','construction worker','child','wheelchair','stroller'],
            TRAFFIC_CONTROL: ['red traffic light','green traffic light','yellow traffic light','pedestrian signal','stop sign','yield sign','speed limit sign','no entry sign','crosswalk marking','stop line marking'],
            ROAD_OBSTACLE: ['traffic cone','debris','pothole','standing water','rock','fallen branch','barrier','construction zone'],
            ENVIRONMENT: ['wet road surface','fog','heavy rain','nighttime','snow','glare','sun glare']
        }};
        const STATE_OPTIONS = {{
            OTHER_AGENT: ['stationary','moving slowly','moderate speed','high speed','decelerating','accelerating','stopped at signal','parked','turning left','turning right','changing lanes','crossing road','waiting to cross'],
            TRAFFIC_CONTROL: ['red','green','yellow','flashing','active','visible','partially occluded','unoccupied','occupied'],
            ROAD_OBSTACLE: ['stationary','partially blocking lane','fully blocking lane','on road edge','in ego path'],
            ENVIRONMENT: ['active','worsening','improving','mild','severe','intermittent']
        }};
        const POSITION_OPTIONS = [
            'ahead same lane','ahead left lane','ahead right lane','ahead-left','ahead-right',
            'behind same lane','behind-left','behind-right',
            'left adjacent lane','right adjacent lane',
            'far left','far right','oncoming lane',
            'at intersection','in crosswalk','on sidewalk','on road shoulder'
        ];
        const VISIBLE_CAMS = [
            'CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT',
            'BEV'
        ];

        function visibleInCheckboxes(inputId, currentValue) {{
            const selected = (currentValue || '').split(',').map(s => s.trim()).filter(Boolean);
            let html = `<input type="hidden" id="${{inputId}}" value="${{(currentValue||'').replace(/"/g,'&quot;')}}"/>`;
            html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">';
            VISIBLE_CAMS.forEach(c => {{
                const checked = selected.includes(c) ? 'checked' : '';
                const short = c.replace('CAM_','');
                html += `<label style="font-size:10px;color:#aaa;display:flex;align-items:center;gap:2px;cursor:pointer">`
                    + `<input type="checkbox" value="${{c}}" ${{checked}} style="width:12px;height:12px;accent-color:#3498DB" `
                    + `onchange="updateVisCheckboxes('${{inputId}}')">`
                    + `${{short}}</label>`;
            }});
            html += '</div>';
            return html;
        }}
        function updateVisCheckboxes(inputId) {{
            const hidden = document.getElementById(inputId);
            if (!hidden) return;
            const wrap = hidden.parentElement;
            const checked = wrap.querySelectorAll('input[type=checkbox]:checked');
            hidden.value = Array.from(checked).map(cb => cb.value).join(', ');
        }}

        function datalistHTML(id, options) {{
            return `<datalist id="${{id}}">` + options.map(o => `<option value="${{o}}">`).join('') + '</datalist>';
        }}

        // ── Classify nodes: active (has edges), dormant (no edges), ego ──
        function classifyNodes() {{
            const egoEdges = editableGraph.ego_edges || [];
            const interEdges = editableGraph.inter_node_edges || [];
            const connectedIds = new Set();
            egoEdges.forEach(e => {{ connectedIds.add(e.source); connectedIds.add(e.target); }});
            interEdges.forEach(e => {{ connectedIds.add(e.source); connectedIds.add(e.target); }});
            const active = [], dormant = [];
            (editableGraph.nodes || []).forEach((n, i) => {{
                if (n.id === 'ego') return;  // ego always shown separately
                if (connectedIds.has(n.id)) active.push({{ node: n, idx: i }});
                else dormant.push({{ node: n, idx: i }});
            }});
            return {{ active, dormant }};
        }}

        // ── Auto-move orphaned nodes to distractors ──
        function autoDistractOrphans() {{
            const {{ dormant }} = classifyNodes();
            const moved = [];
            dormant.forEach(d => {{
                if (d.node.id === 'ego') return;
                moved.push(d.node);
            }});
            if (!moved.length) return;
            moved.forEach(n => {{
                editableGraph.distractors = editableGraph.distractors || [];
                const dist = Object.assign({{}}, n);
                dist.rejection_reason = dist.rejection_reason || 'Auto-moved: no remaining causal edges';
                editableGraph.distractors.push(dist);
                const idx = editableGraph.nodes.findIndex(nd => nd.id === n.id);
                if (idx >= 0) editableGraph.nodes.splice(idx, 1);
            }});
        }}

        // ── Helper: category dropdown ──
        function categoryDropdown(id, nodeType, selected) {{
            const cats = CATEGORY_OPTIONS[nodeType] || [];
            let html = `<input id="${{id}}" list="${{id}}-list" value="${{(selected||'').replace(/"/g,'&quot;')}}" placeholder="e.g. vehicle.car"/>`;
            html += `<datalist id="${{id}}-list">`;
            cats.forEach(c => {{ html += `<option value="${{c}}">`; }});
            html += '</datalist>';
            return html;
        }}

        // ── Render edit panel ──
        function renderEditPanel() {{
            renderNodeList();
            renderEdgeList();
        }}

        // ── NODE LIST ──
        function renderNodeList() {{
            const el = document.getElementById('edit-nodes');
            const {{ active }} = classifyNodes();
            const distractors = editableGraph.distractors || [];
            const nodes = editableGraph.nodes || [];
            // Active = all nodes (including ego)
            const activeNodes = [];
            nodes.forEach((n, i) => {{ activeNodes.push({{ node: n, idx: i }}); }});
            let html = '';

            // Sub-tabs
            html += `<div class="edit-subtabs">
                <div class="edit-subtab active" onclick="switchNodeSubTab('active', this)">Active/Dormant (${{activeNodes.length}})</div>
                <div class="edit-subtab" onclick="switchNodeSubTab('distractor', this)">Distractors (${{distractors.length}})</div>
            </div>`;

            // Active tab
            html += '<div id="node-subtab-active" class="node-subtab-content active"><div class="edit-scroll-section">';
            activeNodes.forEach(item => {{
                const isEgo = item.node.id === 'ego';
                html += `<div class="edit-row" data-node-idx="${{item.idx}}">
                    <span class="row-id">${{item.node.id}}</span>
                    <span class="row-type">${{item.node.type || ''}}</span>
                    <span class="row-label">${{item.node.label || item.node.category || ''}}</span>
                    <span class="row-actions">
                        <button onclick="showEditNodeForm(${{item.idx}})">Edit</button>
                        ${{isEgo ? '' : '<button class="danger" onclick="removeNode('+item.idx+')">Remove</button>'}}
                    </span>
                </div>`;
            }});
            if (!activeNodes.length) html += '<div style="padding:6px 12px;color:#666;font-size:12px">None</div>';
            html += '</div></div>';

            // Distractors tab
            html += '<div id="node-subtab-distractor" class="node-subtab-content"><div class="edit-scroll-section">';
            distractors.forEach((d, i) => {{
                html += `<div class="edit-row">
                    <span class="row-id">${{d.id}}</span>
                    <span class="row-type">${{d.type || ''}}</span>
                    <span class="row-label">${{d.label || d.category || d.id}}</span>
                    <span class="row-actions">
                        <button class="promote" onclick="promoteDistractor(${{i}})">Promote</button>
                        <button class="danger" onclick="deleteDistractor(${{i}})">Delete</button>
                    </span>
                </div>`;
            }});
            if (!distractors.length) html += '<div style="padding:6px 12px;color:#666;font-size:12px">None</div>';
            html += '</div></div>';

            html += `<button class="edit-add-btn" onclick="showAddNodeForm()">+ Add Node</button>`;
            html += `<div id="node-form-container"></div>`;
            el.innerHTML = html;
        }}

        // ── Available agents from state.json ──
        function getAvailableAgents() {{
            const agents = (sceneRecord && sceneRecord.state && sceneRecord.state.agents) || {{}};
            const inGraph = new Set((editableGraph.nodes || []).map(n => n.id));
            const result = [];
            Object.entries(agents).forEach(([id, info]) => {{
                if (!inGraph.has(id)) result.push({{ id, category: info.category || 'unknown' }});
            }});
            return result;
        }}

        function showAddNodeForm() {{
            const container = document.getElementById('node-form-container');
            const available = getAvailableAgents();
            let optionsHtml = '<option value="">-- Select source --</option>';
            if (available.length) {{
                optionsHtml += '<optgroup label="From scene agents">';
                available.forEach(a => {{
                    const inDist = (editableGraph.distractors || []).some(d => d.id === a.id);
                    const suffix = inDist ? ' (in distractors)' : '';
                    optionsHtml += `<option value="agent:${{a.id}}">${{a.id}} (${{a.category}})${{suffix}}</option>`;
                }});
                optionsHtml += '</optgroup>';
            }}
            optionsHtml += `<optgroup label="Manual">
                <option value="manual:TRAFFIC_CONTROL">New TRAFFIC_CONTROL</option>
                <option value="manual:ROAD_OBSTACLE">New ROAD_OBSTACLE</option>
                <option value="manual:ENVIRONMENT">New ENVIRONMENT</option>
            </optgroup>`;
            const defaultType = 'OTHER_AGENT';
            container.innerHTML = `<div class="edit-form">
                <div class="form-row"><label>Source</label>
                    <select id="add-node-source" onchange="onAddNodeSourceChange()">${{optionsHtml}}</select></div>
                <div class="form-row"><label>ID</label><input id="add-node-id" readonly/></div>
                <div class="form-row"><label>Type</label>
                    <select id="add-node-type" onchange="onAddNodeTypeChange()">${{NODE_TYPES.filter(t=>t!=='EGO').map(t=>'<option value="'+t+'">'+t+'</option>').join('')}}</select></div>
                <div class="form-row"><label>Category</label><div id="add-node-cat-wrap">${{categoryDropdown('add-node-category',defaultType,'')}}</div></div>
                <div class="form-row"><label>Label</label><input id="add-node-label" list="add-node-label-list" placeholder="Descriptive label"/>
                    ${{datalistHTML('add-node-label-list', LABEL_OPTIONS[defaultType] || [])}}</div>
                <div class="form-row"><label>State</label><input id="add-node-state" list="add-node-state-list" placeholder="Current state"/>
                    ${{datalistHTML('add-node-state-list', STATE_OPTIONS[defaultType] || [])}}</div>
                <div class="form-row"><label>Position</label><input id="add-node-position" list="add-node-pos-list" placeholder="e.g. ahead-left, behind same lane"/>
                    ${{datalistHTML('add-node-pos-list', POSITION_OPTIONS)}}</div>
                <div class="form-row"><label>Visible In</label>${{visibleInCheckboxes('add-node-visible', '')}}</div>
                <div class="form-actions">
                    <button class="btn-cancel" onclick="document.getElementById('node-form-container').innerHTML=''">Cancel</button>
                    <button class="btn-save" onclick="commitAddNode()">Add Node</button>
                </div>
            </div>`;
        }}
        function onAddNodeSourceChange() {{
            const val = document.getElementById('add-node-source').value;
            if (!val) return;
            const [kind, key] = val.split(':');
            if (kind === 'agent') {{
                const agents = (sceneRecord && sceneRecord.state && sceneRecord.state.agents) || {{}};
                const info = agents[key] || {{}};
                document.getElementById('add-node-id').value = key;
                document.getElementById('add-node-type').value = 'OTHER_AGENT';
                document.getElementById('add-node-cat-wrap').innerHTML = categoryDropdown('add-node-category','OTHER_AGENT', info.category || '');
                document.getElementById('add-node-label').value = key.replace('_', ' ');
                // Update datalists for OTHER_AGENT
                const lblList = document.getElementById('add-node-label-list');
                const stList = document.getElementById('add-node-state-list');
                if (lblList) lblList.innerHTML = (LABEL_OPTIONS['OTHER_AGENT']||[]).map(o=>'<option value="'+o+'">').join('');
                if (stList) stList.innerHTML = (STATE_OPTIONS['OTHER_AGENT']||[]).map(o=>'<option value="'+o+'">').join('');
            }} else {{
                document.getElementById('add-node-id').value = nextNodeId();
                document.getElementById('add-node-type').value = key;
                document.getElementById('add-node-cat-wrap').innerHTML = categoryDropdown('add-node-category', key, '');
                document.getElementById('add-node-label').value = '';
                const lblList = document.getElementById('add-node-label-list');
                const stList = document.getElementById('add-node-state-list');
                if (lblList) lblList.innerHTML = (LABEL_OPTIONS[key]||[]).map(o=>'<option value="'+o+'">').join('');
                if (stList) stList.innerHTML = (STATE_OPTIONS[key]||[]).map(o=>'<option value="'+o+'">').join('');
            }}
        }}
        function onAddNodeTypeChange() {{
            const t = document.getElementById('add-node-type').value;
            document.getElementById('add-node-cat-wrap').innerHTML = categoryDropdown('add-node-category', t, '');
            const lblList = document.getElementById('add-node-label-list');
            const stList = document.getElementById('add-node-state-list');
            if (lblList) lblList.innerHTML = (LABEL_OPTIONS[t]||[]).map(o=>'<option value="'+o+'">').join('');
            if (stList) stList.innerHTML = (STATE_OPTIONS[t]||[]).map(o=>'<option value="'+o+'">').join('');
        }}
        function commitAddNode() {{
            const id = document.getElementById('add-node-id').value.trim();
            if (!id) return alert('Select a source first');
            if (editableGraph.nodes.some(n => n.id === id)) return alert('Node ID already exists');
            const node = {{
                id: id,
                type: document.getElementById('add-node-type').value,
                category: document.getElementById('add-node-category').value,
                label: document.getElementById('add-node-label').value,
                state: document.getElementById('add-node-state').value,
                position_relative_to_ego: document.getElementById('add-node-position').value,
                visible_in: document.getElementById('add-node-visible').value, motion: {{}}
            }};
            editableGraph.nodes.push(node);
            // Remove from distractors if present
            editableGraph.distractors = (editableGraph.distractors || []).filter(d => d.id !== id);
            pushEditState();
            refreshAll();
        }}

        function showEditNodeForm(idx) {{
            const n = editableGraph.nodes[idx];
            if (!n) return;
            const row = document.querySelector(`[data-node-idx="${{idx}}"]`);
            if (!row) return;
            const existing = row.nextElementSibling;
            if (existing && existing.classList.contains('edit-form')) {{ existing.remove(); return; }}
            const typeOpts = NODE_TYPES.map(t => `<option value="${{t}}" ${{t===n.type?'selected':''}}>${{t}}</option>`).join('');
            const form = document.createElement('div');
            form.className = 'edit-form';
            const nType = n.type || 'OTHER_AGENT';
            form.innerHTML = `
                <div class="form-row"><label>Type</label>
                    <select id="edn-type-${{idx}}" onchange="onEditNodeTypeChange(${{idx}})">${{typeOpts}}</select></div>
                <div class="form-row"><label>Category</label>
                    <div id="edn-cat-wrap-${{idx}}">${{categoryDropdown('edn-cat-'+idx, nType, n.category||'')}}</div></div>
                <div class="form-row"><label>Label</label><input id="edn-label-${{idx}}" list="edn-label-list-${{idx}}" value="${{(n.label||'').replace(/"/g,'&quot;')}}"/>
                    ${{datalistHTML('edn-label-list-'+idx, LABEL_OPTIONS[nType]||[])}}</div>
                <div class="form-row"><label>State</label><input id="edn-state-${{idx}}" list="edn-state-list-${{idx}}" value="${{(n.state||'').replace(/"/g,'&quot;')}}"/>
                    ${{datalistHTML('edn-state-list-'+idx, STATE_OPTIONS[nType]||[])}}</div>
                <div class="form-row"><label>Position</label><input id="edn-pos-${{idx}}" list="edn-pos-list-${{idx}}" value="${{(n.position_relative_to_ego||'').replace(/"/g,'&quot;')}}"/>
                    ${{datalistHTML('edn-pos-list-'+idx, POSITION_OPTIONS)}}</div>
                <div class="form-row"><label>Visible In</label>${{visibleInCheckboxes('edn-vis-'+idx, n.visible_in||'')}}</div>
                <div class="form-actions">
                    <button class="btn-cancel" onclick="this.closest('.edit-form').remove()">Cancel</button>
                    <button class="btn-save" onclick="commitEditNode(${{idx}})">Save</button>
                </div>`;
            row.after(form);
        }}
        function onEditNodeTypeChange(idx) {{
            const t = document.getElementById('edn-type-' + idx).value;
            document.getElementById('edn-cat-wrap-' + idx).innerHTML = categoryDropdown('edn-cat-' + idx, t, '');
            const lblList = document.getElementById('edn-label-list-' + idx);
            const stList = document.getElementById('edn-state-list-' + idx);
            if (lblList) lblList.innerHTML = (LABEL_OPTIONS[t]||[]).map(o=>'<option value="'+o+'">').join('');
            if (stList) stList.innerHTML = (STATE_OPTIONS[t]||[]).map(o=>'<option value="'+o+'">').join('');
        }}
        function commitEditNode(idx) {{
            const n = editableGraph.nodes[idx];
            n.type = document.getElementById('edn-type-' + idx).value;
            n.category = document.getElementById('edn-cat-' + idx).value;
            n.label = document.getElementById('edn-label-' + idx).value;
            n.state = document.getElementById('edn-state-' + idx).value;
            n.position_relative_to_ego = document.getElementById('edn-pos-' + idx).value;
            n.visible_in = document.getElementById('edn-vis-' + idx).value;
            pushEditState();
            refreshAll();
        }}

        function removeNode(idx) {{
            const n = editableGraph.nodes[idx];
            if (!n || n.id === 'ego') return;
            const dist = Object.assign({{}}, n);
            dist.rejection_reason = dist.rejection_reason || 'Removed by annotator';
            editableGraph.distractors.push(dist);
            editableGraph.nodes.splice(idx, 1);
            editableGraph.ego_edges = (editableGraph.ego_edges || []).filter(e => e.source !== n.id && e.target !== n.id);
            editableGraph.inter_node_edges = (editableGraph.inter_node_edges || []).filter(e => e.source !== n.id && e.target !== n.id);
            pushEditState();
            refreshAll();
        }}

        // ── EDGE LIST ──
        function renderEdgeList() {{
            const el = document.getElementById('edit-edges');
            const egoEdges = editableGraph.ego_edges || [];
            const interEdges = editableGraph.inter_node_edges || [];
            let html = '<div class="edit-section-header">Ego Edges (' + egoEdges.length + ')</div>';
            egoEdges.forEach((e, i) => {{
                html += `<div class="edit-row" data-ego-edge-idx="${{i}}">
                    <span class="row-edge">
                        <span class="edge-source">${{e.source}}</span>
                        <span class="edge-arrow">&rarr;</span>
                        <span class="edge-target">${{e.target}}</span>
                        <span class="edge-effect">${{e.effect_type}}</span>
                        <span class="edge-status">${{e.causal_status || ''}}</span>
                    </span>
                    <span class="row-actions">
                        <button onclick="showEditEgoEdgeForm(${{i}})">Edit</button>
                        <button class="danger" onclick="removeEgoEdge(${{i}})">Delete</button>
                    </span>
                </div>`;
            }});
            html += `<button class="edit-add-btn" onclick="showAddEgoEdgeForm()">+ Add Ego Edge</button>`;
            html += `<div id="ego-edge-form-container"></div>`;

            html += '<div class="edit-section-header" style="margin-top:12px">Inter-Node Edges (' + interEdges.length + ')</div>';
            interEdges.forEach((e, i) => {{
                html += `<div class="edit-row" data-inter-edge-idx="${{i}}">
                    <span class="row-edge">
                        <span class="edge-source">${{e.source}}</span>
                        <span class="edge-arrow">&rarr;</span>
                        <span class="edge-target">${{e.target}}</span>
                        <span class="edge-effect">${{e.relation}}</span>
                        <span class="edge-status">Layer ${{e.layer || '?'}}</span>
                    </span>
                    <span class="row-actions">
                        <button onclick="showEditInterEdgeForm(${{i}})">Edit</button>
                        <button class="danger" onclick="removeInterEdge(${{i}})">Delete</button>
                    </span>
                </div>`;
            }});
            html += `<button class="edit-add-btn" onclick="showAddInterEdgeForm()">+ Add Inter-Node Edge</button>`;
            html += `<div id="inter-edge-form-container"></div>`;
            el.innerHTML = html;
        }}

        function nonEgoNodeOptions(exclude, selected) {{
            return (editableGraph.nodes || []).filter(n => n.id !== 'ego' && n.id !== exclude)
                .map(n => `<option value="${{n.id}}" ${{n.id===selected?'selected':''}}>${{n.id}} (${{n.type}})</option>`).join('');
        }}
        function effectOptions(sourceId, selected) {{
            const node = (editableGraph.nodes || []).find(n => n.id === sourceId);
            const valid = node ? (VALID_EFFECTS[node.type] || []) : [];
            return valid.map(e => `<option value="${{e}}" ${{e===selected?'selected':''}}>${{e}}</option>`).join('');
        }}
        function affectedStateDisplay(effectType) {{
            const states = AFFECTED_MAP[effectType] || [];
            return states.map(s => `<span>${{s}}</span>`).join('');
        }}

        function egoEdgeFormHTML(prefix, data) {{
            data = data || {{}};
            const srcOpts = '<option value="">-- source --</option>' + nonEgoNodeOptions('', data.source || '');
            const statusOpts = CAUSAL_STATUSES.map(s => `<option value="${{s}}" ${{s===(data.causal_status||'')?'selected':''}}>${{s}}</option>`).join('');
            return `
                <div class="form-row"><label>Source</label>
                    <select id="${{prefix}}-src" onchange="updateEffectDropdown('${{prefix}}')">${{srcOpts}}</select></div>
                <div class="form-row"><label>Target</label><div class="auto-filled">ego</div></div>
                <div class="form-row"><label>Effect Type</label>
                    <select id="${{prefix}}-effect" onchange="updateAffectedState('${{prefix}}')">${{effectOptions(data.source||'', data.effect_type||'')}}</select></div>
                <div class="form-row"><label>Affected State</label>
                    <div class="auto-filled" id="${{prefix}}-affected">${{affectedStateDisplay(data.effect_type||'')}}</div></div>
                <div class="form-row"><label>Causal Status</label><select id="${{prefix}}-status">${{statusOpts}}</select></div>
                <div class="form-row"><label>Justification</label><textarea id="${{prefix}}-just">${{data.justification||''}}</textarea></div>`;
        }}
        function updateEffectDropdown(prefix) {{
            const srcId = document.getElementById(prefix + '-src').value;
            document.getElementById(prefix + '-effect').innerHTML = effectOptions(srcId, '');
            updateAffectedState(prefix);
        }}
        function updateAffectedState(prefix) {{
            const eff = document.getElementById(prefix + '-effect').value;
            document.getElementById(prefix + '-affected').innerHTML = affectedStateDisplay(eff);
        }}

        function showAddEgoEdgeForm() {{
            const c = document.getElementById('ego-edge-form-container');
            if (c.querySelector('.edit-form')) {{ c.innerHTML = ''; return; }}
            c.innerHTML = `<div class="edit-form">
                ${{egoEdgeFormHTML('add-ego', {{}})}}
                <div class="form-actions">
                    <button class="btn-cancel" onclick="document.getElementById('ego-edge-form-container').innerHTML=''">Cancel</button>
                    <button class="btn-save" onclick="commitAddEgoEdge()">Add Edge</button>
                </div></div>`;
        }}
        function commitAddEgoEdge() {{
            const src = document.getElementById('add-ego-src').value;
            const eff = document.getElementById('add-ego-effect').value;
            if (!src || !eff) return alert('Select source and effect type');
            const status = document.getElementById('add-ego-status').value;
            editableGraph.ego_edges = editableGraph.ego_edges || [];
            editableGraph.ego_edges.push({{
                source: src, target: 'ego', effect_type: eff,
                affected_state: AFFECTED_MAP[eff] || [],
                active: status !== 'dormant',
                causal_status: status,
                justification: document.getElementById('add-ego-just').value
            }});
            pushEditState();
            refreshAll();
        }}
        function showEditEgoEdgeForm(idx) {{
            const row = document.querySelector(`[data-ego-edge-idx="${{idx}}"]`);
            if (!row) return;
            const existing = row.nextElementSibling;
            if (existing && existing.classList.contains('edit-form')) {{ existing.remove(); return; }}
            const e = editableGraph.ego_edges[idx];
            const form = document.createElement('div');
            form.className = 'edit-form';
            form.innerHTML = egoEdgeFormHTML('edit-ego-' + idx, e) +
                `<div class="form-actions">
                    <button class="btn-cancel" onclick="this.closest('.edit-form').remove()">Cancel</button>
                    <button class="btn-save" onclick="commitEditEgoEdge(${{idx}})">Save</button>
                </div>`;
            row.after(form);
        }}
        function commitEditEgoEdge(idx) {{
            const p = 'edit-ego-' + idx;
            const e = editableGraph.ego_edges[idx];
            e.source = document.getElementById(p + '-src').value;
            e.effect_type = document.getElementById(p + '-effect').value;
            e.causal_status = document.getElementById(p + '-status').value;
            e.active = e.causal_status !== 'dormant';
            e.affected_state = AFFECTED_MAP[e.effect_type] || [];
            e.justification = document.getElementById(p + '-just').value;
            pushEditState();
            refreshAll();
        }}
        function removeEgoEdge(idx) {{
            editableGraph.ego_edges.splice(idx, 1);
            autoDistractOrphans();
            pushEditState();
            refreshAll();
        }}

        // ── Inter-Node Edge forms ──
        function interEdgeFormHTML(prefix, data) {{
            data = data || {{}};
            const srcOpts = '<option value="">-- source --</option>' + nonEgoNodeOptions('', data.source || '');
            const tgtOpts = '<option value="">-- target --</option>' + nonEgoNodeOptions(data.source || '', data.target || '');
            const relOpts = INTER_RELATIONS.map(r => `<option value="${{r}}" ${{r===(data.relation||'')?'selected':''}}>${{r}}</option>`).join('');
            const isConf = (data.relation === 'CONFOUNDED_BY');
            const layerOpts = ['1','2','3','N/A'].map(l => `<option value="${{l}}" ${{l===String(data.layer||'')?'selected':''}}>${{l}}</option>`).join('');
            return `
                <div class="form-row"><label>Source</label><select id="${{prefix}}-src">${{srcOpts}}</select></div>
                <div class="form-row"><label>Target</label><select id="${{prefix}}-tgt">${{tgtOpts}}</select></div>
                <div class="form-row"><label>Relation</label>
                    <select id="${{prefix}}-rel" onchange="onInterRelChange('${{prefix}}')">${{relOpts}}</select></div>
                <div class="form-row"><label>Layer</label><select id="${{prefix}}-layer">${{layerOpts}}</select></div>
                <div class="form-row" id="${{prefix}}-cause-row" style="${{isConf?'':'display:none'}}">
                    <label>Inferred Cause</label><input id="${{prefix}}-cause" value="${{(data.inferred_cause||'').replace(/"/g,'&quot;')}}"/></div>
                <div class="form-row"><label>Justification</label><textarea id="${{prefix}}-just">${{data.justification||''}}</textarea></div>`;
        }}
        function onInterRelChange(prefix) {{
            const rel = document.getElementById(prefix + '-rel').value;
            const causeRow = document.getElementById(prefix + '-cause-row');
            const layerSel = document.getElementById(prefix + '-layer');
            if (rel === 'CONFOUNDED_BY') {{
                causeRow.style.display = '';
                layerSel.value = 'N/A';
            }} else {{
                causeRow.style.display = 'none';
            }}
        }}
        function showAddInterEdgeForm() {{
            const c = document.getElementById('inter-edge-form-container');
            if (c.querySelector('.edit-form')) {{ c.innerHTML = ''; return; }}
            c.innerHTML = `<div class="edit-form">
                ${{interEdgeFormHTML('add-inter', {{}})}}
                <div class="form-actions">
                    <button class="btn-cancel" onclick="document.getElementById('inter-edge-form-container').innerHTML=''">Cancel</button>
                    <button class="btn-save" onclick="commitAddInterEdge()">Add Edge</button>
                </div></div>`;
        }}
        function commitAddInterEdge() {{
            let src = document.getElementById('add-inter-src').value;
            let tgt = document.getElementById('add-inter-tgt').value;
            const rel = document.getElementById('add-inter-rel').value;
            if (!src || !tgt || !rel) return alert('Fill source, target, and relation');
            if (rel === 'CONFOUNDED_BY' && src > tgt) {{ [src, tgt] = [tgt, src]; }}
            editableGraph.inter_node_edges = editableGraph.inter_node_edges || [];
            editableGraph.inter_node_edges.push({{
                source: src, target: tgt, relation: rel,
                layer: document.getElementById('add-inter-layer').value,
                inferred_cause: rel === 'CONFOUNDED_BY' ? document.getElementById('add-inter-cause').value : undefined,
                justification: document.getElementById('add-inter-just').value
            }});
            pushEditState();
            refreshAll();
        }}
        function showEditInterEdgeForm(idx) {{
            const row = document.querySelector(`[data-inter-edge-idx="${{idx}}"]`);
            if (!row) return;
            const existing = row.nextElementSibling;
            if (existing && existing.classList.contains('edit-form')) {{ existing.remove(); return; }}
            const e = editableGraph.inter_node_edges[idx];
            const form = document.createElement('div');
            form.className = 'edit-form';
            form.innerHTML = interEdgeFormHTML('edit-inter-' + idx, e) +
                `<div class="form-actions">
                    <button class="btn-cancel" onclick="this.closest('.edit-form').remove()">Cancel</button>
                    <button class="btn-save" onclick="commitEditInterEdge(${{idx}})">Save</button>
                </div>`;
            row.after(form);
        }}
        function commitEditInterEdge(idx) {{
            const p = 'edit-inter-' + idx;
            const e = editableGraph.inter_node_edges[idx];
            e.source = document.getElementById(p + '-src').value;
            e.target = document.getElementById(p + '-tgt').value;
            e.relation = document.getElementById(p + '-rel').value;
            e.layer = document.getElementById(p + '-layer').value;
            e.inferred_cause = e.relation === 'CONFOUNDED_BY' ? document.getElementById(p + '-cause').value : undefined;
            e.justification = document.getElementById(p + '-just').value;
            if (e.relation === 'CONFOUNDED_BY' && e.source > e.target) {{
                [e.source, e.target] = [e.target, e.source];
            }}
            pushEditState();
            refreshAll();
        }}
        function removeInterEdge(idx) {{
            editableGraph.inter_node_edges.splice(idx, 1);
            autoDistractOrphans();
            pushEditState();
            refreshAll();
        }}

        // ── DISTRACTOR LIST (integrated into Nodes tab) ──
        function promoteDistractor(idx) {{
            const d = editableGraph.distractors[idx];
            if (!d) return;
            const node = Object.assign({{}}, d);
            // Ensure required node fields exist
            node.type = node.type || 'OTHER_AGENT';
            node.category = node.category || '';
            node.label = node.label || node.id.replace('_', ' ');
            node.state = node.state || '';
            node.position_relative_to_ego = node.position_relative_to_ego || '';
            node.visible_in = node.visible_in || '';
            node.motion = node.motion || {{}};
            delete node.rejection_reason;
            editableGraph.nodes.push(node);
            editableGraph.distractors.splice(idx, 1);
            pushEditState();
            refreshAll();
            // Switch to nodes tab so user can refine the promoted node
            const tab = document.querySelector('.edit-tab');
            if (tab) switchEditTab('nodes', tab);
        }}
        function deleteDistractor(idx) {{
            if (!editableGraph || !editableGraph.distractors) return;
            editableGraph.distractors.splice(idx, 1);
            pushEditState();
            refreshAll();
        }}

        // ── Save / Export ──
        const SERVER_MODE = window.SERVER_MODE || false;
        const CURRENT_USER = window.CURRENT_USER || '';
        const SCENE_ID = '{scene_id}';

        function saveEditedGraph() {{
            if (!editableGraph) return;
            const output = deepClone(editableGraph);
            output.nodes = (output.nodes || []).map(n => {{
                const clean = {{}};
                ['id','type','category','label','state','position_relative_to_ego',
                 'visible_in','motion'].forEach(k => {{ if (n[k] !== undefined) clean[k] = n[k]; }});
                return clean;
            }});

            if (SERVER_MODE) {{
                // POST to server
                const saveBtn = document.getElementById('btn-save');
                saveBtn.textContent = 'Saving...';
                saveBtn.disabled = true;
                fetch('/api/save', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ scene_id: SCENE_ID, graph: output }})
                }}).then(r => {{
                    if (!r.ok) throw new Error('Save failed: ' + r.status);
                    return r.json();
                }}).then(data => {{
                    saveBtn.textContent = 'Saved!';
                    saveBtn.style.borderColor = '#2ECC71';
                    saveBtn.style.color = '#2ECC71';
                    // Update edited data + graph in-place without reload
                    editedGraphData = deepClone(output);
                    Object.keys(output).forEach(k => {{ if (k !== '_usage') graphData[k] = output[k]; }});
                    if (prunedGraphData) Object.keys(output).forEach(k => {{ if (k !== '_usage') prunedGraphData[k] = output[k]; }});
                    // Show Edited Graph tab in JSON viewer
                    const editedTab = document.getElementById('tab-edited');
                    if (editedTab) editedTab.style.display = '';
                    setTimeout(() => {{
                        saveBtn.textContent = 'Save JSON';
                        saveBtn.disabled = false;
                        saveBtn.style.borderColor = '';
                        saveBtn.style.color = '';
                    }}, 2000);
                }}).catch(err => {{
                    alert('Save failed: ' + err.message);
                    saveBtn.textContent = 'Save JSON';
                    saveBtn.disabled = false;
                }});
            }} else {{
                // Fallback: download locally
                const blob = new Blob([JSON.stringify(output, null, 2)], {{type: 'application/json'}});
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = (editableGraph.scene_id || 'scene') + '_edited.json';
                a.click();
            }}
        }}
    </script>
</body>
</html>"""
