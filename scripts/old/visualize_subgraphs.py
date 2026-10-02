#!/usr/bin/env python3
"""
Interactive visualization tool for extracted causal subgraphs.
Creates HTML visualizations with the ability to highlight and explore different subgraph patterns.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any


def create_subgraph_visualization(graph_data: Dict, subgraphs: Dict, output_file: str) -> bool:
    """
    Create an interactive HTML visualization of the causal graph with subgraph highlighting.

    Args:
        graph_data: The full graph data containing nodes and edges
        subgraphs: Dictionary of extracted subgraphs (direct, chain, fork, collider)
        output_file: Path to save the HTML file
    """

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Causal Subgraph Explorer</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}

            .container {{
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
            }}

            h1 {{
                color: white;
                text-align: center;
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}

            .main-panel {{
                display: flex;
                gap: 20px;
                height: 800px;
            }}

            #graph-container {{
                flex: 1;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                position: relative;
                overflow: hidden;
            }}

            #control-panel {{
                width: 350px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 20px;
                overflow-y: auto;
            }}

            .pattern-section {{
                margin-bottom: 25px;
            }}

            .pattern-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 8px;
                cursor: pointer;
                transition: transform 0.2s;
            }}

            .pattern-header:hover {{
                transform: translateX(5px);
            }}

            .pattern-count {{
                background: rgba(255,255,255,0.3);
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
            }}

            .pattern-list {{
                margin-top: 10px;
                max-height: 200px;
                overflow-y: auto;
            }}

            .subgraph-item {{
                padding: 8px 12px;
                margin: 4px 0;
                background: #f8f9fa;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s;
                border: 2px solid transparent;
            }}

            .subgraph-item:hover {{
                background: #e9ecef;
                transform: translateX(5px);
            }}

            .subgraph-item.selected {{
                background: #e3f2fd;
                border-color: #2196f3;
                box-shadow: 0 2px 8px rgba(33,150,243,0.3);
            }}

            .node {{
                cursor: pointer;
                transition: all 0.3s;
            }}

            .node-ego {{
                fill: #ff6b6b;
                stroke: #c92a2a;
                stroke-width: 3px;
            }}

            .node-vru {{
                fill: #ffd43b;
                stroke: #fab005;
                stroke-width: 2px;
            }}

            .node-vehicle {{
                fill: #4dabf7;
                stroke: #1971c2;
                stroke-width: 2px;
            }}

            .node-traffic {{
                fill: #51cf66;
                stroke: #2f9e44;
                stroke-width: 2px;
            }}

            .node-road {{
                fill: #868e96;
                stroke: #495057;
                stroke-width: 2px;
            }}

            .node-env {{
                fill: #cc5de8;
                stroke: #9c36b5;
                stroke-width: 2px;
            }}

            .node-nav {{
                fill: #ff8787;
                stroke: #fa5252;
                stroke-width: 2px;
            }}

            .node-obstacle {{
                fill: #fd7e14;
                stroke: #e8590c;
                stroke-width: 2px;
            }}

            .node-emergency {{
                fill: #ff0000;
                stroke: #c92a2a;
                stroke-width: 2px;
                animation: pulse 1s infinite;
            }}

            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
                100% {{ opacity: 1; }}
            }}

            .node.highlighted {{
                stroke-width: 4px;
                filter: drop-shadow(0 0 8px rgba(255,193,7,0.8));
            }}

            .node.dimmed {{
                opacity: 0.2;
            }}

            .link {{
                fill: none;
                stroke-width: 2;
                transition: all 0.3s;
            }}

            .link-ego {{
                stroke: #e03131;
                stroke-width: 3;
                marker-end: url(#arrow-ego);
            }}

            .link-ego.link-active {{
                opacity: 1;
            }}

            .link-ego.link-latent {{
                opacity: 0.3;
                stroke-dasharray: 2,2;
            }}

            .link-inter {{
                stroke: #1971c2;
                stroke-dasharray: 5,5;
                marker-end: url(#arrow-inter);
            }}

            .link.highlighted {{
                stroke: #ffd93d;
                stroke-width: 4;
                filter: drop-shadow(0 0 6px rgba(255,193,7,0.6));
            }}

            .link.dimmed {{
                opacity: 0.1;
            }}

            .node-label {{
                font-size: 12px;
                font-weight: bold;
                pointer-events: none;
                text-anchor: middle;
                fill: white;
            }}

            .pattern-diagram {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 6px;
                margin-top: 10px;
            }}

            .mini-graph {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .mini-node {{
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background: #4dabf7;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 10px;
                font-weight: bold;
            }}

            .mini-node.ego {{
                background: #ff6b6b;
            }}

            .mini-arrow {{
                width: 20px;
                height: 2px;
                background: #868e96;
                position: relative;
            }}

            .mini-arrow::after {{
                content: '';
                position: absolute;
                right: -4px;
                top: -3px;
                width: 0;
                height: 0;
                border-left: 8px solid #868e96;
                border-top: 4px solid transparent;
                border-bottom: 4px solid transparent;
            }}

            #tooltip {{
                position: absolute;
                padding: 12px;
                background: rgba(0,0,0,0.9);
                color: white;
                border-radius: 6px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 12px;
                max-width: 300px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}

            .stats-box {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}

            .stat-item {{
                display: flex;
                justify-content: space-between;
                padding: 5px 0;
                border-bottom: 1px solid #e9ecef;
            }}

            .stat-item:last-child {{
                border-bottom: none;
            }}

            .stat-label {{
                color: #495057;
                font-size: 13px;
            }}

            .stat-value {{
                font-weight: bold;
                color: #212529;
            }}

            .clear-button {{
                width: 100%;
                padding: 10px;
                background: #868e96;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                transition: background 0.3s;
            }}

            .clear-button:hover {{
                background: #495057;
            }}

            .pattern-legend {{
                display: none;
                margin-top: 10px;
                padding: 10px;
                background: #f1f3f5;
                border-radius: 6px;
                font-size: 12px;
                color: #495057;
            }}

            .pattern-section.active .pattern-legend {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Causal Subgraph Explorer</h1>

            <div class="main-panel">
                <div id="graph-container">
                    <div id="tooltip"></div>
                </div>

                <div id="control-panel">
                    <div class="stats-box">
                        <div class="stat-item">
                            <span class="stat-label">Total Nodes:</span>
                            <span class="stat-value">{node_count}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Ego Edges:</span>
                            <span class="stat-value">{ego_edge_count}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Inter-Node Edges:</span>
                            <span class="stat-value">{inter_edge_count}</span>
                        </div>
                    </div>

                    <div class="pattern-section" data-pattern="direct">
                        <div class="pattern-header">
                            <span>Direct Effects (A → ego)</span>
                            <span class="pattern-count">{direct_count}</span>
                        </div>
                        <div class="pattern-diagram">
                            <div class="mini-graph">
                                <div class="mini-node">A</div>
                                <div class="mini-arrow"></div>
                                <div class="mini-node ego">ego</div>
                            </div>
                        </div>
                        <div class="pattern-list" id="direct-list"></div>
                        <div class="pattern-legend">
                            Direct causal effect from node A to ego vehicle
                        </div>
                    </div>

                    <div class="pattern-section" data-pattern="chain">
                        <div class="pattern-header">
                            <span>Chains (A → B → ego)</span>
                            <span class="pattern-count">{chain_count}</span>
                        </div>
                        <div class="pattern-diagram">
                            <div class="mini-graph">
                                <div class="mini-node">A</div>
                                <div class="mini-arrow"></div>
                                <div class="mini-node">B</div>
                                <div class="mini-arrow"></div>
                                <div class="mini-node ego">ego</div>
                            </div>
                        </div>
                        <div class="pattern-list" id="chain-list"></div>
                        <div class="pattern-legend">
                            Indirect effect through intermediate node B
                        </div>
                    </div>

                    <div class="pattern-section" data-pattern="fork">
                        <div class="pattern-header">
                            <span>Forks (ego ← C → B)</span>
                            <span class="pattern-count">{fork_count}</span>
                        </div>
                        <div class="pattern-diagram">
                            <div class="mini-graph">
                                <div class="mini-node ego">ego</div>
                                <div class="mini-arrow" style="transform: rotate(180deg)"></div>
                                <div class="mini-node">C</div>
                                <div class="mini-arrow"></div>
                                <div class="mini-node">B</div>
                            </div>
                        </div>
                        <div class="pattern-list" id="fork-list"></div>
                        <div class="pattern-legend">
                            Common cause C affects both ego and B
                        </div>
                    </div>

                    <div class="pattern-section" data-pattern="collider">
                        <div class="pattern-header">
                            <span>Colliders (A → ego ← B)</span>
                            <span class="pattern-count">{collider_count}</span>
                        </div>
                        <div class="pattern-diagram">
                            <div class="mini-graph">
                                <div class="mini-node">A</div>
                                <div class="mini-arrow"></div>
                                <div class="mini-node ego">ego</div>
                                <div class="mini-arrow" style="transform: rotate(180deg)"></div>
                                <div class="mini-node">B</div>
                            </div>
                        </div>
                        <div class="pattern-list" id="collider-list"></div>
                        <div class="pattern-legend">
                            Ego is affected by multiple independent causes
                        </div>
                    </div>

                    <button class="clear-button" onclick="clearHighlights()">Clear All Highlights</button>
                </div>
            </div>
        </div>

        <script>
            // Graph data
            const graphData = {graph_data};
            const subgraphData = {subgraph_data};

            // Process nodes
            const nodes = graphData.nodes.map((node, i) => ({{
                ...node,
                index: i,
                x: null,
                y: null
            }}));

            const nodeMap = {{}};
            nodes.forEach(node => {{
                nodeMap[node.id] = node;
            }});

            // Process edges
            const links = [];

            // Add ego edges
            if (graphData.ego_edges) {{
                graphData.ego_edges.forEach(edge => {{
                    if (nodeMap[edge.source] && nodeMap[edge.target]) {{
                        links.push({{
                            source: nodeMap[edge.source],
                            target: nodeMap[edge.target],
                            type: 'ego',
                            effect: edge.effect_type || '',
                            active: edge.active !== false
                        }});
                    }}
                }});
            }}

            // Add inter-node edges
            if (graphData.inter_node_edges) {{
                graphData.inter_node_edges.forEach(edge => {{
                    if (nodeMap[edge.source] && nodeMap[edge.target]) {{
                        links.push({{
                            source: nodeMap[edge.source],
                            target: nodeMap[edge.target],
                            type: 'inter',
                            relation: edge.relation || ''
                        }});
                    }}
                }});
            }}

            // Setup SVG
            const container = d3.select("#graph-container");
            const rect = container.node().getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;

            const svg = container.append("svg")
                .attr("width", width)
                .attr("height", height);

            const g = svg.append("g");

            // Zoom behavior
            const zoom = d3.zoom()
                .extent([[0, 0], [width, height]])
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => {{
                    g.attr("transform", event.transform);
                }});

            svg.call(zoom);

            // Arrow markers
            const defs = svg.append("defs");

            ["arrow-ego", "arrow-inter"].forEach(id => {{
                defs.append("marker")
                    .attr("id", id)
                    .attr("viewBox", "0 -5 10 10")
                    .attr("refX", 25)
                    .attr("refY", 0)
                    .attr("markerWidth", 6)
                    .attr("markerHeight", 6)
                    .attr("orient", "auto")
                    .append("path")
                    .attr("d", "M0,-5L10,0L0,5")
                    .attr("fill", id === "arrow-ego" ? "#e03131" : "#1971c2");
            }});

            // Force simulation
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(120))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(35));

            // Create links
            const link = g.append("g")
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("class", d => {{
                    let classes = `link link-${{d.type}}`;
                    if (d.type === 'ego') {{
                        classes += ` link-${{d.active ? 'active' : 'latent'}}`;
                    }}
                    return classes;
                }})
                .attr("stroke-width", d => d.type === 'ego' ? 3 : 2);

            // Create nodes
            const node = g.append("g")
                .selectAll("circle")
                .data(nodes)
                .enter().append("circle")
                .attr("r", d => d.id === "ego" ? 25 : 20)
                .attr("class", d => {{
                    const typeClass = {{
                        'EGO_VEHICLE': 'node-ego',
                        'VULNERABLE_ROAD_USER': 'node-vru',
                        'OTHER_VEHICLE': 'node-vehicle',
                        'TRAFFIC_CONTROL': 'node-traffic',
                        'ROAD_BOUNDARY_GEOMETRY': 'node-road',
                        'ENVIRONMENTAL_CONDITION': 'node-env',
                        'NAVIGATION_COMMAND': 'node-nav',
                        'ROAD_OBSTACLE': 'node-obstacle',
                        'EMERGENCY_VEHICLE_ACTIVE': 'node-emergency'
                    }};
                    return `node ${{typeClass[d.type] || 'node-vehicle'}}`;
                }})
                .call(drag(simulation));

            // Node labels
            const nodeLabels = g.append("g")
                .selectAll("text")
                .data(nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .text(d => d.id);

            // Tooltip
            const tooltip = d3.select("#tooltip");

            node.on("mouseenter", function(event, d) {{
                const pageX = event.pageX || event.clientX;
                const pageY = event.pageY || event.clientY;

                tooltip.style("opacity", 1)
                    .style("left", (pageX + 10) + "px")
                    .style("top", (pageY - 10) + "px")
                    .html(`
                        <strong>${{d.label || d.id}}</strong><br/>
                        Type: ${{d.type || 'Unknown'}}<br/>
                        ${{d.state ? `State: ${{d.state}}<br/>` : ''}}
                        ${{d.position_relative_to_ego ? `Position: ${{d.position_relative_to_ego}}<br/>` : ''}}
                    `);
            }})
            .on("mouseleave", function() {{
                tooltip.style("opacity", 0);
            }});

            // Update positions
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                nodeLabels
                    .attr("x", d => d.x)
                    .attr("y", d => d.y + 4);
            }});

            // Drag behavior
            function drag(simulation) {{
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}

                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}

                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}

                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
            }}

            // Populate subgraph lists
            function populateSubgraphLists() {{
                // Direct patterns
                const directList = document.getElementById('direct-list');
                subgraphData.direct.forEach((pattern, idx) => {{
                    const item = document.createElement('div');
                    item.className = 'subgraph-item';
                    item.dataset.pattern = 'direct';
                    item.dataset.index = idx;
                    item.dataset.nodes = JSON.stringify(pattern);
                    item.textContent = pattern.join(' → ');
                    item.onclick = () => highlightSubgraph(pattern, 'direct');
                    directList.appendChild(item);
                }});

                // Chain patterns
                const chainList = document.getElementById('chain-list');
                subgraphData.chain.forEach((pattern, idx) => {{
                    const item = document.createElement('div');
                    item.className = 'subgraph-item';
                    item.dataset.pattern = 'chain';
                    item.dataset.index = idx;
                    item.dataset.nodes = JSON.stringify(pattern);
                    item.textContent = pattern.join(' → ');
                    item.onclick = () => highlightSubgraph(pattern, 'chain');
                    chainList.appendChild(item);
                }});

                // Fork patterns
                const forkList = document.getElementById('fork-list');
                subgraphData.fork.forEach((pattern, idx) => {{
                    const item = document.createElement('div');
                    item.className = 'subgraph-item';
                    item.dataset.pattern = 'fork';
                    item.dataset.index = idx;
                    item.dataset.nodes = JSON.stringify(pattern);
                    item.textContent = `${{pattern[0]}} ← ${{pattern[1]}} → ${{pattern[2]}}`;
                    item.onclick = () => highlightSubgraph(pattern, 'fork');
                    forkList.appendChild(item);
                }});

                // Collider patterns
                const colliderList = document.getElementById('collider-list');
                subgraphData.collider.forEach((pattern, idx) => {{
                    const item = document.createElement('div');
                    item.className = 'subgraph-item';
                    item.dataset.pattern = 'collider';
                    item.dataset.index = idx;
                    item.dataset.nodes = JSON.stringify(pattern);
                    item.textContent = `${{pattern[0]}} → ${{pattern[1]}} ← ${{pattern[2]}}`;
                    item.onclick = () => highlightSubgraph(pattern, 'collider');
                    colliderList.appendChild(item);
                }});
            }}

            // Highlight subgraph
            function highlightSubgraph(nodeIds, patternType) {{
                // Clear previous highlights
                clearHighlights();

                // Mark this item as selected
                event.target.classList.add('selected');

                // Create a set of nodes in the subgraph
                const subgraphNodes = new Set(nodeIds);

                // Highlight nodes
                node.classed('highlighted', d => subgraphNodes.has(d.id))
                    .classed('dimmed', d => !subgraphNodes.has(d.id));

                // Highlight edges based on pattern type
                link.classed('highlighted', d => {{
                    if (patternType === 'direct') {{
                        return d.source.id === nodeIds[0] && d.target.id === nodeIds[1];
                    }} else if (patternType === 'chain') {{
                        return (d.source.id === nodeIds[0] && d.target.id === nodeIds[1]) ||
                               (d.source.id === nodeIds[1] && d.target.id === nodeIds[2]);
                    }} else if (patternType === 'fork') {{
                        return (d.source.id === nodeIds[1] && d.target.id === nodeIds[0]) ||
                               (d.source.id === nodeIds[1] && d.target.id === nodeIds[2]);
                    }} else if (patternType === 'collider') {{
                        return (d.source.id === nodeIds[0] && d.target.id === nodeIds[1]) ||
                               (d.source.id === nodeIds[2] && d.target.id === nodeIds[1]);
                    }}
                    return false;
                }})
                .classed('dimmed', function(d) {{
                    return !d3.select(this).classed('highlighted');
                }});

                // Activate pattern section
                document.querySelectorAll('.pattern-section').forEach(section => {{
                    section.classList.remove('active');
                }});
                document.querySelector(`[data-pattern="${{patternType}}"]`).classList.add('active');
            }}

            // Clear highlights
            function clearHighlights() {{
                node.classed('highlighted', false)
                    .classed('dimmed', false);
                link.classed('highlighted', false)
                    .classed('dimmed', false);
                document.querySelectorAll('.subgraph-item').forEach(item => {{
                    item.classList.remove('selected');
                }});
                document.querySelectorAll('.pattern-section').forEach(section => {{
                    section.classList.remove('active');
                }});
            }}

            // Pattern section toggle
            document.querySelectorAll('.pattern-header').forEach(header => {{
                header.addEventListener('click', function(e) {{
                    if (e.target.classList.contains('pattern-count')) return;
                    const section = this.parentElement;
                    section.classList.toggle('active');
                }});
            }});

            // Initialize
            populateSubgraphLists();
        </script>
    </body>
    </html>
    """

    # Format the HTML with actual data
    html_content = html_template.format(
        node_count=len(graph_data.get('nodes', [])),
        ego_edge_count=len(graph_data.get('ego_edges', [])),
        inter_edge_count=len(graph_data.get('inter_node_edges', [])),
        direct_count=len(subgraphs.get('direct', [])),
        chain_count=len(subgraphs.get('chain', [])),
        fork_count=len(subgraphs.get('fork', [])),
        collider_count=len(subgraphs.get('collider', [])),
        graph_data=json.dumps(graph_data),
        subgraph_data=json.dumps(subgraphs)
    )

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✅ Subgraph visualization saved to: {output_file}")
    return True


def extract_subgraphs_from_data(graph_data: Dict) -> Dict:
    """
    Extract subgraphs from graph data using the logic from subgraph_extraction.py
    """
    from collections import defaultdict
    import itertools

    # Build adjacency lists
    children = defaultdict(set)
    parents = defaultdict(set)

    for edge in graph_data.get('ego_edges', []):
        children[edge['source']].add(edge['target'])
        parents[edge['target']].add(edge['source'])

    for edge in graph_data.get('inter_node_edges', []):
        children[edge['source']].add(edge['target'])
        parents[edge['target']].add(edge['source'])

    results = {"direct": [], "chain": [], "fork": [], "collider": []}
    ego_id = "ego"

    # Direct: A -> ego
    for a in parents.get(ego_id, []):
        results["direct"].append([a, ego_id])

    # Chain: A -> B -> ego
    for b in parents.get(ego_id, []):
        for a in parents.get(b, []):
            if a != ego_id:
                results["chain"].append([a, b, ego_id])

    # Fork: ego <- C -> B
    for c in parents.get(ego_id, []):
        for b in children.get(c, []):
            if b != ego_id:
                results["fork"].append([ego_id, c, b])

    # Collider: A -> ego <- B
    ego_parents = list(parents.get(ego_id, []))
    for a, b in itertools.combinations(ego_parents, 2):
        results["collider"].append([a, ego_id, b])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize extracted causal subgraphs with interactive highlighting"
    )
    parser.add_argument(
        "input_file",
        help="Input JSON file with causal graph data"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output HTML file path (default: input_name_subgraphs.html)"
    )
    parser.add_argument(
        "--subgraphs", "-s",
        help="Optional JSON file with pre-extracted subgraphs"
    )

    args = parser.parse_args()

    # Load graph data
    try:
        with open(args.input_file, 'r') as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading graph data: {e}")
        return 1

    # Load or extract subgraphs
    if args.subgraphs:
        try:
            with open(args.subgraphs, 'r') as f:
                subgraphs = json.load(f)
        except Exception as e:
            print(f"❌ Error loading subgraph data: {e}")
            return 1
    else:
        print("📊 Extracting subgraphs from graph data...")
        subgraphs = extract_subgraphs_from_data(graph_data)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = Path(args.input_file).stem
        output_file = f"{base_name}_subgraphs.html"

    # Create visualization
    success = create_subgraph_visualization(graph_data, subgraphs, output_file)

    if success:
        print(f"📊 Summary of extracted subgraphs:")
        print(f"  - Direct effects: {len(subgraphs['direct'])}")
        print(f"  - Chain patterns: {len(subgraphs['chain'])}")
        print(f"  - Fork patterns: {len(subgraphs['fork'])}")
        print(f"  - Collider patterns: {len(subgraphs['collider'])}")
        print(f"\n🌐 Open {output_file} in a web browser to explore the subgraphs!")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())