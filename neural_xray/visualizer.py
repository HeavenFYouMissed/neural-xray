"""
visualizer.py — Neural Concept Visualizer (AutoPsy VISUALIZE stage)
====================================================================

Generates a self-contained interactive HTML file that visualizes the
model's internal concept geometry as a living, animated graph.

The "strings" metaphor:
    Each concept is a glowing node.
    Each co-activation relationship is a curved string (bezier arc).
    When you replay a trace, the strings light up layer by layer —
    exactly like watching dye flow through capillaries.

Features:
    - Force-directed concept graph (D3.js, no server needed)
    - Curved "string" edges, glow intensity = co-activation strength
    - Click any node → see its connections and stats in the panel
    - Select a concept from the dropdown → replay its trace as animation
    - Each frame = one model layer, dominant concept glows orange,
      co-active concepts glow cyan, their connecting strings pulse
    - Timeline bar at the bottom tracks which layer we're at
    - Zoom / pan / drag nodes to rearrange
    - Node size = influence radius (how many concepts it co-activates)
    - Slider to show top N concepts (useful when you have 1000+)

Usage (Python):
    from antroslammer.autopsy.visualizer import NeuralVisualizer

    viz = NeuralVisualizer("autopsy_output/")
    path = viz.generate()
    print(f"Open in browser: {path}")

Usage (CLI):
    python autopsy.py --blueprint autopsy_output/blueprint_xxx/ \\
        --source Qwen/Qwen2-1.5B --stage trace --trace-all
    python autopsy.py --output autopsy_output/ --visualize
"""

import json
import math
from pathlib import Path
from typing import Dict


# ─── HTML Template ────────────────────────────────────────────────────────────
# All trace data is embedded via __JSON_DATA__ replacement (no server needed)

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AntrogenSlammer — Neural Concept Map</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #050510; color: #c8c8ff;
  font-family: 'Courier New', monospace;
  display: flex; height: 100vh; overflow: hidden;
  user-select: none;
}
#graph-wrap { flex: 1; position: relative; overflow: hidden; }
svg { width: 100%; height: 100%; display: block; cursor: grab; }
svg:active { cursor: grabbing; }

/* ── Strings (edges) ──────────────────────────────────────────────── */
.link { fill: none; stroke-linecap: round; }
@keyframes string-pulse {
  0%, 100% { stroke-opacity: 0.85; }
  50%       { stroke-opacity: 1.0; }
}
.link.lit { animation: string-pulse 0.5s ease-in-out infinite; }

/* ── Nodes ────────────────────────────────────────────────────────── */
.node circle { cursor: pointer; }
.node text   { pointer-events: none; }

/* ── Right panel ──────────────────────────────────────────────────── */
#panel {
  width: 300px; min-width: 260px; max-width: 360px;
  background: #07070f; border-left: 1px solid #181830;
  display: flex; flex-direction: column; overflow: hidden;
}
#panel-header { padding: 12px 14px 10px; border-bottom: 1px solid #181830; background: #050510; }
#panel-header h1 { font-size: 13px; color: #00ffcc; letter-spacing: 1px; margin-bottom: 3px; }
#panel-header .meta { font-size: 10px; color: #3a3a5a; }

#controls {
  padding: 10px 14px; border-bottom: 1px solid #181830;
  display: flex; flex-direction: column; gap: 8px;
}
#controls label { font-size: 10px; color: #4455aa; margin-bottom: 2px; display: block; }
select, button {
  width: 100%; background: #0a0a1a; border: 1px solid #222244;
  color: #c8c8ff; padding: 5px 8px; font-family: inherit; font-size: 11px;
  cursor: pointer; border-radius: 2px;
}
select { appearance: none; }
button { transition: all 0.15s; }
button:hover { border-color: #00ffcc; color: #00ffcc; background: #001a14; }
button.on    { border-color: #ff6600; color: #ff6600; background: #1a0a00; }
#play-row { display: flex; gap: 6px; }
#play-row button { flex: 1; }
input[type=range] { width: 100%; accent-color: #00ffcc; cursor: pointer; }

#info { flex: 1; padding: 12px 14px; overflow-y: auto; font-size: 11px; }
#info h2 { font-size: 12px; color: #00ffcc; margin-bottom: 8px; }
.row { display: flex; justify-content: space-between; margin-bottom: 4px; }
.key { color: #3a4a7a; }
.val { color: #aabbff; text-align: right; max-width: 180px; overflow: hidden;
       text-overflow: ellipsis; white-space: nowrap; }
.section { margin-top: 10px; padding-top: 10px; border-top: 1px solid #181830; }

#chain-display {
  font-size: 10px; color: #334; line-height: 1.7; margin-top: 5px; word-break: break-word;
}
#chain-display .cd { color: #00ffcc; font-weight: bold; }
#chain-display .cs { color: #ff6600; }

#timeline {
  height: 66px; padding: 8px 14px 6px;
  border-top: 1px solid #181830; position: relative; overflow: hidden;
}
#tl-label { font-size: 9px; color: #2a2a4a; margin-bottom: 4px; }
#lbars { display: flex; align-items: flex-end; height: 36px; gap: 1px; }
.lbar {
  flex: 0 0 auto; background: #181830; border-radius: 1px;
  transition: background 0.08s, height 0.08s;
}
.lbar.past   { background: #1e2244; }
.lbar.active { background: #00ffcc; box-shadow: 0 0 5px #00ffcc; }

#tooltip {
  position: fixed; background: #0a0a1f; border: 1px solid #00ffcc33;
  color: #c8d8ff; font-size: 11px; padding: 5px 9px; border-radius: 3px;
  pointer-events: none; display: none; max-width: 220px; z-index: 100;
  white-space: pre-line;
}
#legend {
  position: absolute; bottom: 12px; left: 12px;
  font-size: 10px; color: #2a2a44; line-height: 2;
}
.ldot {
  display: inline-block; width: 9px; height: 9px;
  border-radius: 50%; margin-right: 5px; vertical-align: middle;
}
#info::-webkit-scrollbar { width: 4px; }
#info::-webkit-scrollbar-track { background: #050510; }
#info::-webkit-scrollbar-thumb { background: #222244; border-radius: 2px; }
</style>
</head>
<body>

<div id="graph-wrap">
  <svg id="svg">
    <defs>
      <filter id="glow-sm" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2.5" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="glow-lg" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="6" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <g id="zg">
      <g id="ll"></g>
      <g id="nl"></g>
    </g>
  </svg>
  <div id="legend">
    <span class="ldot" style="background:#1e3060"></span>Entity
    &nbsp;&nbsp;<span class="ldot" style="background:#1e4022"></span>Relation
    &nbsp;&nbsp;<span class="ldot" style="background:#00ffcc"></span>Active
    &nbsp;&nbsp;<span class="ldot" style="background:#ff6600"></span>Dominant
  </div>
</div>

<div id="panel">
  <div id="panel-header">
    <h1>&#9672; NEURAL CONCEPT MAP</h1>
    <div class="meta" id="meta-line">Loading…</div>
  </div>

  <div id="controls">
    <div>
      <label>TRACE CONCEPT (select to animate)</label>
      <select id="tsel"><option value="">— select concept —</option></select>
    </div>
    <div id="play-row">
      <button id="btn-play">&#9654; PLAY</button>
      <button id="btn-reset">&#9198; RESET</button>
      <button id="btn-speed">SPEED: 1&#215;</button>
    </div>
    <div>
      <label>SHOW TOP <span id="nc-label">—</span> CONCEPTS</label>
      <input type="range" id="nslider" min="10" max="500" value="200">
    </div>
  </div>

  <div id="info">
    <div id="no-sel" style="color:#2a2a44;margin-top:24px;text-align:center;line-height:2">
      Click any concept node<br>to explore its connections
    </div>
    <div id="sel-panel" style="display:none">
      <h2 id="sel-name">—</h2>
      <div class="row"><span class="key">Influence radius</span><span class="val" id="s-inf">—</span></div>
      <div class="row"><span class="key">String connections</span><span class="val" id="s-conn">—</span></div>
      <div class="row"><span class="key">Has trace</span><span class="val" id="s-tr">—</span></div>
      <div class="section" id="chain-sec" style="display:none">
        <div class="key">Dominant chain:</div>
        <div id="chain-display"></div>
      </div>
      <div class="section" id="shifts-sec" style="display:none">
        <div class="key">Concept shift points:</div>
        <div id="shifts-disp" style="margin-top:5px;font-size:10px;color:#aabbff;line-height:1.8"></div>
      </div>
      <div class="section">
        <div class="key">Top co-activations:</div>
        <div id="conn-list" style="margin-top:6px"></div>
      </div>
    </div>
  </div>

  <div id="timeline">
    <div id="tl-label">LAYER TIMELINE — <span id="lpos">—</span></div>
    <div id="lbars"></div>
  </div>
</div>

<div id="tooltip"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = __JSON_DATA__;

// ── State ──────────────────────────────────────────────────────────────────
let selConcept = null, curTrace = null, curFrame = 0;
let animTimer = null, playing = false, speedIdx = 0;
const SPEEDS = [1, 2, 4, 0.5], SLABELS = ['1×', '2×', '4×', '½×'];
let maxNodes = 200;

const meta   = DATA.meta;
const traces = DATA.traces;
let allNodes = DATA.nodes;
let allLinks = DATA.links;

// ── Meta ───────────────────────────────────────────────────────────────────
document.getElementById('meta-line').textContent =
  (meta.model ? meta.model + ' · ' : '') +
  meta.concept_count + ' concepts · ' +
  meta.trace_count + ' traces' +
  (meta.has_sae ? ' · SAE mode' : '');

// ── Trace selector ─────────────────────────────────────────────────────────
const tsel = document.getElementById('tsel');
Object.keys(traces).sort().forEach(t => {
  const o = document.createElement('option'); o.value = o.textContent = t; tsel.appendChild(o);
});
tsel.addEventListener('change', () => { if (tsel.value) selectConcept(tsel.value); });

// ── D3 setup ───────────────────────────────────────────────────────────────
const svgEl  = d3.select('#svg');
const zg     = svgEl.select('#zg');
const ll     = zg.select('#ll');   // links layer
const nl     = zg.select('#nl');   // nodes layer
let W = svgEl.node().clientWidth, H = svgEl.node().clientHeight;

const zoom = d3.zoom().scaleExtent([0.05, 10])
  .on('zoom', e => zg.attr('transform', e.transform));
svgEl.call(zoom);

// Color helpers
function nodeColor(d)   { return d.group === 1 ? '#112211' : '#101128'; }
function nodeStroke(d)  { return d.group === 1 ? '#1e4022' : '#1e3060'; }
function edgeBase(val)  { return d3.interpolateRgb('#0d0d22', '#00ffcc')(Math.min(val * 1.8, 1)); }
function hexScore(v)    {
  if (v > 0.7) return '00ffcc'; if (v > 0.4) return '88aa44'; return '445566';
}

// ── Graph build ────────────────────────────────────────────────────────────
let sim, linkSel, nodeSel;

function buildGraph() {
  const top = allNodes.slice(0, maxNodes);
  const ids = new Set(top.map(n => n.id));
  const flinks = allLinks.filter(l => {
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return ids.has(s) && ids.has(t);
  });

  if (sim) sim.stop();
  ll.selectAll('*').remove();
  nl.selectAll('*').remove();

  // ── Strings ──────────────────────────────────────────────────────────────
  linkSel = ll.selectAll('path')
    .data(flinks, d => (d.source.id||d.source) + '|' + (d.target.id||d.target))
    .join('path')
    .attr('class', 'link')
    .style('stroke', d => edgeBase(d.value))
    .style('stroke-width', d => Math.max(0.4, d.value * 2.5))
    .style('stroke-opacity', d => 0.1 + d.value * 0.35)
    .on('mouseenter', (e, d) => showTip(e,
      (d.source.id||d.source) + ' ↔ ' + (d.target.id||d.target) +
      '\nco-activation: ' + (d.value * 100).toFixed(1) + '%'))
    .on('mouseleave', hideTip)
    .on('click', (e, d) => selectConcept(d.source.id || d.source));

  // ── Nodes ─────────────────────────────────────────────────────────────────
  nodeSel = nl.selectAll('g.node')
    .data(top, d => d.id)
    .join('g').attr('class', 'node')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e, d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }))
    .on('click',      (e, d) => selectConcept(d.id))
    .on('mouseenter', (e, d) => showTip(e, d.id + '\ninfluence: ' + d.influence))
    .on('mouseleave', hideTip);

  nodeSel.append('circle')
    .attr('r', d => d.radius)
    .attr('fill', d => nodeColor(d))
    .attr('stroke', d => nodeStroke(d))
    .attr('stroke-width', 1);

  nodeSel.append('text')
    .attr('dy', '0.35em').attr('text-anchor', 'middle')
    .style('fill', '#444466')
    .style('font-size', d => Math.min(11, Math.max(7, d.radius - 1)) + 'px')
    .style('opacity', d => d.radius > 7 ? 1 : 0)
    .text(d => d.id);

  // ── Force simulation ──────────────────────────────────────────────────────
  sim = d3.forceSimulation(top)
    .force('link',    d3.forceLink(flinks).id(d => d.id).distance(d => 50 + 70 * (1 - d.value)))
    .force('charge',  d3.forceManyBody().strength(d => -60 - d.radius * 5))
    .force('center',  d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide(d => d.radius + 9))
    .on('tick', tick);
}

function tick() {
  // Bezier arc "string" paths
  linkSel.attr('d', d => {
    const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
    const dx = tx - sx, dy = ty - sy;
    const dr = Math.sqrt(dx*dx + dy*dy) * 0.65;
    if (dr < 1) return 'M' + sx + ',' + sy + 'L' + tx + ',' + ty;
    return 'M' + sx + ',' + sy + 'A' + dr + ',' + dr + ' 0 0,1 ' + tx + ',' + ty;
  });
  nodeSel.attr('transform', d => 'translate(' + d.x + ',' + d.y + ')');
}

// ── Select concept ─────────────────────────────────────────────────────────
function selectConcept(id) {
  stopAnim();
  selConcept = id;
  if (tsel.value !== id) tsel.value = id;

  const nd    = allNodes.find(n => n.id === id);
  const trace = traces[id];

  document.getElementById('no-sel').style.display   = 'none';
  document.getElementById('sel-panel').style.display = '';
  document.getElementById('sel-name').textContent    = id;
  document.getElementById('s-inf').textContent       = nd ? nd.influence : '—';
  document.getElementById('s-tr').textContent        = trace ? '✓ yes' : '✗ no';

  const linkCount = allLinks.filter(l =>
    (l.source.id||l.source) === id || (l.target.id||l.target) === id
  ).length;
  document.getElementById('s-conn').textContent = linkCount;

  // Dominant chain
  const cSec = document.getElementById('chain-sec');
  if (trace && trace.chain && trace.chain.length) {
    cSec.style.display = '';
    const comp = [], seen = [];
    for (const c of trace.chain) { if (c !== seen[seen.length-1]) { comp.push(c); seen.push(c); } }
    document.getElementById('chain-display').innerHTML = comp.map(c =>
      c === id
        ? '<span class="cd">' + c + '</span>'
        : '<span>' + c + '</span>'
    ).join(' <span style="color:#223">→</span> ');
  } else { cSec.style.display = 'none'; }

  // Shift points
  const sSec = document.getElementById('shifts-sec');
  if (trace && trace.shifts && trace.shifts.length) {
    sSec.style.display = '';
    document.getElementById('shifts-disp').innerHTML = trace.shifts.map(s =>
      'Layer ' + s.layer + ': <span style="color:#00ffcc">' + s.from + '</span>' +
      ' → <span style="color:#ff6600">' + s.to + '</span>'
    ).join('<br>');
  } else { sSec.style.display = 'none'; }

  // Top co-activations
  const coLinks = allLinks
    .filter(l => (l.source.id||l.source) === id || (l.target.id||l.target) === id)
    .map(l => ({
      other: (l.source.id||l.source) === id ? (l.target.id||l.target) : (l.source.id||l.source),
      v: l.value
    }))
    .sort((a,b) => b.v - a.v).slice(0, 12);

  document.getElementById('conn-list').innerHTML = coLinks.map(c =>
    '<div style="display:flex;justify-content:space-between;margin-bottom:3px">' +
    '<span style="color:#667799;cursor:pointer" onclick="selectConcept(\'' + c.other + '\')">' + c.other + '</span>' +
    '<span style="color:#' + hexScore(c.v) + '">' + (c.v * 100).toFixed(0) + '%</span>' +
    '</div>'
  ).join('') || '<span style="color:#222244">no connections in view</span>';

  highlightNode(id);

  if (trace) { curTrace = trace; curFrame = 0; buildTimeline(trace.frames.length); }
}

// ── Highlight node and its strings ─────────────────────────────────────────
function highlightNode(id) {
  if (!nodeSel || !linkSel) return;
  const connIds = new Set([id]);
  linkSel.each(d => {
    const s = d.source.id||d.source, t = d.target.id||d.target;
    if (s === id) connIds.add(t);
    if (t === id) connIds.add(s);
  });

  nodeSel.select('circle')
    .attr('fill',         d => d.id === id ? '#002211' : connIds.has(d.id) ? '#101828' : '#080814')
    .attr('stroke',       d => d.id === id ? '#00ffcc' : connIds.has(d.id) ? '#2244aa' : '#111128')
    .attr('stroke-width', d => d.id === id ? 2.5 : connIds.has(d.id) ? 1.5 : 0.5)
    .attr('filter',       d => d.id === id ? 'url(#glow-lg)' : connIds.has(d.id) ? 'url(#glow-sm)' : null);

  nodeSel.select('text')
    .style('fill',     d => d.id === id ? '#ffffff' : connIds.has(d.id) ? '#99aacc' : '#2a2a44')
    .style('font-size',d => d.id === id ? '13px' : null)
    .style('opacity',  d => d.id === id || connIds.has(d.id) || d.radius > 7 ? 1 : 0);

  linkSel
    .style('stroke',         d => { const s=d.source.id||d.source,t=d.target.id||d.target; return (s===id||t===id)?'#00ffcc':edgeBase(d.value); })
    .style('stroke-width',   d => { const s=d.source.id||d.source,t=d.target.id||d.target; return (s===id||t===id)?Math.max(1.5,d.value*4):Math.max(0.4,d.value*2.5); })
    .style('stroke-opacity', d => { const s=d.source.id||d.source,t=d.target.id||d.target; return (s===id||t===id)?0.85:0.06; })
    .classed('lit', d => { const s=d.source.id||d.source,t=d.target.id||d.target; return s===id||t===id; });
}

// ── Trace animation ────────────────────────────────────────────────────────
function applyFrame(frame) {
  if (!nodeSel || !linkSel) return;
  const active = new Set(frame.a.map(a => a[0]));
  const dom    = frame.d;

  nodeSel.select('circle')
    .attr('fill',         d => d.id===dom?'#220800':active.has(d.id)?'#001a1a':'#07070f')
    .attr('stroke',       d => d.id===dom?'#ff6600':active.has(d.id)?'#00ffcc':'#111128')
    .attr('stroke-width', d => d.id===dom?3:active.has(d.id)?2:0.5)
    .attr('filter',       d => d.id===dom?'url(#glow-lg)':active.has(d.id)?'url(#glow-sm)':null);

  nodeSel.select('text')
    .style('fill',    d => d.id===dom?'#ffaa44':active.has(d.id)?'#88ffee':'#2a2a44')
    .style('opacity', d => d.id===dom||active.has(d.id)||d.radius>7?1:0);

  linkSel
    .classed('lit', d => {
      const s=d.source.id||d.source, t=d.target.id||d.target;
      return active.has(s) && active.has(t);
    })
    .style('stroke',         d => { const s=d.source.id||d.source,t=d.target.id||d.target; return active.has(s)&&active.has(t)?'#00ffcc':edgeBase(d.value); })
    .style('stroke-width',   d => { const s=d.source.id||d.source,t=d.target.id||d.target; return active.has(s)&&active.has(t)?Math.max(2,d.value*5):Math.max(0.4,d.value*2.5); })
    .style('stroke-opacity', d => { const s=d.source.id||d.source,t=d.target.id||d.target; return active.has(s)&&active.has(t)?0.9:0.05; });

  document.getElementById('lpos').textContent = 'Layer ' + frame.l + ' — ' + frame.t + ' — ' + dom;
  updateTL(curFrame, curTrace.frames.length);
}

function playTrace() {
  if (!curTrace) return;
  stopAnim(false);
  const delay = Math.round(200 / SPEEDS[speedIdx]);
  animTimer = setInterval(() => {
    if (curFrame >= curTrace.frames.length) { stopAnim(); return; }
    applyFrame(curTrace.frames[curFrame++]);
  }, delay);
  document.getElementById('btn-play').textContent = '⏸ PAUSE';
  document.getElementById('btn-play').classList.add('on');
  playing = true;
}

function stopAnim(rst=true) {
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  if (rst) {
    playing = false;
    document.getElementById('btn-play').textContent = '▶ PLAY';
    document.getElementById('btn-play').classList.remove('on');
  }
}

document.getElementById('btn-play').addEventListener('click', () => {
  if (playing) { stopAnim(); } else { playTrace(); }
});
document.getElementById('btn-reset').addEventListener('click', () => {
  stopAnim(); curFrame = 0;
  if (selConcept) highlightNode(selConcept);
  resetTL();
});
document.getElementById('btn-speed').addEventListener('click', () => {
  speedIdx = (speedIdx + 1) % SPEEDS.length;
  document.getElementById('btn-speed').textContent = 'SPEED: ' + SLABELS[speedIdx];
  if (playing) { stopAnim(false); playTrace(); }
});

// ── Timeline ───────────────────────────────────────────────────────────────
let lbars = [];
function buildTimeline(n) {
  const c = document.getElementById('lbars'); c.innerHTML = ''; lbars = [];
  const bw = Math.max(2, Math.min(7, Math.floor((c.offsetWidth - n) / n)));
  for (let i = 0; i < n; i++) {
    const b = document.createElement('div');
    b.className = 'lbar'; b.style.width = bw + 'px'; b.style.height = '10px';
    c.appendChild(b); lbars.push(b);
  }
  document.getElementById('lpos').textContent = n + ' layers';
}
function updateTL(frame, total) {
  lbars.forEach((b, i) => {
    b.className = 'lbar' + (i===frame-1?' active':i<frame?' past':'');
    b.style.height = (i===frame-1?30:i<frame?14:10) + 'px';
  });
}
function resetTL() {
  lbars.forEach(b => { b.className='lbar'; b.style.height='10px'; });
  document.getElementById('lpos').textContent = curTrace ? curTrace.frames.length + ' layers' : '—';
}

// ── Node slider ────────────────────────────────────────────────────────────
const nsl = document.getElementById('nslider');
nsl.max   = Math.min(allNodes.length, 500);
nsl.value = maxNodes = Math.min(200, allNodes.length);
document.getElementById('nc-label').textContent = maxNodes;
nsl.addEventListener('input', () => {
  maxNodes = parseInt(nsl.value);
  document.getElementById('nc-label').textContent = maxNodes;
  buildGraph();
  if (selConcept) setTimeout(() => selectConcept(selConcept), 150);
});

// ── Tooltip ────────────────────────────────────────────────────────────────
const tip = document.getElementById('tooltip');
function showTip(e, txt) {
  tip.style.display = 'block'; tip.textContent = txt;
  tip.style.left = (e.clientX+13)+'px'; tip.style.top = (e.clientY-8)+'px';
}
function hideTip() { tip.style.display = 'none'; }
svgEl.node().addEventListener('mousemove', e => {
  if (tip.style.display==='block') {
    tip.style.left=(e.clientX+13)+'px'; tip.style.top=(e.clientY-8)+'px';
  }
});

// ── Resize ─────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  W = svgEl.node().clientWidth; H = svgEl.node().clientHeight;
  if (sim) sim.force('center', d3.forceCenter(W/2, H/2)).alpha(0.1).restart();
});

// ── Init ───────────────────────────────────────────────────────────────────
buildGraph();

if (allNodes.length === 0) {
  zg.append('text').attr('x', W/2).attr('y', H/2)
    .attr('text-anchor','middle').style('fill','#2a2a44').style('font-size','13px')
    .text('No trace data found. Run: python autopsy.py --stage trace --trace-all --output <dir>');
}
</script>
</body>
</html>"""


# ─── Visualizer class ─────────────────────────────────────────────────────────

class NeuralVisualizer:
    """
    Generate a self-contained interactive HTML visualization from autopsy output.

    Reads:
        - <output_dir>/contamination_map.json  → concept nodes + co-activation strings
        - <output_dir>/traces/*.json           → per-trace animation frames

    Writes:
        - <output_dir>/neural_viz.html         → open in any browser, no server

    Args:
        output_dir:      Directory containing contamination_map.json + traces/
        max_nodes:       Max concepts to show (default 400; perf degrades above ~600)
        min_edge_weight: Minimum co-activation fraction to draw a string (default 0.05)
    """

    def __init__(
        self,
        output_dir: str,
        max_nodes: int = 400,
        min_edge_weight: float = 0.05,
    ):
        self.output_dir = Path(output_dir)
        self.max_nodes = max_nodes
        self.min_edge_weight = min_edge_weight

    def generate(self) -> str:
        """Build data from traces, render HTML, write to disk. Returns file path."""
        data = self._build_graph_data()
        json_str = json.dumps(data, separators=(",", ":"))
        html = _HTML_TEMPLATE.replace("__JSON_DATA__", json_str)

        out_path = self.output_dir / "neural_viz.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

        n_nodes  = len(data["nodes"])
        n_links  = len(data["links"])
        n_traces = len(data["traces"])
        size_kb  = len(html) // 1024

        print(f"\n[Visualizer] Generated → {out_path}")
        print(f"  {n_nodes} concept nodes · {n_links} strings · {n_traces} animated traces · {size_kb} KB")
        print(f"\n  Open in browser:")
        print(f"    start {out_path}")
        return str(out_path)

    # ─── Internal ────────────────────────────────────────────────────────────

    def _build_graph_data(self) -> dict:
        """Load contamination_map.json + trace files, build D3-compatible data."""
        # Load contamination map
        cmap_path = self.output_dir / "contamination_map.json"
        cmap: dict = {}
        if cmap_path.exists():
            with open(cmap_path, encoding="utf-8") as f:
                cmap = json.load(f)
        else:
            print(f"[Visualizer] WARNING: {cmap_path} not found — run trace-all first")

        # Load individual trace files
        traces_dir = self.output_dir / "traces"
        trace_data: Dict[str, dict] = {}
        if traces_dir.exists():
            trace_files = sorted(traces_dir.glob("*.json"))
            for tf in trace_files[:self.max_nodes]:
                try:
                    with open(tf, encoding="utf-8") as f:
                        td = json.load(f)
                    trace_data[td["concept"]] = self._compress_trace(td)
                except Exception as e:
                    print(f"[Visualizer] WARNING: skipping {tf.name}: {e}")

        # Build concept list
        influence = cmap.get("influence_radius", {})
        all_concepts = cmap.get("concepts", list(trace_data.keys()))
        # Sort highest-influence first, take top max_nodes
        sorted_concepts = sorted(
            all_concepts, key=lambda c: influence.get(c, 0), reverse=True
        )[:self.max_nodes]
        concept_set = set(sorted_concepts)

        # CortexLang relation keywords → group 1 (rendered differently)
        _RELATIONS = {
            "causes", "causes_not", "requires", "enables", "prevents", "is_a",
            "has", "part_of", "leads_to", "blocks", "triggers", "increases",
            "decreases", "contains", "produces",
        }

        nodes = []
        for concept in sorted_concepts:
            inf    = influence.get(concept, 0)
            group  = 1 if concept.lower() in _RELATIONS else 0
            radius = max(4, min(20, 4 + math.sqrt(max(inf, 0))))
            nodes.append({
                "id":        concept,
                "radius":    round(radius, 1),
                "group":     group,
                "influence": inf,
                "has_trace": concept in trace_data,
            })

        # Build unduplicated edges from co_activation
        co_act = cmap.get("co_activation", {})
        links  = []
        seen: set = set()
        for src, targets in co_act.items():
            if src not in concept_set:
                continue
            for tgt, weight in targets.items():
                if tgt not in concept_set or weight < self.min_edge_weight:
                    continue
                key = (min(src, tgt), max(src, tgt))
                if key in seen:
                    continue
                seen.add(key)
                links.append({"source": src, "target": tgt, "value": round(weight, 3)})

        # Sort ascending so heavier edges render on top in SVG
        links.sort(key=lambda l: l["value"])

        # Detect SAE presence
        has_sae = any(self.output_dir.glob("*.pt"))

        return {
            "meta": {
                "model":         cmap.get("model_name", ""),
                "concept_count": len(sorted_concepts),
                "trace_count":   len(trace_data),
                "has_sae":       has_sae,
            },
            "nodes":  nodes,
            "links":  links,
            "traces": trace_data,
        }

    def _compress_trace(self, td: dict) -> dict:
        """Strip heavy fields, keep only what the animation needs (keeps HTML small)."""
        frames = []
        for layer in td.get("layers", []):
            frames.append({
                "l": layer["layer_index"],
                "t": layer["layer_type"],
                "d": layer["dominant"],
                "n": round(layer.get("activation_norm", 0.0), 2),
                "a": [[c, round(s, 3)] for c, s in layer.get("top_concepts", [])],
            })
        return {
            "concept": td["concept"],
            "chain":   td.get("dominant_chain", []),
            "shifts":  td.get("shift_points", []),
            "contamination_map": {
                k: v[:30] for k, v in td.get("contamination_map", {}).items()
            },
            "frames": frames,
        }
