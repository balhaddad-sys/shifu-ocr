"""ShifuMind local web interface."""

import io
import json
from flask import Flask, request, jsonify, render_template_string
from PyPDF2 import PdfReader
from shifu_ocr.mind.mind import ShifuMind

app = Flask(__name__)
mind = ShifuMind()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ShifuMind</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d0d0d; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

  /* Sidebar */
  .sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: 260px; background: #0a0a0a; border-right: 1px solid #1a1a1a; display: flex; flex-direction: column; z-index: 10; }
  .sidebar-header { padding: 20px; border-bottom: 1px solid #1a1a1a; }
  .sidebar-header h1 { font-size: 20px; font-weight: 700; color: #fff; }
  .sidebar-header h1 span { color: #a78bfa; }
  .sidebar-header p { font-size: 11px; color: #666; margin-top: 4px; }
  .stats-panel { padding: 16px; flex: 1; overflow-y: auto; }
  .stats-panel h3 { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; }
  .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #141414; }
  .stat-row .label { font-size: 12px; color: #888; }
  .stat-row .value { font-size: 12px; color: #a78bfa; font-weight: 600; }
  .commands-panel { padding: 16px; border-top: 1px solid #1a1a1a; }
  .commands-panel h3 { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
  .cmd-hint { font-size: 11px; color: #555; padding: 3px 0; cursor: pointer; transition: color 0.2s; }
  .cmd-hint:hover { color: #a78bfa; }
  .cmd-hint code { color: #888; background: #1a1a1a; padding: 1px 5px; border-radius: 3px; font-size: 11px; }

  /* Main area */
  .main { margin-left: 260px; flex: 1; display: flex; flex-direction: column; height: 100vh; }

  /* Chat thread */
  .thread { flex: 1; overflow-y: auto; padding: 24px 0; }
  .thread-inner { max-width: 768px; margin: 0 auto; padding: 0 24px; }

  /* Welcome */
  .welcome { text-align: center; padding: 60px 20px 40px; }
  .welcome h2 { font-size: 28px; font-weight: 300; color: #fff; margin-bottom: 8px; }
  .welcome h2 span { color: #a78bfa; font-weight: 600; }
  .welcome p { color: #666; font-size: 14px; margin-bottom: 32px; }
  .suggestions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 560px; margin: 0 auto; }
  .suggestion { background: #141414; border: 1px solid #1e1e1e; border-radius: 12px; padding: 14px 16px; cursor: pointer; transition: all 0.2s; text-align: left; }
  .suggestion:hover { border-color: #a78bfa; background: #1a1a2a; }
  .suggestion .s-title { font-size: 13px; color: #ddd; font-weight: 500; margin-bottom: 4px; }
  .suggestion .s-desc { font-size: 11px; color: #666; }

  /* Messages */
  .msg { margin-bottom: 24px; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .msg-user { display: flex; justify-content: flex-end; }
  .msg-user .bubble { background: #2a2a3a; border-radius: 18px 18px 4px 18px; padding: 12px 18px; max-width: 85%; font-size: 14px; line-height: 1.5; }
  .msg-ai { display: flex; flex-direction: column; gap: 4px; }
  .msg-ai .sender { font-size: 11px; color: #a78bfa; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; padding-left: 2px; }
  .msg-ai .bubble { background: #141414; border: 1px solid #1e1e1e; border-radius: 4px 18px 18px 18px; padding: 16px 20px; max-width: 100%; font-size: 14px; line-height: 1.6; }

  /* Rich content inside AI bubbles */
  .bubble .metric { display: inline-flex; align-items: center; gap: 6px; background: #1a1a2a; border-radius: 6px; padding: 4px 10px; margin: 3px 2px; font-size: 13px; }
  .bubble .metric .mv { color: #a78bfa; font-weight: 600; }
  .bubble .metric .ml { color: #888; font-size: 11px; }
  .bubble .rank-card { display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: #1a1a2a; border-radius: 10px; margin: 6px 0; }
  .bubble .rank-card .rn { font-size: 22px; font-weight: 700; color: #a78bfa; min-width: 32px; }
  .bubble .rank-card .rw { font-weight: 600; font-size: 15px; }
  .bubble .rank-card .rs { color: #666; font-size: 11px; margin-left: auto; }
  .bubble .bar { height: 5px; background: #1e1e1e; border-radius: 3px; margin-top: 4px; overflow: hidden; }
  .bubble .bar-fill { height: 100%; background: linear-gradient(90deg, #a78bfa, #6c63ff); border-radius: 3px; transition: width 0.5s ease; }
  .bubble .act-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .bubble .act-row .aw { min-width: 90px; font-size: 13px; color: #ccc; }
  .bubble .act-row .bar { flex: 1; }
  .bubble .act-row .ae { font-size: 11px; color: #666; min-width: 50px; text-align: right; }
  .bubble .trace-line { font-size: 12px; color: #777; font-family: 'Cascadia Code', 'Fira Code', monospace; padding: 1px 0; }
  .bubble .section-label { font-size: 11px; color: #a78bfa; text-transform: uppercase; letter-spacing: 1px; margin-top: 12px; margin-bottom: 6px; }
  .bubble .tag { display: inline-block; background: #1a1a2a; color: #a78bfa; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
  .bubble .mono { font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; }
  .bubble .stats-mini { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }

  /* Typing indicator */
  .typing { display: flex; gap: 4px; padding: 8px 0; }
  .typing span { width: 7px; height: 7px; background: #a78bfa; border-radius: 50%; animation: blink 1.4s infinite; }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%,80%,100% { opacity: 0.3; } 40% { opacity: 1; } }

  /* Input area */
  .input-area { border-top: 1px solid #1a1a1a; padding: 16px 24px 20px; background: #0d0d0d; }
  .input-wrap { max-width: 768px; margin: 0 auto; display: flex; align-items: flex-end; gap: 10px; background: #141414; border: 1px solid #2a2a2a; border-radius: 16px; padding: 8px 8px 8px 18px; transition: border-color 0.2s; }
  .input-wrap:focus-within { border-color: #a78bfa; }
  .input-wrap textarea { flex: 1; background: none; border: none; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; resize: none; outline: none; max-height: 120px; min-height: 24px; line-height: 24px; padding: 4px 0; }
  .send-btn, .upload-btn { width: 36px; height: 36px; border-radius: 10px; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.2s; flex-shrink: 0; }
  .send-btn { background: #a78bfa; }
  .send-btn:hover { background: #8b5cf6; }
  .upload-btn { background: #2a2a3a; }
  .upload-btn:hover { background: #3a3a4a; }
  .send-btn svg, .upload-btn svg { width: 18px; height: 18px; fill: #fff; }
  .input-hint { text-align: center; font-size: 11px; color: #444; margin-top: 8px; }
  .bubble .page-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; color: #888; }
  .bubble .page-row .pn { color: #a78bfa; min-width: 60px; }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-header">
    <h1>Shifu<span>Mind</span></h1>
    <p>Cognitive Architecture</p>
  </div>
  <div class="stats-panel">
    <h3>System Status</h3>
    <div id="stats"></div>
  </div>
  <div class="commands-panel">
    <h3>Commands</h3>
    <div class="cmd-hint" onclick="fillCmd('/feed ')"><code>/feed</code> Teach text</div>
    <div class="cmd-hint" onclick="fillCmd('/score ')"><code>/score</code> Coherence</div>
    <div class="cmd-hint" onclick="fillCmd('/rank ')"><code>/rank</code> OCR candidates</div>
    <div class="cmd-hint" onclick="fillCmd('/confidence ')"><code>/conf</code> Word confidence</div>
    <div class="cmd-hint" onclick="fillCmd('/think ')"><code>/think</code> Deliberate</div>
    <div class="cmd-hint" onclick="fillCmd('/describe ')"><code>/desc</code> Describe word</div>
    <div class="cmd-hint" onclick="fillCmd('/activate ')"><code>/act</code> Spread activation</div>
    <div class="cmd-hint" onclick="fillCmd('/stats')"><code>/stats</code> System stats</div>
  </div>
</div>

<!-- Main -->
<div class="main">
  <div class="thread" id="thread">
    <div class="thread-inner" id="thread-inner">

      <!-- Welcome -->
      <div class="welcome" id="welcome">
        <h2>What can <span>ShifuMind</span> help with?</h2>
        <p>Feed knowledge, score coherence, rank OCR candidates, or explore the cognitive graph.</p>
        <div class="suggestions">
          <div class="suggestion" onclick="runSuggestion('/feed Thrombolytic therapy dissolves blood clots in acute ischemic stroke patients')">
            <div class="s-title">Teach it something</div>
            <div class="s-desc">Feed medical text about thrombolytic therapy</div>
          </div>
          <div class="suggestion" onclick="runSuggestion('/rank stroke:0.85, strake:0.90, stoke:0.70 | cerebral, arterial')">
            <div class="s-title">Rank OCR candidates</div>
            <div class="s-desc">Pick the right word using cognitive context</div>
          </div>
          <div class="suggestion" onclick="runSuggestion('/think cerebral ischemic stroke')">
            <div class="s-title">Deliberate on a topic</div>
            <div class="s-desc">Let the mind reason about stroke</div>
          </div>
          <div class="suggestion" onclick="runSuggestion('/activate stroke')">
            <div class="s-title">Spread activation</div>
            <div class="s-desc">See what lights up from "stroke"</div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- Input -->
  <div class="input-area">
    <div class="input-wrap">
      <button class="upload-btn" onclick="$('file-input').click()" title="Upload PDF">
        <svg viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 7V3.5L18.5 9H13zm-3 4h4v1h-4v-1zm0 2h4v1h-4v-1zm-2-4h1v5H8v-5z"/></svg>
      </button>
      <input type="file" id="file-input" accept=".pdf" style="display:none" onchange="uploadPDF(this)">
      <textarea id="input" rows="1" placeholder="Message ShifuMind... (try /feed, /score, /rank, /think, or upload a PDF)" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      <button class="send-btn" onclick="send()">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
    <div class="input-hint">Press Enter to send &middot; Shift+Enter for new line &middot; Click <span style="color:#666">PDF icon</span> to upload</div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const thread = $('thread-inner');

async function api(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  return r.json();
}

// ── Stats ──
async function loadStats() {
  const s = await (await fetch('/api/stats')).json();
  $('stats').innerHTML = [
    ['Vocabulary', s.vocabulary], ['Total Words', s.total_words],
    ['Assemblies', s.assemblies], ['Myelinated', s.myelinated],
    ['Domains', s.domains], ['Trunk Words', s.trunk_words],
    ['Feeds', s.feed_count], ['Plasticity', (s.plasticity*100).toFixed(0)+'%'],
  ].map(([l,v]) => `<div class="stat-row"><span class="label">${l}</span><span class="value">${v}</span></div>`).join('');
  return s;
}

// ── Message rendering ──
function addUser(text) {
  hideWelcome();
  const d = document.createElement('div');
  d.className = 'msg msg-user';
  d.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  thread.appendChild(d);
  scrollDown();
}

function addAI(html) {
  const d = document.createElement('div');
  d.className = 'msg msg-ai';
  d.innerHTML = `<div class="sender">ShifuMind</div><div class="bubble">${html}</div>`;
  thread.appendChild(d);
  scrollDown();
  return d;
}

function addTyping() {
  const d = document.createElement('div');
  d.className = 'msg msg-ai';
  d.id = 'typing';
  d.innerHTML = `<div class="sender">ShifuMind</div><div class="typing"><span></span><span></span><span></span></div>`;
  thread.appendChild(d);
  scrollDown();
}
function removeTyping() { const t = $('typing'); if(t) t.remove(); }

function scrollDown() { const t = $('thread'); t.scrollTop = t.scrollHeight; }
function hideWelcome() { const w = $('welcome'); if(w) w.style.display = 'none'; }
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ── Input handling ──
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
function fillCmd(cmd) {
  const inp = $('input');
  inp.value = cmd;
  inp.focus();
}
function runSuggestion(text) {
  $('input').value = text;
  send();
}

// ── Command router ──
async function send() {
  const inp = $('input');
  const raw = inp.value.trim();
  if (!raw) return;
  inp.value = '';
  inp.style.height = 'auto';

  addUser(raw);
  addTyping();

  try {
    // Parse command
    const text = raw;
    let html = '';

    if (text.startsWith('/feed ')) {
      html = await cmdFeed(text.slice(6));
    } else if (text.startsWith('/score ')) {
      html = await cmdScore(text.slice(7));
    } else if (text.startsWith('/rank ')) {
      html = await cmdRank(text.slice(6));
    } else if (text.startsWith('/confidence ') || text.startsWith('/conf ')) {
      html = await cmdConfidence(text.replace(/^\/(confidence|conf)\s+/, ''));
    } else if (text.startsWith('/think ') || text.startsWith('/deliberate ')) {
      html = await cmdThink(text.replace(/^\/(think|deliberate)\s+/, ''));
    } else if (text.startsWith('/describe ') || text.startsWith('/desc ')) {
      html = await cmdDescribe(text.replace(/^\/(describe|desc)\s+/, ''));
    } else if (text.startsWith('/activate ') || text.startsWith('/act ')) {
      html = await cmdActivate(text.replace(/^\/(activate|act)\s+/, ''));
    } else if (text === '/stats') {
      html = await cmdStats();
    } else if (text.startsWith('/')) {
      html = `Unknown command. Try <code>/feed</code>, <code>/score</code>, <code>/rank</code>, <code>/conf</code>, <code>/think</code>, <code>/desc</code>, <code>/act</code>, or <code>/stats</code>.`;
    } else {
      // Smart mode: auto-detect intent from natural language
      html = await cmdSmart(text);
    }

    removeTyping();
    addAI(html);
    loadStats();
  } catch(err) {
    removeTyping();
    addAI(`<span style="color:#f87171">Error: ${esc(err.message)}</span>`);
  }
}

// ── Commands ──

async function cmdFeed(text) {
  const r = await api('/api/feed', {text});
  if (r.accepted) {
    return `<div style="margin-bottom:8px">Knowledge absorbed.</div>` +
      `<div class="stats-mini">` +
      `<span class="metric"><span class="mv">${r.tokens_absorbed}</span><span class="ml">connections</span></span>` +
      `<span class="metric"><span class="mv">${r.domain}</span><span class="ml">domain</span></span>` +
      `<span class="metric"><span class="mv">${r.quality.toFixed(3)}</span><span class="ml">quality</span></span>` +
      `</div>`;
  }
  return `Text rejected (quality too low: ${r.quality.toFixed(3)}).`;
}

async function cmdScore(text) {
  const r = await api('/api/score', {text});
  const tokens = text.toLowerCase().split(/\s+/);
  let rows = r.scores.map((s, i) =>
    `<div class="act-row"><span class="aw">${tokens[i] || '?'}</span><div class="bar"><div class="bar-fill" style="width:${(s*100).toFixed(0)}%"></div></div><span class="ae">${s.toFixed(3)}</span></div>`
  ).join('');
  return `<div class="section-label">Coherence Score</div>` +
    `<span class="metric"><span class="mv">${r.coherence.toFixed(3)}</span></span>` +
    `<div class="section-label" style="margin-top:14px">Per-word breakdown</div>` + rows;
}

async function cmdRank(text) {
  // Format: word:score, word:score | context1, context2
  const parts = text.split('|');
  const candsRaw = parts[0].trim();
  const ctxRaw = parts[1] ? parts[1].trim() : '';

  const cands = candsRaw.split(',').map(c => {
    const [w, s] = c.trim().split(':');
    return [w.trim(), parseFloat(s || 0.5)];
  });
  const ctx = ctxRaw ? ctxRaw.split(',').map(w => w.trim()).filter(Boolean) : [];

  const r = await api('/api/predict', {candidates: cands, context: ctx});
  let html = ctx.length ? `<div style="margin-bottom:10px;color:#888;font-size:12px">Context: ${ctx.map(w=>`<span class="tag">${esc(w)}</span>`).join(' ')}</div>` : '';
  html += r.map(c =>
    `<div class="rank-card"><span class="rn">#${c.rank}</span><span class="rw">${esc(c.word)}</span><span class="rs">OCR ${c.ocr_score.toFixed(2)} | Field ${c.field_score.toFixed(2)} | Combined ${c.combined.toFixed(3)}</span></div>`
  ).join('');
  html += r.map(c =>
    `<div style="margin-top:2px"><span style="color:#666;font-size:11px">${esc(c.word)}</span><div class="bar"><div class="bar-fill" style="width:${(c.combined*100).toFixed(0)}%"></div></div></div>`
  ).join('');
  return html;
}

async function cmdConfidence(word) {
  const r = await api('/api/confidence', {word: word.trim()});
  return `<div style="margin-bottom:8px">Confidence for <strong>${esc(word.trim())}</strong></div>` +
    `<div class="stats-mini">` +
    `<span class="metric"><span class="mv">${r.score}/100</span><span class="ml">score</span></span>` +
    `<span class="metric"><span class="mv">${r.state}</span><span class="ml">state</span></span>` +
    `<span class="metric"><span class="mv">${r.layers}</span><span class="ml">layers</span></span>` +
    `<span class="metric"><span class="mv">${r.assemblies}</span><span class="ml">assemblies</span></span>` +
    `<span class="metric"><span class="mv">${r.myelinated ? 'Yes' : 'No'}</span><span class="ml">myelinated</span></span>` +
    `</div>`;
}

async function cmdThink(query) {
  const r = await api('/api/deliberate', {query});
  let html = `<div class="stats-mini" style="margin-bottom:12px">` +
    `<span class="metric"><span class="mv">${r.coherence.toFixed(3)}</span><span class="ml">coherence</span></span>` +
    `<span class="metric"><span class="mv">${r.steps}</span><span class="ml">steps</span></span>` +
    `<span class="metric"><span class="mv">${r.converged ? 'Yes' : 'No'}</span><span class="ml">converged</span></span>` +
    `</div>`;

  if (r.focus && r.focus.length) {
    html += `<div class="section-label">Focus</div><div>${r.focus.map(w=>`<span class="tag">${esc(w)}</span>`).join(' ')}</div>`;
  }
  if (r.retrieved && r.retrieved.length) {
    html += `<div class="section-label">Retrieved Concepts</div>`;
    html += r.retrieved.slice(0,10).map(x =>
      `<div class="act-row"><span class="aw">${esc(x.word)}</span><div class="bar"><div class="bar-fill" style="width:${Math.min(x.energy*100, 100).toFixed(0)}%"></div></div><span class="ae">${x.energy.toFixed(3)}</span></div>`
    ).join('');
  }
  if (r.trace && r.trace.length) {
    html += `<div class="section-label">Reasoning Trace</div>`;
    html += r.trace.map(t => `<div class="trace-line">${esc(t)}</div>`).join('');
  }
  return html;
}

async function cmdDescribe(word) {
  const r = await api('/api/describe', {word: word.trim()});
  return `<div style="margin-bottom:6px"><strong>${esc(word.trim())}</strong></div><div class="mono" style="color:#ccc">${esc(r.description)}</div>`;
}

async function cmdActivate(word) {
  const r = await api('/api/activate', {word: word.trim()});
  const entries = Object.entries(r).sort((a,b) => b[1]-a[1]).slice(0, 20);
  const max = entries[0] ? entries[0][1] : 1;
  let html = `<div style="margin-bottom:10px">Activation spread from <strong>${esc(word.trim())}</strong> &mdash; ${Object.keys(r).length} nodes reached</div>`;
  html += entries.map(([w, e]) =>
    `<div class="act-row"><span class="aw">${esc(w)}</span><div class="bar"><div class="bar-fill" style="width:${(e/max*100).toFixed(0)}%"></div></div><span class="ae">${e.toFixed(3)}</span></div>`
  ).join('');
  return html;
}

async function cmdStats() {
  const s = await loadStats();
  return `<div class="section-label">System Status</div>` +
    `<div class="stats-mini">` +
    `<span class="metric"><span class="mv">${s.vocabulary}</span><span class="ml">vocabulary</span></span>` +
    `<span class="metric"><span class="mv">${s.total_words}</span><span class="ml">total words</span></span>` +
    `<span class="metric"><span class="mv">${s.assemblies}</span><span class="ml">assemblies</span></span>` +
    `<span class="metric"><span class="mv">${s.myelinated}</span><span class="ml">myelinated</span></span>` +
    `<span class="metric"><span class="mv">${s.domains}</span><span class="ml">domains</span></span>` +
    `<span class="metric"><span class="mv">${s.trunk_words}</span><span class="ml">trunk words</span></span>` +
    `<span class="metric"><span class="mv">${s.feed_count}</span><span class="ml">feeds</span></span>` +
    `<span class="metric"><span class="mv">${(s.plasticity*100).toFixed(0)}%</span><span class="ml">plasticity</span></span>` +
    `</div>`;
}

// ── Knowledge query ──
async function cmdKnowledge() {
  const r = await (await fetch('/api/knowledge')).json();

  // Recent episodes (actual content learned)
  let html = '';
  if (r.episodes && r.episodes.length) {
    html += `<div class="section-label">Recently Learned</div>`;
    html += r.episodes.slice(0, 15).map(ep =>
      `<div style="padding:6px 10px;margin:4px 0;background:#1a1a2a;border-radius:6px;font-size:13px;line-height:1.5"><span style="color:#ccc">${esc(ep.text)}</span> <span style="color:#555;font-size:11px;margin-left:6px">${ep.domain} q:${ep.quality}</span></div>`
    ).join('');
  }

  // Top content words (stop words filtered)
  html += `<div class="section-label" style="margin-top:16px">Top Concepts (stop words filtered)</div>`;
  const max = r.words[0] ? r.words[0].freq : 1;
  html += r.words.slice(0, 25).map(w =>
    `<div class="act-row"><span class="aw">${esc(w.word)}</span><div class="bar"><div class="bar-fill" style="width:${(w.freq/max*100).toFixed(0)}%"></div></div><span class="ae">${w.score}/100 ${w.state}</span></div>`
  ).join('');

  // Domains
  if (r.domains && r.domains.length) {
    html += `<div class="section-label" style="margin-top:16px">Domains</div>`;
    html += r.domains.map(d =>
      `<div style="margin-bottom:8px"><strong>${esc(d.name)}</strong> <span style="color:#666">(${d.size} words)</span><div style="margin-top:4px">${d.top_words.map(w => `<span class="tag">${esc(w)}</span>`).join(' ')}</div></div>`
    ).join('');
  }
  return html;
}

// ── Smart mode: auto-detect intent ──
async function cmdSmart(text) {
  const lower = text.toLowerCase().replace(/[?.!]+$/, '').trim();

  // Meta queries about the mind's knowledge
  if (/^(what do you know|what did you learn|what have you learned|show me what you know|what do you remember|tell me what you know|what knowledge|summarize|summary|overview)/i.test(lower)) {
    return await cmdKnowledge();
  }

  // Stats queries
  if (/^(stats|status|how are you|how big|how much|system info)/i.test(lower)) {
    return await cmdStats();
  }

  // Help
  if (/^(help|commands|what can you do|how do i use)/i.test(lower)) {
    return `<div class="section-label">Commands</div>` +
      `<div style="line-height:2">` +
      `<div><code>/feed &lt;text&gt;</code> &mdash; Teach the mind new knowledge</div>` +
      `<div><code>/score &lt;text&gt;</code> &mdash; Score text coherence</div>` +
      `<div><code>/rank w1:0.9, w2:0.8 | ctx1, ctx2</code> &mdash; Rank OCR candidates</div>` +
      `<div><code>/conf &lt;word&gt;</code> &mdash; Check word confidence</div>` +
      `<div><code>/think &lt;query&gt;</code> &mdash; Multi-step reasoning</div>` +
      `<div><code>/desc &lt;word&gt;</code> &mdash; Describe a word</div>` +
      `<div><code>/act &lt;word&gt;</code> &mdash; Spread activation</div>` +
      `<div><code>/stats</code> &mdash; System status</div>` +
      `<div style="margin-top:8px;color:#888">Or just type naturally &mdash; ask "what do you know", "describe stroke", "think about X"</div>` +
      `<div style="color:#888">Upload PDFs with the document icon on the left</div>` +
      `</div>`;
  }

  // If it looks like a question about a word, describe it
  if (/^(what is|describe|define|explain|tell me about)\s+/i.test(text)) {
    const word = text.replace(/^(what is|describe|define|explain|tell me about)\s+/i, '').replace(/[?.!]+$/, '').trim().split(/\s+/)[0];
    if (word) return await cmdDescribe(word);
  }

  // If it has "score" or "coherence" keywords
  if (/\b(score|coherence|coherent)\b/i.test(text)) {
    const clean = text.replace(/\b(score|coherence|coherent|check|of|the|how|is)\b/gi, '').trim();
    if (clean) return await cmdScore(clean);
  }

  // If it has "confidence" keyword
  if (/\b(confidence|confident|how sure)\b/i.test(text)) {
    const word = text.replace(/\b(confidence|confident|how|sure|about|is|the|word)\b/gi, '').trim().split(/\s+/)[0];
    if (word) return await cmdConfidence(word);
  }

  // If it mentions "activate" or "spread"
  if (/\b(activate|spread|ripple)\b/i.test(text)) {
    const word = text.replace(/\b(activate|spread|ripple|from|the|word)\b/gi, '').trim().split(/\s+/)[0];
    if (word) return await cmdActivate(word);
  }

  // If it mentions "think" or "reason"
  if (/\b(think|reason|deliberate|ponder)\b/i.test(text)) {
    const query = text.replace(/\b(think|reason|deliberate|ponder|about)\b/gi, '').trim();
    if (query) return await cmdThink(query);
  }

  // Short input (< 5 words) = probably a query, not knowledge to feed
  if (text.split(/\s+/).length <= 4) {
    // Try to describe the main word or think about it
    const words = text.replace(/[?.!,]+/g, '').trim().split(/\s+/).filter(w => w.length > 2);
    if (words.length === 1) return await cmdDescribe(words[0]);
    if (words.length >= 2) return await cmdThink(text);
  }

  // Default: feed longer text as knowledge
  return await cmdFeed(text);
}

// ── PDF Upload ──
async function uploadPDF(input) {
  const file = input.files[0];
  if (!file) return;
  input.value = '';

  hideWelcome();
  addUser(`Upload PDF: ${file.name}`);
  addTyping();

  try {
    const form = new FormData();
    form.append('file', file);
    const resp = await fetch('/api/upload', {method: 'POST', body: form});
    const r = await resp.json();
    removeTyping();

    if (r.error) {
      addAI(`<span style="color:#f87171">Error: ${esc(r.error)}</span>`);
      return;
    }

    let html = `<div style="margin-bottom:10px">PDF ingested: <strong>${esc(r.filename)}</strong></div>`;
    html += `<div class="stats-mini">` +
      `<span class="metric"><span class="mv">${r.total_pages}</span><span class="ml">pages</span></span>` +
      `<span class="metric"><span class="mv">${r.pages_with_text}</span><span class="ml">with text</span></span>` +
      `<span class="metric"><span class="mv">${r.paragraphs_accepted}/${r.paragraphs_found}</span><span class="ml">paragraphs</span></span>` +
      `<span class="metric"><span class="mv">${r.tokens_absorbed}</span><span class="ml">connections</span></span>` +
      (r.domain ? `<span class="metric"><span class="mv">${r.domain}</span><span class="ml">domain</span></span>` : '') +
      `</div>`;

    addAI(html);
    loadStats();
  } catch(err) {
    removeTyping();
    addAI(`<span style="color:#f87171">Upload failed: ${esc(err.message)}</span>`);
  }
}

// ── Init ──
loadStats();
$('input').focus();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/stats')
def stats():
    return jsonify(mind.stats())


@app.route('/api/feed', methods=['POST'])
def feed():
    data = request.json
    result = mind.feed(data['text'])
    return jsonify(result)


@app.route('/api/score', methods=['POST'])
def score():
    data = request.json
    result = mind.score_text(data['text'])
    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    candidates = [(c[0], c[1]) for c in data['candidates']]
    context = data.get('context', [])
    result = mind.predict_candidates(candidates, context_words=context)
    return jsonify(result)


@app.route('/api/confidence', methods=['POST'])
def confidence():
    data = request.json
    return jsonify(mind.confidence(data['word']))


@app.route('/api/deliberate', methods=['POST'])
def deliberate():
    data = request.json
    return jsonify(mind.deliberate(data['query']))


@app.route('/api/describe', methods=['POST'])
def describe():
    data = request.json
    return jsonify({'description': mind.describe(data['word'])})


@app.route('/api/knowledge')
def knowledge():
    """Return top known content words with confidence, plus domain info and episodes."""
    # Filter out stop words so we see real content
    stop = mind.gate.stop_words(mind.cortex.word_freq)
    content_words = {w: f for w, f in mind.cortex.word_freq.items()
                     if w not in stop and len(w) > 2}
    top_words = sorted(content_words.items(), key=lambda x: -x[1])[:40]
    words = []
    for w, freq in top_words:
        conf = mind.cortex.confidence(w)
        words.append({
            'word': w, 'freq': freq,
            'score': conf['score'], 'state': conf['state'],
        })
    # Domain info -- also filter stop words from top_words display
    domains = []
    for name, dom in mind.trunk.domains.items():
        top = sorted(
            [w for w in dom.words if w not in stop and len(w) > 2],
            key=lambda w: mind.cortex.word_freq.get(w, 0), reverse=True,
        )[:12]
        domains.append({'name': name, 'size': dom.size(), 'top_words': top})
    # Recent episodes -- show actual sentences/content that was learned
    episodes = []
    for ep in reversed(mind.memory.episodes[-30:]):
        # Reconstruct readable text from tokens
        text = ' '.join(ep.tokens[:20])
        episodes.append({
            'text': text,
            'domain': ep.context.get('domain', '?'),
            'quality': round(ep.context.get('quality', 0), 3),
            'significance': round(ep.significance, 3),
        })
    return jsonify({'words': words, 'domains': domains, 'episodes': episodes})


@app.route('/api/activate', methods=['POST'])
def activate():
    data = request.json
    return jsonify(mind.activate(data['word']))


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    try:
        reader = PdfReader(io.BytesIO(f.read()))
        # Extract all text at once, then bulk-feed
        page_texts = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or '').strip()
            if text:
                page_texts.append({'page': i + 1, 'text': text})

        if not page_texts:
            return jsonify({'error': 'No text found in PDF'}), 400

        # Combine all text and use optimized bulk feed
        full_text = '\n\n'.join(p['text'] for p in page_texts)
        result = mind.feed_document(full_text)

        return jsonify({
            'filename': f.filename,
            'total_pages': len(reader.pages),
            'pages_with_text': len(page_texts),
            'paragraphs_found': result['total_paragraphs'],
            'paragraphs_accepted': result['accepted'],
            'tokens_absorbed': result['tokens_absorbed'],
            'domain': result['domain'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n  ShifuMind running at http://localhost:5000\n")
    app.run(debug=False, port=5000)
