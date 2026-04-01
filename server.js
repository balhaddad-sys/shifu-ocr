// Shifu OCR — Local Interactive Server
// Run: node server.js
// Open: http://localhost:3737

const http = require('http');
const url = require('url');
const { createOrRestore, ShifuPipeline, FeedbackLoop, DocumentIngestor, MetricsTracker } = require('./index');

const PORT = 3737;
const VOCABULARY_TARGET_SIZE = Number(process.env.SHIFU_VOCAB_TARGET || 100000);

// Boot the system
console.log('Booting Shifu OCR...');
const shifu = createOrRestore({
  seed: true,
  loadTrained: true,
  autoSave: true,
  stateDir: '.state',
  vocabularyTargetSize: VOCABULARY_TARGET_SIZE,
});
const pipeline = shifu.createPipeline();
const feedback = new FeedbackLoop(shifu, { stateDir: '.state' });
const metrics = new MetricsTracker({ stateDir: '.state' });
const ingestor = new DocumentIngestor(pipeline, { outputDir: '.ingested' });

// ── Boot Python Mind subprocess ─────────────────────────────
const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');
let mindProcess = null;
let mindReady = false;
let mindPending = {};  // id → {resolve, timer}
let mindSeq = 0;

function bootMind() {
  const script = path.join(__dirname, 'shifu_ocr', 'mind_worker.py');
  mindProcess = spawn('python', [script], { cwd: __dirname, stdio: ['pipe', 'pipe', 'pipe'] });
  const rl = readline.createInterface({ input: mindProcess.stdout });
  rl.on('line', (line) => {
    try {
      const data = JSON.parse(line);
      if (data.status === 'ready') {
        mindReady = true;
        console.log('  Mind ready: ' + (data.vocabulary || 0) + ' words');
        return;
      }
      // Match response to request by _id
      const id = data._id;
      if (id !== undefined && mindPending[id]) {
        const { resolve, timer } = mindPending[id];
        clearTimeout(timer);
        delete mindPending[id];
        delete data._id;
        resolve(data);
      }
    } catch (e) {}
  });
  mindProcess.stderr.on('data', (d) => {
    const msg = d.toString().trim();
    if (msg) console.warn('  Mind stderr:', msg.slice(0, 200));
  });
  mindProcess.on('exit', (code) => {
    console.warn('Mind process exited:', code, '— restarting...');
    mindReady = false;
    mindProcess = null;
    // Reject all pending
    for (const id of Object.keys(mindPending)) {
      const { resolve, timer } = mindPending[id];
      clearTimeout(timer);
      resolve({ ok: false, error: 'mind crashed' });
    }
    mindPending = {};
    // Auto-restart after 1 second
    setTimeout(bootMind, 1000);
  });
}

function mindCommand(cmd) {
  return new Promise((resolve) => {
    if (!mindProcess || !mindReady) return resolve({ ok: false, error: 'mind not ready' });
    const id = ++mindSeq;
    const timer = setTimeout(() => {
      if (mindPending[id]) {
        delete mindPending[id];
        resolve({ ok: false, error: 'timeout' });
      }
    }, 15000);
    mindPending[id] = { resolve, timer };
    cmd._id = id;
    mindProcess.stdin.write(JSON.stringify(cmd) + '\n');
  });
}

bootMind();
console.log('Shifu OCR ready.');

function jsonResponse(res, data, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': 'http://localhost:3737' });
  res.end(JSON.stringify(data, null, 2));
}

const MAX_BODY_SIZE = 10 * 1024 * 1024; // 10 MB
const MAX_TEXT_LENGTH = 100000; // 100k chars for text inputs

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    let size = 0;
    req.on('error', err => reject(err));
    req.on('data', chunk => {
      size += chunk.length;
      if (size > MAX_BODY_SIZE) {
        req.destroy();
        return reject(new Error('Request body too large'));
      }
      body += chunk;
    });
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch { resolve({}); }
    });
  });
}

const server = http.createServer(async (req, res) => {
  try {
  const parsed = url.parse(req.url, true);
  const path = parsed.pathname;

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(200, { 'Access-Control-Allow-Origin': 'http://localhost:3737', 'Access-Control-Allow-Methods': 'GET,POST', 'Access-Control-Allow-Headers': 'Content-Type' });
    res.end();
    return;
  }

  // ── Serve the UI ──────────────────────────────────────────────
  if (path === '/' || path === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(HTML);
    return;
  }

  // ── API Routes ────────────────────────────────────────────────

  // Correct text (handles multi-line: splits on newlines, corrects each)
  if (path === '/api/correct' && req.method === 'POST') {
    const { text } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    if (text.length > MAX_TEXT_LENGTH) return jsonResponse(res, { error: 'text too long' }, 400);
    const lines = text.split('\n').filter(l => l.trim());
    if (lines.length === 1) {
      const result = shifu.correctLine(lines[0]);
      const decision = shifu.assessConfidence(result);
      const coherence = shifu.scoreSentence(result.output);
      return jsonResponse(res, { ...result, decision, coherence: coherence.coherence });
    }
    // Multi-line: correct each line independently
    const results = lines.map(line => {
      const r = shifu.correctLine(line);
      return { ...r, decision: shifu.assessConfidence(r) };
    });
    const allWords = results.flatMap(r => r.words);
    const allFlags = results.flatMap(r => r.safetyFlags);
    const output = results.map(r => r.output).join('\n');
    const avgConf = allWords.length > 0 ? allWords.reduce((s, w) => s + w.confidence, 0) / allWords.length : 0;
    const hasDangers = results.some(r => r.hasDangers);
    const hasWarnings = results.some(r => r.hasWarnings);
    const decision = hasDangers ? 'reject' : hasWarnings ? 'verify' : results.some(r => r.decision !== 'accept') ? 'verify' : 'accept';
    const coherence = shifu.scoreSentence(output);
    return jsonResponse(res, { input: text, output, words: allWords, safetyFlags: allFlags, avgConfidence: Math.round(avgConf * 100) / 100, hasDangers, hasWarnings, decision, coherence: coherence.coherence });
  }

  // Correct a table row (ward census)
  if (path === '/api/correct-row' && req.method === 'POST') {
    const { row } = await parseBody(req);
    if (!row) return jsonResponse(res, { error: 'row required' }, 400);
    const result = pipeline.processTableRow(row);
    return jsonResponse(res, result);
  }

  // Propose + evaluate (feedback loop)
  if (path === '/api/propose' && req.method === 'POST') {
    const { text, row } = await parseBody(req);
    if (text) return jsonResponse(res, feedback.propose(text));
    if (row) return jsonResponse(res, feedback.proposeRow(row));
    return jsonResponse(res, { error: 'text or row required' }, 400);
  }

  if (path === '/api/evaluate' && req.method === 'POST') {
    const { proposal, confirmed } = await parseBody(req);
    if (!proposal || confirmed === undefined) return jsonResponse(res, { error: 'proposal and confirmed required' }, 400);
    const result = feedback.evaluate(proposal, confirmed);
    return jsonResponse(res, result);
  }

  // Learn from a correction
  if (path === '/api/learn' && req.method === 'POST') {
    const { ocrRow, confirmedRow } = await parseBody(req);
    if (!ocrRow || !confirmedRow) return jsonResponse(res, { error: 'ocrRow and confirmedRow required' }, 400);
    const result = shifu.learn(ocrRow, confirmedRow);
    return jsonResponse(res, result);
  }

  // Undo last learn
  if (path === '/api/undo' && req.method === 'POST') {
    const result = shifu.undo();
    return jsonResponse(res, result || { message: 'nothing to undo' });
  }

  // Structural invariance comparison
  if (path === '/api/compare-structure' && req.method === 'POST') {
    const { sentA, sentB } = await parseBody(req);
    if (!sentA || !sentB) return jsonResponse(res, { error: 'sentA and sentB required' }, 400);
    if (sentA.length > MAX_TEXT_LENGTH || sentB.length > MAX_TEXT_LENGTH) return jsonResponse(res, { error: 'text too long' }, 400);
    return jsonResponse(res, shifu.compareStructure(sentA, sentB));
  }

  // Score a sentence
  if (path === '/api/score' && req.method === 'POST') {
    const { text } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    if (text.length > MAX_TEXT_LENGTH) return jsonResponse(res, { error: 'text too long' }, 400);
    return jsonResponse(res, shifu.scoreSentence(text));
  }

  // Extract roles
  if (path === '/api/roles' && req.method === 'POST') {
    const { text } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    if (text.length > MAX_TEXT_LENGTH) return jsonResponse(res, { error: 'text too long' }, 400);
    return jsonResponse(res, shifu.extractRoles(text));
  }

  // System stats
  if (path === '/api/stats') {
    return jsonResponse(res, {
      ...shifu.stats(),
      feedback: feedback.getMetrics(),
      metrics: metrics.summary(),
    });
  }

  // History — return metadata only, strip raw patient data
  if (path === '/api/history') {
    const history = shifu.getHistory(20).map(h => ({
      id: h.id,
      timestamp: h.timestamp,
      columns: h.columns || [],
      rejectedCount: h.rejectedCount || 0,
    }));
    return jsonResponse(res, history);
  }

  // Ingest raw text
  if (path === '/api/ingest' && req.method === 'POST') {
    const { text, format, columns, delimiter } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    if (text.length > MAX_TEXT_LENGTH) return jsonResponse(res, { error: 'text too long' }, 400);
    const result = ingestor.ingestRawText(text, { format, columns, delimiter });
    return jsonResponse(res, result);
  }

  // Upload a file (PDF, CSV, TXT, images) — multipart form or raw body
  if (path === '/api/upload' && req.method === 'POST') {
    const fs = require('fs');
    const os = require('os');
    const tmpDir = os.tmpdir();

    // Read raw body with size limit
    const chunks = [];
    let uploadSize = 0;
    await new Promise((resolve, reject) => {
      req.on('error', err => reject(err));
      req.on('data', chunk => {
        uploadSize += chunk.length;
        if (uploadSize > MAX_BODY_SIZE) {
          req.destroy();
          return reject(new Error('Upload too large'));
        }
        chunks.push(chunk);
      });
      req.on('end', resolve);
    });
    const body = Buffer.concat(chunks);

    // Extract filename from Content-Disposition or query param
    const qs = url.parse(req.url, true).query;
    const rawFilename = qs.filename || 'upload' + Date.now();
    const correctNativeText = qs.correctNativeText === '1' || qs.correctNativeText === 'true';
    // Sanitize filename: strip path separators and traversal sequences, then verify
    const filename = require('path').basename(rawFilename).replace(/\.\./g, '_').replace(/[/\\]/g, '_');
    const tmpPath = require('path').join(tmpDir, 'shifu_' + filename);
    if (!tmpPath.startsWith(tmpDir)) return jsonResponse(res, { error: 'invalid filename' }, 400);

    // Check for multipart form data
    const contentType = req.headers['content-type'] || '';
    if (contentType.includes('multipart/form-data')) {
      let boundary = contentType.split('boundary=')[1];
      if (boundary) {
        // Strip quotes from boundary (RFC 2046 allows boundary="token")
        boundary = boundary.replace(/^["']|["']$/g, '').split(';')[0].trim();
        // Per RFC 2046, boundary delimiters appear as \r\n--boundary at line starts.
        // The first delimiter may omit the leading \r\n. We search for \r\n--boundary
        // to avoid false matches inside file content.
        const lineDelim = Buffer.from('\r\n--' + boundary);
        const firstDelim = Buffer.from('--' + boundary);
        const hdrSep = Buffer.from('\r\n\r\n');

        // Find the first boundary (may not have leading \r\n)
        let firstStart = body.indexOf(firstDelim);
        if (firstStart === -1) {
          return jsonResponse(res, { error: 'no boundary found in upload' }, 400);
        }

        // Find all parts by searching for \r\n--boundary (line-start only)
        const parts = [];
        let partStart = firstStart + firstDelim.length;
        // Skip past the \r\n after the first boundary line
        if (body[partStart] === 0x0D && body[partStart + 1] === 0x0A) partStart += 2;

        while (partStart < body.length) {
          const nextDelim = body.indexOf(lineDelim, partStart);
          const partEnd = nextDelim === -1 ? body.length : nextDelim;
          parts.push(body.slice(partStart, partEnd));
          if (nextDelim === -1) break;
          partStart = nextDelim + lineDelim.length;
          // Skip \r\n or -- after boundary
          if (body[partStart] === 0x2D && body[partStart + 1] === 0x2D) break; // closing --
          if (body[partStart] === 0x0D && body[partStart + 1] === 0x0A) partStart += 2;
        }

        for (const partBuf of parts) {
          const hdrEnd = partBuf.indexOf(hdrSep);
          if (hdrEnd === -1) continue;
          const headers = partBuf.slice(0, hdrEnd).toString('utf8');
          const fileData = partBuf.slice(hdrEnd + hdrSep.length);
          const fnMatch = headers.match(/filename="([^"]+)"/);
          if (fnMatch) {
            const realName = require('path').basename(fnMatch[1]).replace(/\.\./g, '_').replace(/[/\\]/g, '_');
            const realPath = require('path').join(tmpDir, 'shifu_' + realName);
            if (!realPath.startsWith(tmpDir)) continue;
            fs.writeFileSync(realPath, fileData);
            try {
              const doc = await ingestor.ingestFile(realPath, { correctNativeText });
              return jsonResponse(res, doc);
            } catch (err) {
              console.warn('Upload processing error:', err.message);
              return jsonResponse(res, { error: 'failed to process upload' }, 500);
            } finally {
              try { fs.unlinkSync(realPath); } catch {}
            }
          }
        }
      }
      return jsonResponse(res, { error: 'no file found in upload' }, 400);
    }

    // Raw body upload
    fs.writeFileSync(tmpPath, body);
    try {
      const doc = await ingestor.ingestFile(tmpPath, { correctNativeText });
      return jsonResponse(res, doc);
    } catch (err) {
      return jsonResponse(res, { error: err.message }, 500);
    } finally {
      try { fs.unlinkSync(tmpPath); } catch {}
    }
  }

  // ── Mind API routes ────────────────────────────────────────
  if (path === '/api/mind/stats') return jsonResponse(res, await mindCommand({ cmd: 'stats' }));
  if (path === '/api/mind/hungry') return jsonResponse(res, await mindCommand({ cmd: 'hungry' }));

  if (path.startsWith('/api/mind/describe/')) {
    const word = decodeURIComponent(path.split('/').pop());
    return jsonResponse(res, await mindCommand({ cmd: 'describe', word }));
  }
  if (path.startsWith('/api/mind/generate/')) {
    const word = decodeURIComponent(path.split('/').pop());
    return jsonResponse(res, await mindCommand({ cmd: 'generate', word }));
  }
  if (path.startsWith('/api/mind/activate/')) {
    const word = decodeURIComponent(path.split('/').pop());
    return jsonResponse(res, await mindCommand({ cmd: 'activate', word }));
  }
  if (path.startsWith('/api/mind/confidence/')) {
    const word = decodeURIComponent(path.split('/').pop());
    return jsonResponse(res, await mindCommand({ cmd: 'confidence', word }));
  }

  if (path === '/api/mind/feed' && req.method === 'POST') {
    const { text } = await parseBody(req);
    return jsonResponse(res, await mindCommand({ cmd: 'feed', text }));
  }
  if (path === '/api/mind/feed-batch' && req.method === 'POST') {
    const { texts } = await parseBody(req);
    return jsonResponse(res, await mindCommand({ cmd: 'feed_batch', texts }));
  }
  if (path === '/api/mind/score' && req.method === 'POST') {
    const { text } = await parseBody(req);
    return jsonResponse(res, await mindCommand({ cmd: 'score', text }));
  }
  if (path === '/api/mind/candidates' && req.method === 'POST') {
    const { candidates, context } = await parseBody(req);
    return jsonResponse(res, await mindCommand({ cmd: 'candidates', candidates, context }));
  }
  if (path === '/api/mind/deliberate' && req.method === 'POST') {
    const { query } = await parseBody(req);
    return jsonResponse(res, await mindCommand({ cmd: 'deliberate', query }));
  }
  if (path === '/api/mind/connect' && req.method === 'POST') {
    const body = await parseBody(req);
    return jsonResponse(res, await mindCommand({ cmd: 'connect', from: body.from, to: body.to }));
  }
  if (path === '/api/mind/save' && req.method === 'POST') {
    return jsonResponse(res, await mindCommand({ cmd: 'save' }));
  }

  jsonResponse(res, { error: 'not found' }, 404);
  } catch (err) {
    console.warn('Request error:', err.message);
    if (!res.headersSent) {
      const status = /too large/i.test(err.message) ? 413 : 500;
      const msg = status === 413 ? err.message : 'internal error';
      jsonResponse(res, { error: msg }, status);
    }
  }
});

// ── Unified Chat UI ──────────────────────────────────────────────

// ── Unified Chat UI ──────────────────────────────────────────────
const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Shifu</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0a0a;--surface:#111;--accent:#c8a97e;--accent2:#a08050;--green:#7c9885;--blue:#5b8fd4;--amber:#f59e0b;--red:#ef4444;--purple:#a78bfa;--text:#e8e8e8;--dim:#777;--faint:#444;--user-bg:#161626;--shifu-bg:#141e14;--border:#1e1e1e;--radius:16px}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:-apple-system,'Segoe UI',system-ui,sans-serif}
body{display:flex;flex-direction:column}
.top{padding:12px 20px;display:flex;align-items:center;gap:16px;background:var(--bg);border-bottom:1px solid var(--border);flex:0 0 auto;z-index:10}
.logo{font-size:20px;font-weight:700;color:var(--accent);letter-spacing:2px}
.stats{font-size:11px;color:var(--dim);line-height:1.4}
.chat{flex:1 1 0;min-height:0;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:16px 20px 100px 20px;display:flex;flex-direction:column;gap:12px}
.chat::-webkit-scrollbar{width:4px}
.chat::-webkit-scrollbar-thumb{background:var(--faint);border-radius:2px}
.msg{max-width:85%;animation:fadeIn .25s ease}
.msg.user{align-self:flex-end}
.msg.shifu{align-self:flex-start}
.bubble{padding:14px 18px;border-radius:var(--radius);font-size:14px;line-height:1.7;word-wrap:break-word}
.msg.user .bubble{background:var(--user-bg);border-bottom-right-radius:4px;color:#b8c8e0}
.msg.shifu .bubble{background:var(--shifu-bg);border-bottom-left-radius:4px}
.trace{font-size:11px;color:var(--dim);margin-top:8px;padding:8px 12px;background:rgba(200,169,126,0.06);border-radius:8px;line-height:1.7}
.trace .lbl{color:var(--accent);font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.5px}
.word{display:inline-block;padding:2px 6px;margin:1px;border-radius:4px;font-size:13px}
.word.exact{background:#064e3b;color:#6ee7b7}
.word.high_confidence,.word.mind_corrected{background:#164e63;color:#67e8f9}
.word.corrected_verify{background:#713f12;color:#fde68a}
.word.low_confidence,.word.unknown{background:#7f1d1d;color:#fca5a5}
.word.number,.word.dosage,.word.punctuation,.word.passthrough,.word.room_code,.word.title,.word.short,.word.clean,.word.short_unknown{background:#1e293b;color:#94a3b8}
.word.digraph_corrected{background:#312e81;color:#c4b5fd}
.word.DANGER_medication_ambiguity{background:#dc2626;color:white;font-weight:bold}
.flag{display:inline-block;padding:3px 8px;margin:2px;border-radius:4px;font-size:11px;font-weight:600}
.flag.danger{background:#dc2626;color:white}
.flag.warning{background:#d97706;color:white}
.decision{display:inline-block;padding:3px 10px;border-radius:10px;font-size:12px;font-weight:700;margin-top:6px}
.decision.accept{background:#064e3b;color:#6ee7b7}
.decision.verify{background:#713f12;color:#fde68a}
.decision.reject{background:#7f1d1d;color:#fca5a5}
.bottom{position:fixed;bottom:0;left:0;right:0;z-index:20;padding:10px 16px;background:var(--bg);border-top:1px solid var(--border);display:flex;gap:10px;align-items:flex-end}
.bottom textarea{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:24px;padding:12px 18px;color:var(--text);font-size:14px;resize:none;height:44px;max-height:120px;font-family:inherit;outline:none;transition:border-color .2s}
.bottom textarea:focus{border-color:var(--accent)}
.bottom textarea::placeholder{color:var(--faint)}
.btn{width:40px;height:40px;border-radius:50%;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;transition:all .15s}
.btn-send{background:var(--accent);color:#000;font-weight:700;font-size:18px}
.btn-send:hover{background:var(--accent2)}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@media(max-width:600px){.chat{padding:10px 12px 100px 12px}.bubble{font-size:13px;padding:10px 14px}}
</style>
</head>
<body>
<div class="top">
  <div class="logo">SHIFU</div>
  <div class="stats" id="topStats">booting...</div>
</div>
<div class="chat" id="chat">
  <div class="msg shifu"><div class="bubble">
    <div style="color:var(--accent);font-weight:500">Model the medium. Detect the perturbation. Let experience shape the landscape.</div>
    <div style="font-size:12px;color:var(--dim);margin-top:8px">
      <b>Ask</b> anything &middot; <b>Paste OCR text</b> to correct &middot; <b>Feed</b> text to teach me<br>
      <span style="color:var(--faint)">Commands: /correct &middot; /describe &middot; /generate &middot; /connect &middot; /stats &middot; /feed</span>
    </div>
  </div></div>
</div>
<div class="bottom">
  <button class="btn" style="background:transparent;border:1px solid var(--border);color:var(--dim);font-size:15px" onclick="document.getElementById('fileInput').click()" title="Upload PDFs / text files">+</button>
  <input type="file" id="fileInput" accept=".pdf,.txt,.csv,.tsv,.md,.html,.json,text/*,application/pdf" multiple onchange="handleFiles(event)" style="display:none">
  <textarea id="input" placeholder="Ask Shifu anything..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}" oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,120)+'px'"></textarea>
  <button class="btn btn-send" onclick="send()">&rarr;</button>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('input');
function addMsg(h,w){const d=document.createElement('div');d.className='msg '+w;d.innerHTML='<div class="bubble">'+h+'</div>';chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
async function api(p,b){const o=b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{};return(await fetch(p,o)).json()}
function rw(ws){return(ws||[]).map(w=>{const c=w.flag||'unknown',t=w.original!==w.corrected?w.original+' \\u2192 '+w.corrected:w.corrected;return'<span class="word '+c+'" title="'+t+' ('+c+')">'+w.corrected+'</span>'}).join(' ')}
function rf(fs){if(!fs||!fs.length)return'';return'<div style="margin-top:6px">'+fs.map(f=>'<span class="flag '+(f.severity||'warning')+'">'+f.message+'</span>').join(' ')+'</div>'}

function detect(t){
  t=t.trim();
  if(t.startsWith('/correct '))return{i:'correct',p:t.slice(9)};
  if(t.startsWith('/feed '))return{i:'feed',p:t.slice(6)};
  if(t.startsWith('/describe '))return{i:'describe',p:t.slice(10).trim().split(/\\s+/)[0]};
  if(t.startsWith('/generate '))return{i:'generate',p:t.slice(10).trim().split(/\\s+/)[0]};
  if(t.startsWith('/connect ')){const p=t.slice(9).trim().split(/\\s+/);return{i:'connect',f:p[0],t:p[1]||p[0]}}
  if(t==='/stats'||t.startsWith('/stats'))return{i:'stats'};
  if(t.startsWith('/score '))return{i:'score',p:t.slice(7)};
  if(/^(what|how|why|describe|explain|define|tell|discuss)\\b/i.test(t))return{i:'deliberate',p:t};
  if(t.endsWith('?')||/^(is|are|does|can|should|when|which)\\b/i.test(t))return{i:'deliberate',p:t};
  return{i:'correct',p:t};
}

async function send(){
  const text=inp.value.trim();if(!text)return;
  inp.value='';inp.style.height='44px';
  addMsg('<div style="white-space:pre-wrap">'+text.replace(/</g,'&lt;')+'</div>','user');
  const d=detect(text),ld=addMsg('<div style="color:var(--dim);font-style:italic">Thinking...</div>','shifu');
  let h='';
  try{
    if(d.i==='correct'){
      const r=await api('/api/correct',{text:d.p});
      h='<div style="margin-bottom:6px"><b>Corrected:</b></div><div>'+rw(r.words)+'</div>'+rf(r.safetyFlags);
      if(r.decision)h+='<div><span class="decision '+r.decision+'">'+r.decision.toUpperCase()+'</span></div>';
      try{const ms=await api('/api/mind/score',{text:r.output||d.p});if(ms.ok)h+='<div class="trace"><span class="lbl">mind coherence</span> '+(ms.coherence||0).toFixed(3)+'</div>'}catch{}
    }
    else if(d.i==='deliberate'){
      const r=await api('/api/mind/deliberate',{query:d.p});
      if(r.ok){
        h='<div style="margin-bottom:4px"><b>Focus:</b> '+(r.focus||[]).join(', ')+'</div>';
        h+='<div><b>Retrieved:</b> '+(r.retrieved||[]).map(x=>x.word).join(', ')+'</div>';
        h+='<div class="trace"><span class="lbl">coherence</span> '+(r.coherence||0).toFixed(3)+' &middot; <span class="lbl">steps</span> '+(r.steps||0)+' &middot; <span class="lbl">converged</span> '+(r.converged?'yes':'no')+'</div>';
        const fw=(r.focus||[]).find(w=>w.length>3);
        if(fw){try{const ds=await api('/api/mind/describe/'+fw);if(ds.ok)h+='<div style="margin-top:8px">'+ds.description+'</div>'}catch{}}
      }else{const r2=await api('/api/score',{text:d.p});h='<div class="trace"><span class="lbl">coherence</span> '+(r2.coherence||0).toFixed(3)+'</div>'}
    }
    else if(d.i==='feed'){
      const[r1,r2]=await Promise.all([api('/api/mind/feed',{text:d.p}),api('/api/score',{text:d.p})]);
      h='<div><b>Fed:</b> '+(r1.accepted?'accepted':'rejected')+'</div>';
      if(r1.domain)h+='<div class="trace"><span class="lbl">domain</span> '+r1.domain+' &middot; <span class="lbl">quality</span> '+(r1.quality||0).toFixed(2)+'</div>';
    }
    else if(d.i==='describe'){const r=await api('/api/mind/describe/'+d.p);h='<div>'+(r.description||'Unknown.')+'</div>'}
    else if(d.i==='generate'){const r=await api('/api/mind/generate/'+d.p);h='<div style="font-style:italic;color:var(--accent)">'+(r.text||'...')+'</div>'}
    else if(d.i==='connect'){const r=await api('/api/mind/connect',{from:d.f,to:d.t});h=r.connected&&r.path?'<div>'+r.path.join(' \\u2192 ')+'</div>':'<div style="color:var(--dim)">No path found</div>'}
    else if(d.i==='score'){const[js,m]=await Promise.all([api('/api/score',{text:d.p}),api('/api/mind/score',{text:d.p})]);h='<div class="trace"><span class="lbl">js</span> '+(js.coherence||0).toFixed(3)+' &middot; <span class="lbl">mind</span> '+(m.coherence||0).toFixed(3)+'</div>'}
    else if(d.i==='stats'){const[js,m]=await Promise.all([api('/api/stats'),api('/api/mind/stats')]);h='<div class="trace"><span class="lbl">JS Engine</span><br>Vocab: '+(js.core?.vocabulary||0)+' &middot; Resonance: '+(js.core?.resonancePairs||0)+'<br><br><span class="lbl">Python Mind</span><br>Vocab: '+(m.vocabulary||0)+' &middot; Domains: '+(m.domains||0)+' &middot; Assemblies: '+(m.assemblies||0)+' &middot; Myelinated: '+(m.myelinated||0)+'</div>'}
    ld.querySelector('.bubble').innerHTML=h||'<div style="color:var(--dim)">No response.</div>';
  }catch(e){ld.querySelector('.bubble').innerHTML='<div style="color:var(--red)">Error: '+e.message+'</div>'}
  updateStats();
}

// ── File upload handler (multi-file, auto-feed to mind) ──
async function handleFiles(event) {
  const files = event.target.files;
  if (!files || !files.length) return;
  addMsg('<div style="color:var(--dim)">Uploading ' + files.length + ' file(s)...</div>', 'user');
  const loading = addMsg('<div id="feedProgress" style="color:var(--dim)">Processing...</div>', 'shifu');
  const prog = loading.querySelector('#feedProgress');
  let totalFed = 0, totalFiles = files.length, doneFiles = 0;

  function updateProgress(fileIdx, fileName, status, detail) {
    const pct = Math.round((fileIdx / totalFiles) * 100);
    let bar = '<div style="background:var(--border);border-radius:4px;height:6px;margin:8px 0;overflow:hidden"><div style="background:var(--accent);height:100%;width:' + pct + '%;transition:width 0.3s"></div></div>';
    bar += '<div style="font-size:11px;color:var(--dim)">' + fileIdx + '/' + totalFiles + ' files &middot; ' + totalFed + ' passages fed</div>';
    if (fileName) bar += '<div style="font-size:12px;margin-top:4px">' + status + ' <b>' + fileName + '</b>' + (detail ? ' &middot; ' + detail : '') + '</div>';
    prog.innerHTML = bar;
  }

  let resultsHtml = '';
  for (let fi = 0; fi < files.length; fi++) {
    const file = files[fi];
    updateProgress(fi, file.name, 'Processing', '');
    try {
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        updateProgress(fi, file.name, 'Extracting text from', '');
        const res = await fetch('/api/upload?filename=' + encodeURIComponent(file.name), { method: 'POST', body: file });
        const data = await res.json();
        if (data.error) { resultsHtml += '<div style="color:var(--red)">' + file.name + ': ' + data.error + '</div>'; continue; }
        const lines = (data.lines || []).map(l => l.corrected || l.output || l.input || '').filter(l => l.length > 10);
        resultsHtml += '<div style="margin-bottom:8px"><b>' + file.name + '</b>: ' + (data.pageCount || '?') + ' pages, ' + lines.length + ' lines</div>';
        if (lines.length > 0) {
          updateProgress(fi, file.name, 'Feeding', lines.length + ' lines to mind');
          const feedResult = await api('/api/mind/feed-batch', { texts: lines });
          resultsHtml += '<div class="trace"><span class="lbl">fed</span> ' + (feedResult.accepted || 0) + '/' + (feedResult.total || 0) + ' accepted</div>';
          totalFed += feedResult.accepted || 0;
        }
      } else {
        updateProgress(fi, file.name, 'Reading', '');
        const text = await file.text();
        const sentences = text.split(/[.!?\\n]+/).map(s => s.trim()).filter(s => s.length > 10);
        resultsHtml += '<div style="margin-bottom:8px"><b>' + file.name + '</b>: ' + sentences.length + ' sentences</div>';
        if (sentences.length > 0) {
          updateProgress(fi, file.name, 'Feeding', sentences.length + ' sentences to mind');
          const feedResult = await api('/api/mind/feed-batch', { texts: sentences });
          resultsHtml += '<div class="trace"><span class="lbl">fed</span> ' + (feedResult.accepted || 0) + '/' + (feedResult.total || 0) + ' accepted</div>';
          totalFed += feedResult.accepted || 0;
        }
      }
    } catch (e) { resultsHtml += '<div style="color:var(--red)">' + file.name + ': ' + e.message + '</div>'; }
    doneFiles++;
  }
  updateProgress(totalFiles, null, '', '');
  resultsHtml += '<div style="margin-top:8px;color:var(--green);font-weight:600">Done: ' + totalFed + ' passages fed to mind from ' + doneFiles + ' file(s)</div>';
  prog.innerHTML = resultsHtml;
  event.target.value = '';
  updateStats();
}

async function updateStats(){
  try{const[js,m]=await Promise.all([api('/api/stats'),api('/api/mind/stats')]);
  document.getElementById('topStats').innerHTML=(js.core?.vocabulary||0)+'w &middot; '+(js.core?.resonancePairs||0)+' res &middot; '+(m.vocabulary||0)+' mind &middot; '+(m.domains||0)+' dom &middot; '+(m.myelinated||0)+' myel'}catch{}
}
updateStats();
</script>
</body>
</html>`;

server.listen(PORT, () => {
  console.log(`\n  Shifu OCR running at http://localhost:${PORT}\n`);
  console.log('  Tabs:');
  console.log('    Correct Text  — paste OCR output, get corrected text + safety flags');
  console.log('    Ward Census   — correct a full ward census row');
  console.log('    Learn         — teach the system from nurse corrections');
  console.log('    Structure     — compare sentences for structural invariance');
  console.log('    Stats         — view system stats, confusion pairs, metrics\n');
});
