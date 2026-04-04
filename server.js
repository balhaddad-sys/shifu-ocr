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

// Boot the chat engine (uses the same ShifuEngine — shared brain)
const { ShifuChat } = require('./teaching/chat');
const chat = new ShifuChat(shifu.core);
console.log('Loading chat knowledge...');
const chatLoaded = chat.loadProjectData();
console.log(`Chat ready: ${chat.stats().vocabSize} words, ${chat.stats().knowledgeBase} sentences.`);
console.log('Shifu OCR ready.');

function jsonResponse(res, data, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': 'http://localhost:3737' });
  res.end(JSON.stringify(data, null, 2));
}

const MAX_BODY_SIZE = 10 * 1024 * 1024; // 10 MB
const MAX_TEXT_LENGTH = 20000; // 20k chars for text inputs (mitigate DoS)

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
    
    // JS Coherence Memory
    const result = shifu.learn(ocrRow, confirmedRow);
    
    // Deep Native Architectural Memory (Asynchronous Continuous Learning)
    const { spawn } = require('child_process');
    const pathModule = require('path');
    const learnLiveScript = pathModule.join(__dirname, 'shifu_ocr', 'learn_live.py');
    const bgTrain = spawn('python', [learnLiveScript, '--predicted', String(ocrRow), '--truth', String(confirmedRow)], { detached: true, stdio: 'ignore' });
    bgTrain.unref(); // Fire and forget so we don't stall the UI response
    
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

  // ── Chat API ────────────────────────────────────────────────────
  if (path === '/api/chat' && req.method === 'POST') {
    const { message } = await parseBody(req);
    if (!message) return jsonResponse(res, { error: 'message required' }, 400);
    if (message.length > MAX_TEXT_LENGTH) return jsonResponse(res, { error: 'message too long' }, 400);
    const result = chat.respond(message);
    return jsonResponse(res, result);
  }

  if (path === '/api/chat/reset' && req.method === 'POST') {
    chat.resetMemory();
    return jsonResponse(res, { message: 'Conversation memory reset.' });
  }

  if (path === '/api/chat/stats') {
    return jsonResponse(res, chat.stats());
  }

  if (path === '/api/chat/teach' && req.method === 'POST') {
    const { text, domain } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    if (text.length > MAX_TEXT_LENGTH) return jsonResponse(res, { error: 'text too long' }, 400);
    const sentences = text.split(/[.!?\n]+/).map(s => s.trim()).filter(s => s.length > 10);
    for (const s of sentences) chat.learn(s, domain || 'general');
    return jsonResponse(res, { taught: sentences.length, vocabSize: chat.stats().vocabSize });
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

// ── HTML UI ───────────────────────────────────────────────────────
const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SHIFU OCR | Clinical Insights</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0B0C10;--surface:#131920;--surface2:#1A2230;--surface3:#222D3A;--border:#2A3545;--text:#E0E6ED;--text2:#8B98A8;--text3:#5A6A7A;--gold:#C5A880;--gold2:#D4BC9A;--teal:#45A29E;--teal2:#5BC0BC;--teal-glow:rgba(69,162,158,.15);--red:#E74C5E;--green:#4ADE80;--yellow:#E2B93B;--radius:12px;--font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono','Consolas',monospace}
  body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column}

  /* Top Bar */
  .topbar{background:linear-gradient(180deg,var(--surface),var(--bg));border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;gap:14px}
  .topbar .lotus{font-size:1.4rem;filter:drop-shadow(0 0 8px var(--teal))}
  .topbar .brand{font-size:1.1rem;font-weight:700;letter-spacing:.04em;color:var(--gold)}
  .topbar .divider{width:1px;height:20px;background:var(--border)}
  .topbar .subtitle{font-size:.82rem;color:var(--teal);font-weight:500}
  .topbar .sep{flex:1}
  .topbar .pill{font-size:.72rem;padding:4px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:20px;color:var(--text2);display:flex;align-items:center;gap:6px;cursor:pointer}
  .topbar .pill .dot{width:7px;height:7px;border-radius:50%;background:var(--teal);box-shadow:0 0 6px var(--teal)}
  .topbar .pill:hover{border-color:var(--teal)}

  /* Layout */
  .layout{display:flex;flex:1;overflow:hidden}

  /* Left Sidebar */
  .sidebar-l{width:180px;min-width:180px;background:var(--surface);border-right:1px solid var(--border);padding:16px 0;display:flex;flex-direction:column;gap:2px}
  .sidebar-l .nav{padding:9px 18px;color:var(--text3);font-size:.82rem;cursor:pointer;border-left:3px solid transparent;transition:all .15s;display:flex;align-items:center;gap:8px}
  .sidebar-l .nav:hover{color:var(--text2);background:var(--teal-glow)}
  .sidebar-l .nav.active{color:var(--teal2);border-left-color:var(--teal);background:var(--teal-glow);font-weight:600}
  .sidebar-l .nav-head{font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);padding:18px 18px 4px;font-weight:700}
  .sidebar-l .nav-icon{font-size:.9rem;width:18px;text-align:center}

  /* Right Sidebar */
  .sidebar-r{width:200px;min-width:200px;background:var(--surface);border-left:1px solid var(--border);padding:16px;display:flex;flex-direction:column;gap:14px;font-size:.78rem}
  .sidebar-r .card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:12px}
  .sidebar-r .card h4{color:var(--gold);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
  .sidebar-r .card p{color:var(--text2);line-height:1.5}
  .sidebar-r .card .val{color:var(--teal2);font-family:var(--mono);font-weight:600}

  /* Main */
  .main{flex:1;overflow-y:auto;display:flex;flex-direction:column}
  .page{display:none;flex:1;flex-direction:column}
  .page.active{display:flex}

  /* Chat */
  .chat-area{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:16px}
  .chat-msg{max-width:75%;line-height:1.55;font-size:.85rem;position:relative;animation:fadeIn .3s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .chat-msg.user{align-self:flex-end;background:var(--surface3);color:var(--text);padding:12px 16px;border-radius:var(--radius) var(--radius) 4px var(--radius);border:1px solid var(--border)}
  .chat-msg.user .time{text-align:right}
  .chat-msg.shifu{align-self:flex-start;background:linear-gradient(135deg,var(--surface2),var(--surface));color:var(--text);padding:14px 16px;border-radius:var(--radius) var(--radius) var(--radius) 4px;border:1px solid var(--border);border-left:2px solid var(--teal);font-family:var(--mono);font-size:.82rem}
  .chat-msg.shifu .label{color:var(--teal);font-weight:600;font-size:.75rem;margin-bottom:6px;font-family:var(--font)}
  .chat-msg .quote{border-left:2px solid var(--gold);padding-left:10px;margin:8px 0;color:var(--text2);font-size:.8rem;background:rgba(197,168,128,.04);padding:8px 10px;border-radius:0 6px 6px 0}
  .chat-msg .time{font-size:.65rem;color:var(--text3);margin-top:6px}
  .chat-msg.system{align-self:center;color:var(--text3);font-size:.78rem;font-style:italic;padding:4px 16px}
  .chat-input-wrap{padding:14px 24px;border-top:1px solid var(--border);background:linear-gradient(180deg,transparent,rgba(69,162,158,.03))}
  .chat-input-row{display:flex;gap:8px;align-items:center;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:4px 4px 4px 14px;transition:border .2s}
  .chat-input-row:focus-within{border-color:var(--teal);box-shadow:0 0 12px rgba(69,162,158,.1)}
  .chat-input-row input{flex:1;background:none;border:none;color:var(--text);font-size:.9rem;font-family:var(--font);padding:8px 0;margin:0}
  .chat-input-row input:focus{outline:none}
  .chat-input-row .send-btn{background:var(--teal);color:var(--bg);border:none;border-radius:8px;padding:8px 18px;font-size:.82rem;font-weight:600;cursor:pointer;transition:all .15s;font-family:var(--font)}
  .chat-input-row .send-btn:hover{background:var(--teal2);box-shadow:0 0 12px rgba(69,162,158,.3)}

  /* Typing indicator */
  .typing{display:none;align-self:flex-start;padding:8px 16px;color:var(--teal);font-size:.78rem;font-style:italic}
  .typing.show{display:flex;align-items:center;gap:6px}
  .typing .dots span{display:inline-block;width:5px;height:5px;background:var(--teal);border-radius:50%;animation:bounce .6s infinite alternate}
  .typing .dots span:nth-child(2){animation-delay:.15s}
  .typing .dots span:nth-child(3){animation-delay:.3s}
  @keyframes bounce{to{opacity:.3;transform:translateY(-4px)}}

  /* Generic page styles */
  .page-content{padding:24px 28px;overflow-y:auto;flex:1}
  h2{font-size:1.05rem;font-weight:700;margin-bottom:16px;color:var(--gold)}
  h2 span{font-weight:400;color:var(--text3);font-size:.78rem;margin-left:8px}
  label{display:block;font-size:.7rem;color:var(--text3);margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em;font-weight:600}
  input,textarea{width:100%;padding:10px 14px;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:.9rem;font-family:var(--mono);margin-bottom:10px;transition:border .15s}
  input:focus,textarea:focus{outline:none;border-color:var(--teal)}
  textarea{min-height:80px;resize:vertical}
  .btn{padding:9px 20px;background:var(--teal);color:var(--bg);border:none;border-radius:8px;cursor:pointer;font-size:.82rem;font-weight:600;transition:all .15s;font-family:var(--font)}
  .btn:hover{background:var(--teal2);transform:translateY(-1px)}
  .btn-ghost{background:var(--surface2);color:var(--text2);border:1px solid var(--border)}
  .btn-ghost:hover{background:var(--surface3);color:var(--text)}
  .btn-danger{background:var(--red)}
  .btn-sm{padding:6px 12px;font-size:.75rem}
  .result{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-top:14px;font-family:var(--mono);font-size:.82rem;white-space:pre-wrap;max-height:500px;overflow-y:auto;line-height:1.6}
  .word{display:inline-block;padding:2px 7px;margin:1px;border-radius:5px;font-size:.82rem}
  .word.exact{background:rgba(69,162,158,.12);color:var(--teal2)}
  .word.high_confidence{background:rgba(69,162,158,.08);color:#67e8f9}
  .word.corrected_verify,.word.verify{background:rgba(226,185,59,.1);color:#fde68a}
  .word.low_confidence,.word.unknown{background:rgba(231,76,94,.1);color:#fca5a5}
  .word.number,.word.title,.word.dosage,.word.room_code,.word.short,.word.punctuation,.word.passthrough{background:var(--surface2);color:var(--text3)}
  .word.digraph_corrected{background:rgba(197,168,128,.12);color:var(--gold2)}
  .word.DANGER_medication_ambiguity{background:var(--red);color:white;font-weight:bold}
  .flag{display:inline-block;padding:3px 8px;margin:2px;border-radius:5px;font-size:.75rem;font-weight:600}
  .flag.danger{background:var(--red);color:white}
  .flag.warning{background:var(--yellow);color:#000}
  .flag.error{background:#ea580c;color:white}
  .decision{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.8rem;font-weight:700;margin-top:6px}
  .decision.accept{background:rgba(74,222,128,.12);color:var(--green)}
  .decision.verify{background:rgba(226,185,59,.1);color:var(--yellow)}
  .decision.reject{background:rgba(231,76,94,.1);color:var(--red)}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .actions{display:flex;gap:8px;margin-top:10px}
  @media(max-width:900px){.sidebar-l,.sidebar-r{display:none}.main{padding:0}}
</style>
</head>
<body>
<div class="topbar">
  <span class="lotus">&#x2740;</span>
  <span class="brand">SHIFU OCR</span>
  <span class="divider"></span>
  <span class="subtitle">Clinical Insights</span>
  <span class="sep"></span>
  <div class="pill" onclick="go('stats')"><span class="dot"></span> <span id="topVocab">Loading...</span></div>
</div>
<div class="layout">
  <div class="sidebar-l">
    <div class="nav-head">Chat</div>
    <div class="nav active" onclick="go('chat')"><span class="nav-icon">&#x2728;</span> New Conversation</div>
    <div class="nav-head">Clinical Tools</div>
    <div class="nav" onclick="go('correct')"><span class="nav-icon">&#x270F;</span> Correct Text</div>
    <div class="nav" onclick="go('ward')"><span class="nav-icon">&#x1F3E5;</span> Ward Census</div>
    <div class="nav" onclick="go('learn')"><span class="nav-icon">&#x1F9E0;</span> Learn</div>
    <div class="nav" onclick="go('upload')"><span class="nav-icon">&#x1F4C4;</span> Upload File</div>
    <div class="nav-head">Analysis</div>
    <div class="nav" onclick="go('structure')"><span class="nav-icon">&#x1F50D;</span> Structure</div>
    <div class="nav" onclick="go('stats')"><span class="nav-icon">&#x2699;</span> Settings</div>
  </div>
  <div class="main">

  <!-- Chat (default) -->
  <div id="chat" class="page active">
    <div class="chat-area" id="chatMessages">
      <div class="chat-msg system">How may I assist with your clinical cases today?</div>
    </div>
    <div class="typing" id="typingIndicator"><span>Shifu is thinking</span><span class="dots"><span></span><span></span><span></span></span></div>
    <div class="chat-input-wrap">
      <div class="chat-input-row">
        <input id="chatInput" placeholder="Ask Shifu OCR clinical queries..." onkeydown="if(event.key==='Enter')sendChat()">
        <button class="send-btn" onclick="sendChat()">Send</button>
      </div>
    </div>
  </div>

  <!-- Correct Text -->
  <div id="correct" class="page">
    <div class="page-content">
    <h2>Correct Text <span>Paste raw OCR output for correction + safety flags</span></h2>
    <label>OCR Text</label>
    <textarea id="ocrInput" placeholder="e.g. Levetiracetarn 5OOmg for seizure prophylaxis">Levetiracetarn 5OOmg for seizure prophylaxis</textarea>
    <button class="btn" onclick="correctText()">Correct</button>
    <div id="correctResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab 2: Ward Census Row -->
  <div id="ward" class="page">
    <h2>Ward Census <span>Correct a full ward census row</span></h2>
    <div class="grid2">
      <div><label>Patient</label><input id="wPatient" placeholder="e.g. Bader A1athoub"></div>
      <div><label>Diagnosis</label><input id="wDiagnosis" placeholder="e.g. lschemic str0ke"></div>
      <div><label>Doctor</label><input id="wDoctor" placeholder="e.g. Dr. Hisharn"></div>
      <div><label>Medication</label><input id="wMedication" placeholder="e.g. Levetiracetarn"></div>
      <div><label>Room</label><input id="wRoom" placeholder="e.g. 3O2A"></div>
      <div><label>Status</label><input id="wStatus" placeholder="e.g. Red"></div>
    </div>
    <div class="actions">
      <button class="btn" onclick="correctRow()">Correct Row</button>
    </div>
    <div id="wardResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab 3: Learn -->
  <div id="learn" class="page">
    <h2>Learn <span>Teach from nurse corrections</span></h2>
    <div class="grid2">
      <div>
        <label>OCR Column</label><input id="learnCol" value="Diagnosis">
        <label>OCR Text</label><input id="learnOcr" placeholder="e.g. str0ke">
      </div>
      <div>
        <label>&nbsp;</label><input disabled style="visibility:hidden">
        <label>Confirmed Text</label><input id="learnConfirmed" placeholder="e.g. stroke">
      </div>
    </div>
    <div class="actions">
      <button class="btn" onclick="learnCorrection()">Learn</button>
      <button class="btn btn-danger btn-sm" onclick="undoLearn()">Undo Last</button>
    </div>
    <div id="learnResult" class="result" style="display:none"></div>
    <div style="margin-top:16px"><label>Correction History</label></div>
    <div id="historyResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab: Upload File -->
  <div id="upload" class="page">
    <h2>Upload File <span>PDF, CSV, TSV, TXT, or image</span></h2>
    <div style="border: 2px dashed #2a3050; border-radius: 8px; padding: 40px; text-align: center; margin-bottom: 16px; cursor: pointer" id="dropZone" onclick="document.getElementById('fileInput').click()" ondrop="handleDrop(event)" ondragover="event.preventDefault(); this.style.borderColor='#7eb8ff'" ondragleave="this.style.borderColor='#2a3050'">
      <p style="color: #667; font-size: 1rem;">Drop a file here or click to browse</p>
      <p style="color: #445; font-size: 0.8rem; margin-top: 8px;">Supports: PDF, CSV, TSV, TXT, PNG, JPG, TIFF, BMP</p>
      <input type="file" id="fileInput" style="display:none" accept=".pdf,.csv,.tsv,.txt,.png,.jpg,.jpeg,.tiff,.tif,.bmp" onchange="uploadFile(this.files[0])">
    </div>
    <div id="uploadStatus" style="display:none; color: #7eb8ff; margin-bottom: 12px;"></div>
    <div id="uploadResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab 4: Structure -->
  <div id="structure" class="page">
    <h2>Structure <span>Compare structural invariance</span></h2>
    <input id="structA" placeholder="e.g. doctor treats patient with medication">
    <input id="structB" placeholder="e.g. patient was treated by the doctor with medication">
    <div class="actions">
      <button onclick="compareStructure()">Compare</button>
      <button class="secondary" onclick="extractRoles()">Extract Roles (sentence A)</button>
    </div>
    <div id="structResult" class="result" style="display:none"></div>
  </div>

  <!-- Stats -->
  <div id="stats" class="page">
    <div class="page-content">
    <h2>System Stats</h2>
    <button class="btn btn-ghost" onclick="loadStats()">Refresh</button>
    <div id="statsResult" class="result" style="display:none"></div>
    </div>
  </div>

  </div><!-- /main -->

  <div class="sidebar-r" id="sidebarR">
    <div class="card">
      <h4>Domain</h4>
      <p id="rDomain" style="color:var(--teal2)">Detecting...</p>
    </div>
    <div class="card">
      <h4>Concepts</h4>
      <p id="rConcepts" class="val">-</p>
    </div>
    <div class="card">
      <h4>Knowledge Base</h4>
      <p><span class="val" id="rKB">-</span> sentences</p>
    </div>
    <div class="card">
      <h4>Vocabulary</h4>
      <p><span class="val" id="rVocab">-</span> words</p>
    </div>
  </div>

</div><!-- /layout -->

<script>
const API = '';

function go(name) {
  document.querySelectorAll('.nav').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  if(event&&event.target)event.target.classList.add('active');
  if(name==='stats')loadStats();
}
// Compat
function switchTab(n){go(n)}

async function api(path, body) {
  const res = await fetch(API + path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function renderWords(words) {
  if (!words || !words.length) return '';
  return words.map(w => {
    const cls = esc(w.flag || 'unknown');
    const orig = esc(w.original);
    const corr = esc(w.corrected || w.original);
    const tip = w.original !== w.corrected ? orig + ' -> ' + corr : orig;
    return '<span class="word ' + cls + '" title="' + tip + ' [' + cls + '] conf:' + (w.confidence||0).toFixed(2) + '">' + corr + '</span>';
  }).join(' ');
}

function renderFlags(flags) {
  if (!flags || !flags.length) return '';
  return flags.map(f => '<span class="flag ' + esc(f.severity||'warning') + '">' + esc(f.message||f.status) + '</span>').join(' ');
}

function show(id, html) {
  const el = document.getElementById(id);
  el.style.display = 'block';
  el.innerHTML = html;
}

async function correctText() {
  const text = document.getElementById('ocrInput').value;
  if (!text) return;
  const r = await api('/api/correct', { text });
  let html = '<b>Output:</b> ' + renderWords(r.words) + '\\n\\n';
  html += '<b>Text:</b> ' + esc(r.output||'') + '\\n';
  html += '<b>Decision:</b> <span class="decision ' + (r.decision||'verify') + '">' + (r.decision||'?') + '</span>\\n';
  html += '<b>Coherence:</b> ' + (r.coherence||0).toFixed(4) + '\\n';
  html += '<b>Avg Confidence:</b> ' + (r.avgConfidence||0) + '\\n';
  if (r.safetyFlags && r.safetyFlags.length) html += '\\n<b>Safety Flags:</b> ' + renderFlags(r.safetyFlags);
  show('correctResult', html);
}

async function correctRow() {
  const row = {};
  for (const col of ['Patient','Diagnosis','Doctor','Medication','Room','Status']) {
    const v = document.getElementById('w' + col).value;
    if (v) row[col] = v;
  }
  if (!Object.keys(row).length) return;
  const r = await api('/api/correct-row', { row });
  let html = '<b>Decision:</b> <span class="decision ' + (r.decision||'verify') + '">' + (r.decision||'?') + '</span>\\n\\n';
  for (const [col, data] of Object.entries(r.corrected || {})) {
    html += '<b>' + esc(col) + ':</b> ' + esc(data.input||'') + ' -> ' + renderWords(data.words) + '\\n';
  }
  if (r.safetyFlags && r.safetyFlags.length) html += '\\n<b>Safety Flags:</b>\\n' + renderFlags(r.safetyFlags);
  show('wardResult', html);
}

async function learnCorrection() {
  const col = document.getElementById('learnCol').value || 'Diagnosis';
  const ocr = document.getElementById('learnOcr').value;
  const confirmed = document.getElementById('learnConfirmed').value;
  if (!ocr || !confirmed) return;
  const ocrRow = {}; ocrRow[col] = ocr;
  const confirmedRow = {}; confirmedRow[col] = confirmed;
  const r = await api('/api/learn', { ocrRow, confirmedRow });
  let html = '<b>Accepted:</b> ' + r.accepted + '\\n';
  if (r.rejected && r.rejected.length) html += '<b>Rejected:</b> ' + esc(JSON.stringify(r.rejected)) + '\\n';
  if (r.reason) html += '<b>Reason:</b> ' + esc(r.reason) + '\\n';
  show('learnResult', html);
  loadHistory();
}

async function undoLearn() {
  const r = await api('/api/undo', {});
  show('learnResult', '<b>Undo:</b> ' + esc(JSON.stringify(r, null, 2)));
  loadHistory();
}

async function loadHistory() {
  const r = await api('/api/history');
  if (!r || !r.length) { show('historyResult', 'No corrections yet.'); return; }
  let html = r.map((h,i) => '#' + esc(h.id) + ' [' + esc(h.timestamp) + '] columns: ' + esc((h.columns||[]).join(', '))).join('\\n');
  show('historyResult', html);
}

async function compareStructure() {
  const a = document.getElementById('structA').value;
  const b = document.getElementById('structB').value;
  if (!a || !b) return;
  const r = await api('/api/compare-structure', { sentA: a, sentB: b });
  let html = '<b>Invariance Score:</b> ' + (r.invariance||0).toFixed(4) + '\\n\\n';
  html += '<b>Sentence A roles:</b>\\n';
  html += '  Agent: ' + esc(r.rolesA?.agent||'-') + ', Action: ' + esc(r.rolesA?.action||'-') + ', Patient: ' + esc(r.rolesA?.patient||'-') + ' (passive: ' + (r.rolesA?.passive||false) + ')\\n';
  html += '<b>Sentence B roles:</b>\\n';
  html += '  Agent: ' + esc(r.rolesB?.agent||'-') + ', Action: ' + esc(r.rolesB?.action||'-') + ', Patient: ' + esc(r.rolesB?.patient||'-') + ' (passive: ' + (r.rolesB?.passive||false) + ')\\n\\n';
  const c = r.comparison || {};
  html += '<b>Agent similarity:</b> ' + (c.agentSim||0).toFixed(3) + '\\n';
  html += '<b>Action similarity:</b> ' + (c.actionSim||0).toFixed(3) + '\\n';
  html += '<b>Patient similarity:</b> ' + (c.patientSim||0).toFixed(3) + '\\n';
  if (c.swapDetected) html += '\\n<span class="flag danger">ROLE SWAP DETECTED</span>';
  show('structResult', html);
}

async function extractRoles() {
  const a = document.getElementById('structA').value;
  if (!a) return;
  const r = await api('/api/roles', { text: a });
  show('structResult', esc(JSON.stringify(r, null, 2)));
}

async function loadStats() {
  const r = await api('/api/stats');
  let html = '<b>Core Engine</b>\\n';
  html += '  Vocabulary: ' + (r.core?.vocabulary||0) + ' words\\n';
  html += '  Sentences fed: ' + (r.core?.sentences||0) + '\\n';
  html += '  Resonance pairs: ' + (r.core?.resonancePairs||0) + '\\n\\n';
  html += '<b>Learning Engine</b>\\n';
  html += '  Total corrections: ' + (r.learning?.totalCorrections||0) + '\\n';
  html += '  Confusion pairs: ' + (r.learning?.confusionPairs||0) + '\\n';
  html += '  Vocabulary size: ' + (r.learning?.vocabularySize||0) + '\\n';
  html += '  Learned words: ' + (r.learning?.learnedWords||0) + '\\n';
  html += '  Context chains: ' + (r.learning?.contextChains||0) + '\\n\\n';
  if (r.learning?.topConfusions?.length) {
    html += '<b>Top Confusions:</b>\\n';
    for (const c of r.learning.topConfusions) html += '  ' + esc(c.pair) + ' (' + c.count + 'x, cost: ' + c.cost.toFixed(3) + ')\\n';
  }
  show('statsResult', html);
}

function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.style.borderColor = '#2a3050';
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  if (!file) return;
  document.getElementById('uploadStatus').style.display = 'block';
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  document.getElementById('uploadStatus').innerHTML = 'Processing <b>' + esc(file.name) + '</b> (' + sizeMB + ' MB)...<br><span style="color:#94a3b8;font-size:0.85rem">Large PDFs (1000+ pages) may take a few minutes for text extraction.</span>';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(API + '/api/upload?filename=' + encodeURIComponent(file.name), {
      method: 'POST',
      body: formData,
    });
    const r = await res.json();
    document.getElementById('uploadStatus').textContent = 'Done: ' + file.name;

    let html = '<b>Type:</b> ' + esc(r.type||'unknown') + '\\n';
    if (r.extractionMethod) {
      html += '<b>Extraction:</b> ' + esc(r.extractionMethod) + '\\n';
      if (r.extractionDetail) html += '<span style="color:#94a3b8;font-size:0.85rem">' + esc(r.extractionDetail) + '</span>\\n';
      if (r.pageCount) html += '<b>Pages:</b> ' + r.pageCount + '\\n';
      if (r.charCount) html += '<b>Characters:</b> ' + r.charCount.toLocaleString() + '\\n';
    }
    if (r.sourceMode) html += '<b>Mode:</b> ' + esc(r.sourceMode) + '\\n';
    if (r.error) html += '<b style="color:#fca5a5">Error:</b> ' + esc(r.error) + '\\n';
    if (r.overallDecision) html += '<b>Decision:</b> <span class="decision ' + r.overallDecision + '">' + r.overallDecision + '</span>\\n';

    // Raw extracted text (what the PDF/file reader saw)
    if (r.rawText) {
      html += '\\n<b style="color:#94a3b8">--- Raw Extracted Text ---</b>\\n';
      html += '<span style="color:#667">' + r.rawText.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\\n';
    }
    if (r.rawLines && r.rawLines.length && !r.rawText) {
      html += '\\n<b style="color:#94a3b8">--- Raw Extracted Lines ---</b>\\n';
      for (const rl of r.rawLines) {
        html += '<span style="color:#667">' + rl.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\\n';
      }
    }

    // Reconstructed table (from spatial coordinates)
    if (r.table && r.table.rows && r.table.rows.length > 0) {
      html += '\\n<b style="color:#7eb8ff">--- Reconstructed Table (' + r.table.columns + ' columns, ' + r.table.rows.length + ' rows) ---</b>\\n';
      html += '<table style="border-collapse:collapse;width:100%;font-size:0.8rem;margin:8px 0">';
      for (let ri = 0; ri < r.table.rows.length; ri++) {
        const row = r.table.rows[ri];
        const isHeader = ri <= 2 && row.some(c => /patient|name|diagnosis|doctor|status|room|ward/i.test(c));
        html += '<tr>';
        for (const cell of row) {
          const tag = isHeader ? 'th' : 'td';
          html += '<' + tag + ' style="border:1px solid #2a3050;padding:4px 8px;text-align:left;color:' + (isHeader ? '#7eb8ff' : '#ccc') + '">' + esc(cell || '') + '</' + tag + '>';
        }
        html += '</tr>';
      }
      html += '</table>\\n';
    }

    // Corrected lines (text/PDF)
    if (r.lines && r.lines.length) {
      html += '\\n<b style="color:#7eb8ff">--- Corrected Output ---</b>\\n';
      for (let li = 0; li < r.lines.length; li++) {
        const line = r.lines[li];
        const rawLine = (r.rawLines && r.rawLines[li]) || '';
        if (rawLine) html += '<span style="color:#445;font-size:0.75rem">RAW: ' + rawLine.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\\n';
        if (line && line.words) {
          html += renderWords(line.words) + '\\n';
          if (line.safetyFlags && line.safetyFlags.length) html += renderFlags(line.safetyFlags) + '\\n';
        } else if (line && line.corrected) {
          html += esc(line.corrected) + '\\n';
        }
        html += '\\n';
      }
    }

    // Rows (CSV)
    if (r.rows && r.rows.length) {
      html += '\\n<b>Rows (' + r.rows.length + '):</b>\\n';
      for (let i = 0; i < r.rows.length; i++) {
        const row = r.rows[i];
        html += '\\n<b>Row ' + (i+1) + '</b> <span class="decision ' + (row.decision||'verify') + '">' + (row.decision||'?') + '</span>\\n';
        if (row.corrected) {
          for (const [col, data] of Object.entries(row.corrected)) {
            html += '  <b>' + esc(col) + ':</b> ' + esc(data.input||'') + ' -> ';
            if (data.words) html += renderWords(data.words);
            else html += esc(data.output||data);
            html += '\\n';
          }
        }
        if (row.safetyFlags && row.safetyFlags.length) {
          html += '  ' + renderFlags(row.safetyFlags) + '\\n';
        }
      }
    }

    // Summary
    if (r.summary) {
      html += '\\n<b>Summary:</b> ' + r.summary.total + ' rows, ' + r.summary.accepted + ' accepted, ' + r.summary.needsVerification + ' verify, ' + r.summary.rejected + ' rejected';
    }

    show('uploadResult', html);
  } catch (err) {
    document.getElementById('uploadStatus').textContent = 'Error: ' + err.message;
    show('uploadResult', '<b>Error:</b> ' + err.message);
  }
}

// ── Chat Functions ──────────────────────────────────────────────
function timeStr(){const d=new Date();return d.getHours()+':'+String(d.getMinutes()).padStart(2,'0')+' '+(['AM','PM'][d.getHours()>=12?1:0])}

function addChatMsg(cls, html) {
  const box = document.getElementById('chatMessages');
  const d = document.createElement('div');
  d.className = 'chat-msg ' + cls;
  if(cls==='shifu') d.innerHTML='<div class="label">Shifu OCR</div>'+html+'<div class="time">'+timeStr()+'</div>';
  else if(cls==='user') d.innerHTML=html+'<div class="time">'+timeStr()+'</div>';
  else d.innerHTML=html;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

function showTyping(v){document.getElementById('typingIndicator').classList.toggle('show',v)}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addChatMsg('user', esc(msg));
  showTyping(true);

  try {
    const r = await fetch(API + '/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:msg}) });
    showTyping(false);
    const data = await r.json();
    let html = '';
    // Show OCR corrections if any
    if (data.corrections && data.corrections.length) {
      html += '<div style="color:var(--gold);font-size:.75rem;margin-bottom:6px;font-family:var(--font)">';
      html += data.corrections.map(c => '<s style="color:var(--text3)">' + esc(c.from) + '</s> &rarr; <b>' + esc(c.to) + '</b>').join(' &nbsp; ');
      html += '</div>';
    }
    const lines = (data.text || 'No response').split('\\n');
    for (const l of lines) {
      if (l.startsWith('> ')) html += '<div class="quote">' + esc(l.slice(2)) + '</div>';
      else if(l.trim()) html += '<div>' + esc(l) + '</div>';
    }
    addChatMsg('shifu', html);
    // Update right sidebar
    document.getElementById('rDomain').textContent = data.domain ? data.domain.charAt(0).toUpperCase()+data.domain.slice(1) : 'General';
    document.getElementById('rConcepts').textContent = (data.concepts||[]).join(', ') || '-';
    document.getElementById('rKB').textContent = data.kbSize||'-';
    document.getElementById('rVocab').textContent = (data.vocabSize||0).toLocaleString();
    document.getElementById('topVocab').textContent = (data.vocabSize||0).toLocaleString() + ' words';
  } catch(e) {
    showTyping(false);
    addChatMsg('system', 'Error: ' + e.message);
  }
}

async function resetChat() {
  await fetch(API + '/api/chat/reset', { method:'POST' });
  document.getElementById('chatMessages').innerHTML = '';
  addChatMsg('system', 'How may I assist with your clinical cases today?');
  document.getElementById('rDomain').textContent='Detecting...';
  document.getElementById('rConcepts').textContent='-';
}

async function teachChat() {
  const text = document.getElementById('teachInput').value.trim();
  const domain = document.getElementById('teachDomain').value.trim();
  if (!text) return;
  try {
    const r = await fetch(API + '/api/chat/teach', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text, domain: domain||undefined}) });
    const data = await r.json();
    document.getElementById('teachInput').value = '';
    addChatMsg('system', 'Learned ' + data.taught + ' sentences. Vocabulary now: ' + data.vocabSize);
    document.getElementById('rVocab').textContent = (data.vocabSize||0).toLocaleString();
    document.getElementById('rKB').textContent = data.kbSize || '-';
  } catch(e) { alert('Error: ' + e.message); }
}

// Load initial stats
fetch(API+'/api/chat/stats').then(r=>r.json()).then(d=>{
  document.getElementById('topVocab').textContent=(d.vocabSize||0).toLocaleString()+' words';
  document.getElementById('rVocab').textContent=(d.vocabSize||0).toLocaleString();
  document.getElementById('rKB').textContent=d.knowledgeBase||'-';
}).catch(()=>{});
</script>
</body>
</html>`;

server.listen(PORT, '127.0.0.1', () => {
  console.log(`\n  Shifu OCR running at http://127.0.0.1:${PORT}\n`);
  console.log('  Tabs:');
  console.log('    Correct Text  — paste OCR output, get corrected text + safety flags');
  console.log('    Ward Census   — correct a full ward census row');
  console.log('    Learn         — teach the system from nurse corrections');
  console.log('    Chat          — talk to Shifu (no LLM, pure neural network)');
  console.log('    Structure     — compare sentences for structural invariance');
  console.log('    Stats         — view system stats, confusion pairs, metrics\n');
});
