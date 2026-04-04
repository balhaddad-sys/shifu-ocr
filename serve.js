// Shifu v2 — Lightweight PWA Server
// Serves the chatbot + streams the trained brain state.
// Usage: node serve.js [port] [state-file]
// Default: port 3737, state from .state/embryo_v2_tutored.json

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.argv[2]) || 3737;
const STATE_FILE = process.argv[3] || '.state/embryo_v2_tutored.json';
const ROOT = __dirname;

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.ico': 'image/x-icon', '.txt': 'text/plain'
};

function serveFile(res, filePath, mime) {
  const stat = fs.statSync(filePath);
  res.writeHead(200, {
    'Content-Type': mime || 'application/octet-stream',
    'Content-Length': stat.size,
    'Cache-Control': 'public, max-age=3600'
  });
  fs.createReadStream(filePath).pipe(res);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;

  // CORS for local dev
  res.setHeader('Access-Control-Allow-Origin', '*');

  // /state — stream the trained brain (can be 400MB+)
  if (pathname === '/state') {
    const statePath = path.resolve(ROOT, STATE_FILE);
    if (!fs.existsSync(statePath)) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end('{"error":"no trained state"}');
      return;
    }
    const stat = fs.statSync(statePath);
    console.log(`Streaming brain: ${statePath} (${Math.round(stat.size / 1024 / 1024)}MB)`);
    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Content-Length': stat.size,
      'Cache-Control': 'no-cache'
    });
    fs.createReadStream(statePath).pipe(res);
    return;
  }

  // / → shifu_chatbot.html
  let filePath;
  if (pathname === '/' || pathname === '/index.html') {
    filePath = path.join(ROOT, 'shifu_chatbot.html');
  } else {
    // Serve static files from root
    filePath = path.join(ROOT, pathname);
  }

  // Prevent directory traversal
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403); res.end('Forbidden'); return;
  }

  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not found');
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  serveFile(res, filePath, MIME[ext]);
});

server.listen(PORT, () => {
  const stateExists = fs.existsSync(path.resolve(ROOT, STATE_FILE));
  console.log(`Shifu v2 — http://localhost:${PORT}`);
  console.log(`Brain: ${stateExists ? STATE_FILE + ' (' + Math.round(fs.statSync(path.resolve(ROOT, STATE_FILE)).size / 1024 / 1024) + 'MB)' : 'none (starting empty)'}`);
  console.log(`PWA ready for install.`);
});
