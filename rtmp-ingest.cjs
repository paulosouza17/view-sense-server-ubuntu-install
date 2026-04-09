/**
 * rtmp-ingest.cjs — ViewSense RTMP Ingest Server
 *
 * Recebe streams RTMP das câmeras na porta 55935,
 * serve HLS na porta 8001 e gera snapshots periódicos.
 *
 * Portas:
 *   RTMP:  rtmp://SERVER_IP:55935/live/{streamKey}
 *   HLS:   http://SERVER_IP:8001/live/{streamKey}/index.m3u8
 *   Snap:  http://SERVER_IP:8001/snapshots/{streamKey}.jpg
 */

'use strict';

const NodeMediaServer = require('node-media-server');
const fs   = require('fs');
const path = require('path');
const http = require('http');
const url  = require('url');
const { execFile } = require('child_process');

// ─── Paths ────────────────────────────────────────────────────────────────────
const MEDIA_ROOT   = '/opt/viewsense/media';
const SNAP_DIR     = path.join(MEDIA_ROOT, 'snapshots');
const STREAMS_JSON = '/opt/viewsense/active_streams.json';

[MEDIA_ROOT, SNAP_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

// ─── Node-Media-Server (RTMP → HLS) ──────────────────────────────────────────
const nmsConfig = {
  rtmp: {
    port: 55935,
    chunk_size: 60000,
    gop_cache: true,
    ping: 30,
    ping_timeout: 60,
  },
  http: {
    port: 8001,
    mediaroot: MEDIA_ROOT,
    allow_origin: '*',
  },
};

const nms = new NodeMediaServer(nmsConfig);
const hlsProcesses = {};

// ─── Snapshot generator ───────────────────────────────────────────────────────
function captureSnapshot(streamKey) {
  const hlsUrl  = `http://127.0.0.1:8001/live/${streamKey}/index.m3u8`;
  const outFile = path.join(SNAP_DIR, `${streamKey}.jpg`);

  execFile('ffmpeg', [
    '-y', '-i', hlsUrl,
    '-frames:v', '1',
    '-q:v', '3',
    outFile,
  ], { timeout: 15000 }, (err) => {
    if (err) {
      // Retry via RTMP directly
      const rtmpUrl = `rtmp://127.0.0.1:55935/live/${streamKey}`;
      execFile('ffmpeg', [
        '-y', '-i', rtmpUrl,
        '-frames:v', '1',
        '-q:v', '3',
        outFile,
      ], { timeout: 10000 }, () => {});
    }
  });
}

// Periodic snapshot for all active streams (every 30s)
setInterval(() => {
  try {
    const active = JSON.parse(fs.readFileSync(STREAMS_JSON, 'utf8') || '[]');
    active.forEach(key => captureSnapshot(key));
  } catch (_) {}
}, 30000);

// Also capture on stream publish
nms.on('postPublish', (id, args) => {
  // node-media-server v4: args may be an object {streamPath} or a string
  const rawPath = (typeof args === 'string' ? args : (args && (args.streamPath || args.path || '')));
  if (!rawPath) return;
  const key = rawPath.replace('/live/', '');
  console.log(`[RTMP] Stream publicado: ${key}`);
  
  // Start manual HLS Transmuxing
  const streamDir = path.join(MEDIA_ROOT, 'live', key);
  fs.mkdirSync(streamDir, { recursive: true });
  const hlsOut = path.join(streamDir, 'index.m3u8');
  console.log(`[RTMP] Iniciando transcode HLS para ${key}`);
  const rtmpUrl = `rtmp://127.0.0.1:55935/live/${key}`;
  
  const ffmpeg = execFile('ffmpeg', [
    '-y', '-i', rtmpUrl,
    '-c:v', 'copy',
    '-c:a', 'copy',
    '-f', 'hls',
    '-hls_time', '2',
    '-hls_list_size', '6',
    '-hls_flags', 'delete_segments',
    hlsOut
  ], (err) => {
    if (err && err.signal !== 'SIGKILL') console.error(`[HLS] erro ${key}:`, err.message);
  });
  hlsProcesses[key] = ffmpeg;

  // Snapshot after 5s (stream needs time to buffer)
  setTimeout(() => captureSnapshot(key), 5000);
});

nms.on('donePublish', (id, args) => {
  const rawPath = (typeof args === 'string' ? args : (args && (args.streamPath || args.path || '')));
  if (!rawPath) return;
  const key = rawPath.replace('/live/', '');
  console.log(`[RTMP] Stream encerrado: ${key}`);
  
  if (hlsProcesses[key]) {
    console.log(`[RTMP] Parando transcode HLS para ${key}`);
    hlsProcesses[key].kill('SIGKILL');
    delete hlsProcesses[key];
  }
});

// ─── Minimal HTTP server for snapshots (CORS-safe) ──────────────────────────
// node-media-server já serve /live/* em :8001, mas o snapshot dir precisa de
// endpoint explícito porque não é um caminho de stream HLS.
const snapServer = http.createServer((req, res) => {
  const parsed = url.parse(req.url);
  const p = parsed.pathname || '';

  res.setHeader('Access-Control-Allow-Origin', '*');

  if (p.startsWith('/snapshots/')) {
    const filename = path.basename(p);
    const filepath = path.join(SNAP_DIR, filename);
    if (fs.existsSync(filepath)) {
      res.setHeader('Content-Type', 'image/jpeg');
      res.setHeader('Cache-Control', 'no-cache');
      fs.createReadStream(filepath).pipe(res);
    } else {
      res.writeHead(404);
      res.end('Snapshot não disponível');
    }
    return;
  }

  // Health check
  if (p === '/health') {
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ status: 'ok', rtmp_port: 55935, hls_port: 8001 }));
    return;
  }

  // ── Snapshot on demand: POST /snapshot-now/:streamKey ────────
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.writeHead(204); res.end(); return;
  }

  if (req.method === 'POST' && p.startsWith('/snapshot-now/')) {
    const streamKey = path.basename(p);
    if (!streamKey) { res.writeHead(400); res.end('Missing streamKey'); return; }
    const rtmpUrl  = `rtmp://127.0.0.1:55935/live/${streamKey}`;
    const outFile  = path.join(SNAP_DIR, `${streamKey}.jpg`);
    console.log(`[RTMP] Snapshot solicitado para: ${streamKey}`);
    execFile('ffmpeg', [
      '-y', '-i', rtmpUrl,
      '-frames:v', '1', '-q:v', '3',
      outFile,
    ], { timeout: 10000 }, (err) => {
      if (err) {
        console.error(`[RTMP] Snapshot error: ${err.message}`);
        res.writeHead(502);
        res.end(JSON.stringify({ error: err.message }));
      } else {
        console.log(`[RTMP] Snapshot salvo: ${outFile}`);
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, key: streamKey }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});


// ─── Start ───────────────────────────────────────────────────────────────────
nms.run();
snapServer.listen(8002, '0.0.0.0', () => {
  console.log('[ViewSense RTMP] Snapshot server em :8002');
});

console.log('[ViewSense RTMP] Servidor RTMP iniciado na porta 55935');
console.log('[ViewSense RTMP] HLS disponível em http://SERVER_IP:8001/live/{streamKey}/index.m3u8');
console.log('[ViewSense RTMP] Snapshots em http://SERVER_IP:8002/snapshots/{streamKey}.jpg');
