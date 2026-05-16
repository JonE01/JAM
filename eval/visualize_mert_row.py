"""
Visualize a random row from the MERT dataset in a local web browser.

Usage:
    python visualize_mert_row.py \
        --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
        --port 9999

Then SSH tunnel and open http://localhost:9999 in your browser.
"""

import argparse
import base64
import json
import os
import random
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
from datasets import load_from_disk


def make_handler(dataset, window_size, feature_dim):

    class Handler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/" or parsed.path == "/index.html":
                self._serve_html()
            elif parsed.path == "/api/random":
                self._serve_random_row()
            elif parsed.path == "/api/row":
                qs = parse_qs(parsed.query)
                idx = int(qs.get("idx", [0])[0])
                self._serve_row(idx)
            elif parsed.path == "/api/audio":
                qs = parse_qs(parsed.query)
                idx = int(qs.get("idx", [0])[0])
                self._serve_audio(idx)
            else:
                self.send_error(404)

        def _serve_html(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        def _serve_row(self, idx):
            idx = max(0, min(idx, len(dataset) - 1))
            self._send_row_json(idx)

        def _serve_random_row(self):
            idx = random.randint(0, len(dataset) - 1)
            self._send_row_json(idx)

        def _serve_audio(self, idx):
            idx = max(0, min(idx, len(dataset) - 1))
            row = dataset[idx]
            audio_b64 = row.get("audio_b64")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(audio_bytes)))
                self.end_headers()
                self.wfile.write(audio_bytes)
            else:
                self.send_error(404, "No audio for this row")

        def _send_row_json(self, idx):
            row = dataset[idx]

            # Convert MERT output bytes to tensor, then to stats for JSON
            mert_bytes = row.get("MERT_output")
            if mert_bytes is not None:
                tensor = torch.frombuffer(
                    bytearray(mert_bytes), dtype=torch.float32
                ).reshape(window_size, feature_dim)
                mert_stats = {
                    "shape": list(tensor.shape),
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                    "norm_per_timestep": tensor.norm(dim=1).tolist(),
                }
            else:
                mert_stats = None

            # Build metadata
            metadata = {}
            skip_cols = {"MERT_output", "audio_b64"}
            for col in dataset.column_names:
                if col in skip_cols:
                    continue
                val = row[col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                elif hasattr(val, 'isoformat'):
                    val = val.isoformat()
                metadata[col] = val

            payload = {
                "index": idx,
                "total_rows": len(dataset),
                "metadata": metadata,
                "mert": mert_stats,
                "has_audio": "audio_b64" in row and row["audio_b64"] is not None,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def log_message(self, format, *args):
            print(f"  {args[0]}")

    return Handler


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MERT Dataset Explorer</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1a1a24;
    --border: #2a2a3a;
    --text: #e0e0e8;
    --text-dim: #8888a0;
    --accent: #6c5ce7;
    --accent2: #00cec9;
    --accent3: #fd79a8;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  .grain {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    opacity: 0.03; pointer-events: none; z-index: 9999;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  }

  header {
    padding: 2rem 3rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    backdrop-filter: blur(20px);
    position: sticky; top: 0; z-index: 100;
    background: rgba(10, 10, 15, 0.85);
  }

  header h1 {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .controls {
    display: flex; gap: 0.75rem; align-items: center;
  }

  .controls input {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.85rem;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    width: 100px;
    text-align: center;
  }

  button {
    background: var(--accent);
    color: white;
    border: none;
    font-family: var(--mono);
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.55rem 1.2rem;
    border-radius: 6px;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
  }

  button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(108, 92, 231, 0.4); }
  button.secondary { background: var(--surface2); border: 1px solid var(--border); }
  button.secondary:hover { border-color: var(--accent); box-shadow: none; }

  .container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem 3rem;
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 2rem;
    animation: fadeIn 0.5s ease;
  }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }

  .panel-header {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
  }

  .panel-body { padding: 1.25rem; }

  .meta-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(42, 42, 58, 0.5);
    font-size: 0.85rem;
  }

  .meta-row:last-child { border-bottom: none; }
  .meta-key { color: var(--text-dim); font-family: var(--mono); font-size: 0.75rem; }
  .meta-val { color: var(--text); text-align: right; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta-val.wrap { white-space: normal; word-break: break-word; }

  .tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    margin: 0.15rem;
    font-family: var(--mono);
  }

  .tag.genre { border-color: var(--accent); color: var(--accent); }
  .tag.tag-item { border-color: var(--accent2); color: var(--accent2); }

  .right-col { display: flex; flex-direction: column; gap: 2rem; }

  .audio-section audio {
    width: 100%;
    margin-top: 0.5rem;
    border-radius: 8px;
  }

  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
  }

  .stat-card {
    background: var(--surface2);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }

  .stat-card .val {
    font-family: var(--mono);
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent2);
  }

  .stat-card .label {
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  #norm-canvas {
    width: 100%;
    height: 150px;
    border-radius: 8px;
  }

  .loading {
    display: flex; align-items: center; justify-content: center;
    padding: 4rem;
    font-family: var(--mono);
    color: var(--text-dim);
    font-size: 0.85rem;
  }

  .row-badge {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--accent2);
    background: rgba(0, 206, 201, 0.1);
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
  }

  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  @media (max-width: 900px) {
    .container { grid-template-columns: 1fr; padding: 1rem; }
    header { padding: 1rem 1.5rem; flex-wrap: wrap; gap: 0.5rem; }
  }
</style>
</head>
<body>
<div class="grain"></div>

<header>
  <h1>MERT Dataset Explorer</h1>
  <div class="controls">
    <input type="number" id="row-input" placeholder="Row #" min="0">
    <button class="secondary" onclick="loadRow()">Go</button>
    <button onclick="loadRandom()">Random</button>
  </div>
</header>

<div class="container" id="main">
  <div class="loading">Loading...</div>
</div>

<script>
async function loadRandom() {
  const res = await fetch('/api/random');
  const data = await res.json();
  render(data);
}

async function loadRow() {
  const idx = document.getElementById('row-input').value;
  if (idx === '') return;
  const res = await fetch(`/api/row?idx=${idx}`);
  const data = await res.json();
  render(data);
}

function render(data) {
  const main = document.getElementById('main');
  document.getElementById('row-input').value = data.index;
  document.getElementById('row-input').max = data.total_rows - 1;

  const meta = data.metadata;
  const mert = data.mert;

  const title = meta.title || 'Untitled';
  const artist = meta.artist || 'Unknown';

  let genresHTML = '';
  if (meta.genres && meta.genres.length) {
    genresHTML = meta.genres.map(g => `<span class="tag genre">${g}</span>`).join('');
  }

  let tagsHTML = '';
  if (meta.tags && meta.tags.length) {
    tagsHTML = meta.tags.map(t => `<span class="tag tag-item">${t}</span>`).join('');
  }

  const skipKeys = new Set(['title', 'artist', 'genres', 'tags', 'url', 'artist_url', 'artist_website', 'album_url']);
  let otherMeta = '';
  for (const [key, val] of Object.entries(meta)) {
    if (skipKeys.has(key)) continue;
    if (val === null || val === '' || (Array.isArray(val) && val.length === 0)) continue;
    let display = val;
    if (typeof val === 'object') display = JSON.stringify(val);
    otherMeta += `<div class="meta-row"><span class="meta-key">${key}</span><span class="meta-val">${display}</span></div>`;
  }

  let linksHTML = '';
  if (meta.url) linksHTML += `<div class="meta-row"><span class="meta-key">track</span><span class="meta-val"><a href="${meta.url}" target="_blank">open ↗</a></span></div>`;
  if (meta.artist_url) linksHTML += `<div class="meta-row"><span class="meta-key">artist</span><span class="meta-val"><a href="${meta.artist_url}" target="_blank">open ↗</a></span></div>`;
  if (meta.album_url) linksHTML += `<div class="meta-row"><span class="meta-key">album</span><span class="meta-val"><a href="${meta.album_url}" target="_blank">open ↗</a></span></div>`;

  const leftPanel = `
    <div>
      <div class="panel" style="margin-bottom:1.5rem">
        <div class="panel-header">Track Info <span class="row-badge" style="float:right">row ${data.index} / ${data.total_rows}</span></div>
        <div class="panel-body">
          <h2 style="font-size:1.2rem;margin-bottom:0.3rem">${title}</h2>
          <p style="color:var(--text-dim);margin-bottom:1rem">${artist}</p>
          ${genresHTML ? `<div style="margin-bottom:0.75rem">${genresHTML}</div>` : ''}
          ${tagsHTML ? `<div style="margin-bottom:0.75rem">${tagsHTML}</div>` : ''}
          ${otherMeta}
          ${linksHTML}
        </div>
      </div>

      ${data.has_audio ? `
      <div class="panel audio-section">
        <div class="panel-header">Audio</div>
        <div class="panel-body">
          <audio controls preload="none" src="/api/audio?idx=${data.index}"></audio>
        </div>
      </div>` : ''}
    </div>
  `;

  let rightPanel = '';
  if (mert) {
    rightPanel = `
      <div class="right-col">
        <div class="panel">
          <div class="panel-header">MERT Output — Tensor Stats</div>
          <div class="panel-body">
            <div style="font-family:var(--mono);font-size:0.8rem;color:var(--text-dim);margin-bottom:1rem">
              shape: [${mert.shape.join(', ')}] &nbsp;—&nbsp; float32
            </div>
            <div class="stat-grid">
              <div class="stat-card"><div class="val">${mert.min.toFixed(3)}</div><div class="label">Min</div></div>
              <div class="stat-card"><div class="val">${mert.max.toFixed(3)}</div><div class="label">Max</div></div>
              <div class="stat-card"><div class="val">${mert.mean.toFixed(3)}</div><div class="label">Mean</div></div>
              <div class="stat-card"><div class="val">${mert.std.toFixed(3)}</div><div class="label">Std</div></div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">L2 Norm per Timestep</div>
          <div class="panel-body">
            <canvas id="norm-canvas"></canvas>
          </div>
        </div>
      </div>
    `;
  }

  main.innerHTML = leftPanel + rightPanel;
  main.style.animation = 'none';
  main.offsetHeight;
  main.style.animation = 'fadeIn 0.4s ease';

  if (mert) {
    setTimeout(() => {
      drawNormChart(mert.norm_per_timestep);
    }, 50);
  }
}

function drawNormChart(norms) {
  const canvas = document.getElementById('norm-canvas');
  if (!canvas) return;
  const W = canvas.clientWidth * 2;
  const H = canvas.clientHeight * 2;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  const max = Math.max(...norms);
  const min = Math.min(...norms);
  const range = max - min || 1;
  const pad = 20;

  ctx.strokeStyle = '#2a2a3a';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (H - 2 * pad) * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
  }

  const gradient = ctx.createLinearGradient(0, 0, W, 0);
  gradient.addColorStop(0, '#6c5ce7');
  gradient.addColorStop(0.5, '#00cec9');
  gradient.addColorStop(1, '#fd79a8');
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < norms.length; i++) {
    const x = pad + (W - 2 * pad) * (i / (norms.length - 1));
    const y = pad + (H - 2 * pad) * (1 - (norms[i] - min) / range);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const lastX = pad + (W - 2 * pad);
  ctx.lineTo(lastX, H - pad);
  ctx.lineTo(pad, H - pad);
  ctx.closePath();
  const fillGrad = ctx.createLinearGradient(0, 0, 0, H);
  fillGrad.addColorStop(0, 'rgba(108, 92, 231, 0.15)');
  fillGrad.addColorStop(1, 'rgba(108, 92, 231, 0.0)');
  ctx.fillStyle = fillGrad;
  ctx.fill();

  ctx.fillStyle = '#8888a0';
  ctx.font = `${Math.round(H * 0.04)}px JetBrains Mono`;
  ctx.textAlign = 'right';
  ctx.fillText(max.toFixed(1), W - pad, pad + 14);
  ctx.fillText(min.toFixed(1), W - pad, H - pad - 4);
  ctx.textAlign = 'left';
  ctx.fillText('0', pad, H - pad + 16);
  ctx.fillText(`${norms.length - 1}`, W - pad - 20, H - pad + 16);
}

loadRandom();

document.addEventListener('keydown', (e) => {
  if (e.key === 'r' && document.activeElement.tagName !== 'INPUT') loadRandom();
  if (e.key === 'Enter' && document.activeElement.id === 'row-input') loadRow();
});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize MERT dataset rows in a web browser")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the saved dataset on disk")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--window_size", type=int, default=1125)
    parser.add_argument("--feature_dim", type=int, default=768)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)
    print(f"  {len(dataset)} rows, columns: {dataset.column_names}")

    handler = make_handler(dataset, args.window_size, args.feature_dim)

    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"\n  Server running at http://localhost:{args.port}")
        print(f"  Press R to load a random row, or enter a row number")
        print(f"  Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()