"""
Neuron Explorer — visualize top-k activating songs per SAE neuron.

Usage:
    python neuron_explorer.py \
        --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
        --port 9999

Then SSH tunnel and open http://localhost:9999
"""

import argparse
import base64
import json
import os
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
from datasets import load_from_disk


def make_handler(test_set, activations_data, topk_cache):

    activations = activations_data["activations"]
    latent_size = activations_data["latent_size"]
    n_samples = activations_data["n_samples"]
    top_indices = topk_cache["top_indices"]
    top_values = topk_cache["top_values"]
    k = topk_cache["k"]

    # ── Precompute all stats once at startup ──────────────────────────────
    print("  Precomputing neuron stats...")
    mean_per_neuron = activations.mean(dim=0)
    std_per_neuron = activations.std(dim=0)
    sparsity_per_neuron = (activations > 0.1).float().mean(dim=0)
    max_per_neuron = activations.max(dim=0).values
    min_per_neuron = activations.min(dim=0).values

    # ── Pre-build overview JSON bytes ─────────────────────────────────────
    print("  Building overview cache...")
    sorted_indices = mean_per_neuron.argsort(descending=True).tolist()
    overview_neurons = []
    for i in sorted_indices[:200]:
        overview_neurons.append({
            "idx": i,
            "mean": round(float(mean_per_neuron[i]), 6),
            "std": round(float(std_per_neuron[i]), 6),
            "sparsity": round(float(sparsity_per_neuron[i]), 6),
            "max": round(float(max_per_neuron[i]), 6),
        })

    dead = int((mean_per_neuron < 0.01).sum().item())
    overview_bytes = json.dumps({
        "latent_size": latent_size,
        "n_samples": n_samples,
        "k": k,
        "dead_neurons": dead,
        "global_mean": round(float(activations.mean()), 6),
        "global_sparsity": round(float((activations > 0.1).float().mean()), 6),
        "neurons": overview_neurons,
    }).encode("utf-8")

    # ── Pre-build all neuron histograms ───────────────────────────────────
    print("  Precomputing histograms (this may take a minute)...")
    neuron_histograms = {}
    for i in range(latent_size):
        counts, edges = torch.histogram(activations[:, i].float(), bins=50)
        neuron_histograms[i] = {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
        }
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1}/{latent_size} histograms done")
    print(f"  All {latent_size} histograms cached.")

    class Handler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):
            try:
                parsed = urlparse(self.path)

                if parsed.path == "/" or parsed.path == "/index.html":
                    self._serve_html()
                elif parsed.path == "/api/overview":
                    self._serve_overview()
                elif parsed.path == "/api/neuron":
                    qs = parse_qs(parsed.query)
                    idx = int(qs.get("idx", [0])[0])
                    self._serve_neuron(idx)
                elif parsed.path == "/api/audio":
                    qs = parse_qs(parsed.query)
                    idx = int(qs.get("idx", [0])[0])
                    self._serve_audio(idx)
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                print(f"  Error: {e}")
                try:
                    self.send_error(500)
                except Exception:
                    pass

        def _serve_html(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        def _serve_overview(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(overview_bytes)

        def _serve_neuron(self, neuron_idx):
            neuron_idx = max(0, min(neuron_idx, latent_size - 1))

            tracks = []
            for rank in range(k):
                sample_idx = top_indices[neuron_idx][rank].item()
                activation_val = top_values[neuron_idx][rank].item()
                row = test_set[sample_idx]

                track = {
                    "rank": rank + 1,
                    "sample_idx": sample_idx,
                    "activation": activation_val,
                    "title": row.get("title", "Untitled"),
                    "artist": row.get("artist", "Unknown"),
                    "genres": row.get("genres", []),
                    "tags": row.get("tags", []),
                    "url": row.get("url", ""),
                    "album_title": row.get("album_title", ""),
                    "has_audio": row.get("audio_b64") is not None,
                }
                tracks.append(track)

            payload = {
                "neuron_idx": neuron_idx,
                "mean": float(mean_per_neuron[neuron_idx]),
                "std": float(std_per_neuron[neuron_idx]),
                "sparsity": float(sparsity_per_neuron[neuron_idx]),
                "max": float(max_per_neuron[neuron_idx]),
                "min": float(min_per_neuron[neuron_idx]),
                "tracks": tracks,
                "histogram": neuron_histograms[neuron_idx],
            }

            self._send_json(payload)

        def _serve_audio(self, sample_idx):
            sample_idx = max(0, min(sample_idx, len(test_set) - 1))
            row = test_set[sample_idx]
            audio_b64 = row.get("audio_b64")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(audio_bytes)))
                self.end_headers()
                self.wfile.write(audio_bytes)
            else:
                self.send_error(404, "No audio")

        def _send_json(self, payload):
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
<title>Neuron Explorer</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #060609;
    --surface: #0e0e14;
    --surface2: #151520;
    --surface3: #1c1c2b;
    --border: #252538;
    --border-light: #33334d;
    --text: #d8d8e8;
    --text-dim: #7070a0;
    --text-faint: #4a4a70;
    --accent: #ff6b35;
    --accent-dim: rgba(255, 107, 53, 0.15);
    --cyan: #22d3ee;
    --cyan-dim: rgba(34, 211, 238, 0.12);
    --green: #4ade80;
    --red: #f87171;
    --mono: 'Space Mono', monospace;
    --sans: 'Outfit', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
  }

  header {
    padding: 1.5rem 2.5rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
    background: rgba(6, 6, 9, 0.92);
    backdrop-filter: blur(16px);
  }

  .logo {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--accent);
    letter-spacing: 0.08em;
  }

  .logo span { color: var(--text-dim); font-weight: 400; }

  .nav-controls {
    display: flex; gap: 0.6rem; align-items: center;
  }

  .nav-controls input {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.8rem;
    padding: 0.45rem 0.65rem;
    border-radius: 5px;
    width: 90px;
    text-align: center;
  }

  .nav-controls input:focus { outline: none; border-color: var(--accent); }

  button {
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: all 0.15s ease;
    border: 1px solid transparent;
  }

  .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .btn-primary:hover { box-shadow: 0 0 20px rgba(255, 107, 53, 0.3); }
  .btn-ghost { background: transparent; color: var(--text-dim); border-color: var(--border); }
  .btn-ghost:hover { border-color: var(--text-dim); color: var(--text); }
  .btn-ghost.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

  #app { min-height: calc(100vh - 60px); }

  .view-overview {
    padding: 2rem 2.5rem;
    animation: fadeUp 0.35s ease;
  }

  .view-neuron {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    min-height: calc(100vh - 60px);
    animation: fadeUp 0.35s ease;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .overview-stats {
    display: flex; gap: 2rem; margin-bottom: 2rem;
    flex-wrap: wrap;
  }

  .overview-stat {
    display: flex; flex-direction: column;
  }

  .overview-stat .val {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--cyan);
    line-height: 1;
  }

  .overview-stat .label {
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .neuron-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.5rem;
  }

  .neuron-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.85rem 1rem;
    cursor: pointer;
    transition: all 0.15s ease;
    position: relative;
    overflow: hidden;
  }

  .neuron-card:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
  }

  .neuron-card .bar {
    position: absolute;
    bottom: 0; left: 0;
    height: 2px;
    background: var(--accent);
    transition: width 0.3s ease;
  }

  .neuron-card .n-id {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-faint);
  }

  .neuron-card .n-mean {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin: 0.2rem 0;
  }

  .neuron-card .n-sparsity {
    font-size: 0.7rem;
    color: var(--text-dim);
  }

  .neuron-card.dead {
    opacity: 0.4;
    border-style: dashed;
  }

  .section-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-faint);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }

  .neuron-left {
    border-right: 1px solid var(--border);
    padding: 2rem 2.5rem;
    overflow-y: auto;
    max-height: calc(100vh - 60px);
  }

  .neuron-title {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--text-faint);
    margin-bottom: 0.3rem;
  }

  .neuron-title strong {
    font-size: 2rem;
    color: var(--accent);
    display: block;
    line-height: 1;
    margin-bottom: 0.5rem;
  }

  .neuron-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1.5rem 0;
  }

  .n-stat {
    background: var(--surface2);
    border-radius: 6px;
    padding: 0.75rem;
  }

  .n-stat .val {
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 700;
    color: var(--cyan);
  }

  .n-stat .label {
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.15rem;
  }

  #hist-canvas {
    width: 100%;
    height: 120px;
    border-radius: 6px;
    margin-top: 1rem;
  }

  .nav-arrows {
    display: flex; gap: 0.5rem; margin-top: 1.5rem;
  }

  .neuron-right {
    padding: 2rem 2.5rem;
    overflow-y: auto;
    max-height: calc(100vh - 60px);
  }

  .track-list { display: flex; flex-direction: column; gap: 0.6rem; }

  .track-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    display: grid;
    grid-template-columns: 36px 1fr auto;
    gap: 1rem;
    align-items: center;
    transition: border-color 0.15s;
  }

  .track-card:hover { border-color: var(--border-light); }

  .track-rank {
    font-family: var(--mono);
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-faint);
    text-align: center;
  }

  .track-rank.top { color: var(--accent); }

  .track-info h3 {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 350px;
  }

  .track-info .artist {
    font-size: 0.8rem;
    color: var(--text-dim);
  }

  .track-info .genres {
    margin-top: 0.3rem;
  }

  .track-info .genre-tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.6rem;
    background: var(--accent-dim);
    color: var(--accent);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-right: 0.25rem;
  }

  .track-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.4rem;
  }

  .activation-badge {
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--cyan);
    background: var(--cyan-dim);
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
  }

  .track-audio audio {
    height: 28px;
    width: 200px;
  }

  .track-link {
    font-size: 0.7rem;
    color: var(--text-faint);
    text-decoration: none;
  }
  .track-link:hover { color: var(--accent); }

  .loading {
    display: flex; align-items: center; justify-content: center;
    padding: 4rem;
    font-family: var(--mono);
    color: var(--text-dim);
    font-size: 0.8rem;
  }

  @media (max-width: 900px) {
    .view-neuron { grid-template-columns: 1fr; }
    .neuron-left { border-right: none; border-bottom: 1px solid var(--border); max-height: none; }
    .neuron-right { max-height: none; }
    header { padding: 1rem; }
    .view-overview { padding: 1rem; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">NEURON<span> EXPLORER</span></div>
  <div class="nav-controls">
    <button class="btn-ghost" id="btn-overview" onclick="showOverview()">Overview</button>
    <input type="number" id="neuron-input" placeholder="Neuron #" min="0">
    <button class="btn-primary" onclick="goToNeuron()">Go</button>
  </div>
</header>

<div id="app">
  <div class="loading">Loading...</div>
</div>

<script>
let currentNeuron = null;
let overviewData = null;

async function showOverview() {
  document.getElementById('btn-overview').classList.add('active');
  if (!overviewData) {
    const res = await fetch('/api/overview');
    overviewData = await res.json();
  }
  renderOverview(overviewData);
}

async function showNeuron(idx) {
  document.getElementById('btn-overview').classList.remove('active');
  document.getElementById('neuron-input').value = idx;
  currentNeuron = idx;

  const app = document.getElementById('app');
  app.innerHTML = '<div class="loading">Loading neuron ' + idx + '...</div>';

  const res = await fetch(`/api/neuron?idx=${idx}`);
  const data = await res.json();
  renderNeuron(data);
}

function goToNeuron() {
  const val = document.getElementById('neuron-input').value;
  if (val !== '') showNeuron(parseInt(val));
}

function renderOverview(data) {
  const app = document.getElementById('app');
  const neurons = data.neurons;

  let cardsHTML = '';
  const maxMean = Math.max(...neurons.map(n => n.mean));

  for (const n of neurons) {
    const isDead = n.mean < 0.01;
    const barW = maxMean > 0 ? (n.mean / maxMean * 100) : 0;
    cardsHTML += `
      <div class="neuron-card ${isDead ? 'dead' : ''}" onclick="showNeuron(${n.idx})">
        <div class="n-id">#${n.idx}</div>
        <div class="n-mean">${n.mean.toFixed(3)}</div>
        <div class="n-sparsity">sparsity: ${(n.sparsity * 100).toFixed(0)}%</div>
        <div class="bar" style="width:${barW}%"></div>
      </div>`;
  }

  app.innerHTML = `
    <div class="view-overview">
      <div class="overview-stats">
        <div class="overview-stat">
          <div class="val">${data.latent_size.toLocaleString()}</div>
          <div class="label">Neurons</div>
        </div>
        <div class="overview-stat">
          <div class="val">${data.n_samples.toLocaleString()}</div>
          <div class="label">Test samples</div>
        </div>
        <div class="overview-stat">
          <div class="val">${data.dead_neurons.toLocaleString()}</div>
          <div class="label">Dead neurons</div>
        </div>
        <div class="overview-stat">
          <div class="val">${(data.global_sparsity * 100).toFixed(1)}%</div>
          <div class="label">Active (>0.1)</div>
        </div>
        <div class="overview-stat">
          <div class="val">${data.global_mean.toFixed(4)}</div>
          <div class="label">Global mean</div>
        </div>
      </div>
      <div class="section-label">Top ${neurons.length} most active neurons (click to explore)</div>
      <div class="neuron-grid">${cardsHTML}</div>
    </div>`;
}

function renderNeuron(data) {
  const app = document.getElementById('app');
  const n = data;

  let tracksHTML = '';
  for (const t of n.tracks) {
    const genreTags = (t.genres || []).map(g => `<span class="genre-tag">${g}</span>`).join('');
    tracksHTML += `
      <div class="track-card">
        <div class="track-rank ${t.rank <= 3 ? 'top' : ''}">${t.rank}</div>
        <div class="track-info">
          <h3>${t.title}</h3>
          <div class="artist">${t.artist}</div>
          ${genreTags ? `<div class="genres">${genreTags}</div>` : ''}
        </div>
        <div class="track-right">
          <div class="activation-badge">${t.activation.toFixed(4)}</div>
          ${t.has_audio ? `<div class="track-audio"><audio controls preload="none" src="/api/audio?idx=${t.sample_idx}"></audio></div>` : ''}
          ${t.url ? `<a class="track-link" href="${t.url}" target="_blank">source ↗</a>` : ''}
        </div>
      </div>`;
  }

  app.innerHTML = `
    <div class="view-neuron">
      <div class="neuron-left">
        <div class="neuron-title">NEURON<strong>#${n.neuron_idx}</strong></div>
        <div class="neuron-stats">
          <div class="n-stat"><div class="val">${n.mean.toFixed(4)}</div><div class="label">Mean</div></div>
          <div class="n-stat"><div class="val">${n.std.toFixed(4)}</div><div class="label">Std</div></div>
          <div class="n-stat"><div class="val">${n.max.toFixed(4)}</div><div class="label">Max</div></div>
          <div class="n-stat"><div class="val">${n.min.toFixed(4)}</div><div class="label">Min</div></div>
          <div class="n-stat"><div class="val">${(n.sparsity * 100).toFixed(1)}%</div><div class="label">Active >0.1</div></div>
          <div class="n-stat"><div class="val">${n.tracks.length}</div><div class="label">Top-k</div></div>
        </div>
        <div class="section-label">Activation Distribution</div>
        <canvas id="hist-canvas"></canvas>
        <div class="nav-arrows">
          <button class="btn-ghost" onclick="showNeuron(${Math.max(0, n.neuron_idx - 1)})">← prev</button>
          <button class="btn-ghost" onclick="showOverview()">overview</button>
          <button class="btn-ghost" onclick="showNeuron(${n.neuron_idx + 1})">next →</button>
        </div>
      </div>
      <div class="neuron-right">
        <div class="section-label">Top-${n.tracks.length} Activating Tracks</div>
        <div class="track-list">${tracksHTML}</div>
      </div>
    </div>`;

  setTimeout(() => drawHistogram(n.histogram), 50);
}

function drawHistogram(hist) {
  const canvas = document.getElementById('hist-canvas');
  if (!canvas) return;
  const W = canvas.clientWidth * 2;
  const H = canvas.clientHeight * 2;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  const counts = hist.counts;
  const edges = hist.edges;
  const maxCount = Math.max(...counts);
  const pad = 10;
  const barW = (W - 2 * pad) / counts.length;

  for (let i = 0; i < counts.length; i++) {
    const h = maxCount > 0 ? (counts[i] / maxCount) * (H - 2 * pad) : 0;
    const x = pad + i * barW;
    const y = H - pad - h;

    const t = i / counts.length;
    const r = Math.floor(34 + t * 221);
    const g = Math.floor(211 - t * 104);
    const b = Math.floor(238 - t * 185);
    ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
    ctx.fillRect(x, y, barW - 1, h);
  }

  ctx.fillStyle = '#4a4a70';
  ctx.font = `${Math.round(H * 0.07)}px Space Mono`;
  ctx.textAlign = 'left';
  ctx.fillText(edges[0].toFixed(2), pad, H - pad + 14);
  ctx.textAlign = 'right';
  ctx.fillText(edges[edges.length - 1].toFixed(2), W - pad, H - pad + 14);
}

document.addEventListener('keydown', (e) => {
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft' && currentNeuron !== null) showNeuron(Math.max(0, currentNeuron - 1));
  if (e.key === 'ArrowRight' && currentNeuron !== null) showNeuron(currentNeuron + 1);
  if (e.key === 'o') showOverview();
  if (e.key === 'Enter') goToNeuron();
});

document.getElementById('neuron-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') goToNeuron();
});

showOverview();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Explore SAE neurons and their top-k activating tracks")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset with audio")
    parser.add_argument("--latent_dir", type=str, required=True,
                        help="Path to latent_analysis directory")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", type=bool, default=False)
    args = parser.parse_args()

    # Load dataset with memory mapping (not into RAM)
    print(f"Loading dataset from {args.dataset}...")
    full_dataset = load_from_disk(args.dataset, keep_in_memory=False)
    if args.full:
        test_set = full_dataset
    else:
        splits = full_dataset.train_test_split(test_size=0.2, seed=args.seed)
        test_val = splits["test"].train_test_split(test_size=0.5, seed=args.seed)
        test_set = test_val["test"]
    print(f"  Dataset: {len(test_set)} rows")

    # Load latent data
    print(f"Loading latent data from {args.latent_dir}...")
    activations_data = torch.load(
        os.path.join(args.latent_dir, "latent_activations.pt"),
        map_location="cpu",
    )
    topk_cache = torch.load(
        os.path.join(args.latent_dir, "top_k_cache.pt"),
        map_location="cpu",
    )
    print(f"  Activations: {activations_data['activations'].shape}")
    print(f"  Top-k: {topk_cache['top_indices'].shape}")

    handler = make_handler(test_set, activations_data, topk_cache)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"\n  Server running at http://localhost:{args.port}")
        print(f"  Shortcuts: arrow keys = prev/next neuron, O = overview")
        print(f"  Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()