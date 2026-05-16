"""
Export a single neuron's top-k page as a self-contained HTML file.

Usage:
    python export_neuron.py \
        --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
        --neuron 9704 \
        --output neuron_9704.html
"""

import argparse
import base64
import json
import os
import torch
import numpy as np
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--neuron", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"neuron_{args.neuron}.html"

    # Load data
    print("Loading dataset...")
    full_dataset = load_from_disk(args.dataset)
    splits = full_dataset.train_test_split(test_size=0.2, seed=args.seed)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=args.seed)
    test_set = test_val["test"]

    print("Loading latent data...")
    activations_data = torch.load(
        os.path.join(args.latent_dir, "latent_activations.pt"),
        map_location="cpu",
    )
    topk_cache = torch.load(
        os.path.join(args.latent_dir, "top_k_cache.pt"),
        map_location="cpu",
    )

    activations = activations_data["activations"]
    latent_size = activations_data["latent_size"]
    top_indices = topk_cache["top_indices"]
    top_values = topk_cache["top_values"]
    k = topk_cache["k"]
    neuron = args.neuron

    # Neuron stats
    neuron_acts = activations[:, neuron]
    hist_counts, hist_edges = torch.histogram(neuron_acts, bins=50)

    stats = {
        "mean": float(neuron_acts.mean()),
        "std": float(neuron_acts.std()),
        "max": float(neuron_acts.max()),
        "min": float(neuron_acts.min()),
        "sparsity": float((neuron_acts > 0.1).float().mean()),
    }

    # Collect tracks with embedded audio
    print(f"Collecting top-{k} tracks for neuron {neuron}...")
    tracks = []
    for rank in range(k):
        sample_idx = top_indices[neuron][rank].item()
        activation_val = top_values[neuron][rank].item()
        row = test_set[sample_idx]

        audio_b64 = row.get("audio_b64", None)

        genres = row.get("genres", [])
        if isinstance(genres, np.ndarray):
            genres = genres.tolist()

        tags = row.get("tags", [])
        if isinstance(tags, np.ndarray):
            tags = tags.tolist()

        track = {
            "rank": rank + 1,
            "activation": round(activation_val, 6),
            "title": row.get("title", "Untitled"),
            "artist": row.get("artist", "Unknown"),
            "genres": genres,
            "tags": tags,
            "url": row.get("url", ""),
            "album_title": row.get("album_title", ""),
            "audio_b64": audio_b64,
        }
        tracks.append(track)
        print(f"  #{rank+1}: {track['title']} by {track['artist']}")

    # Build HTML
    print("Building HTML...")

    tracks_html = ""
    for t in tracks:
        genre_tags = "".join(
            f'<span class="genre-tag">{g}</span>' for g in (t["genres"] or [])
        )
        audio_el = ""
        if t["audio_b64"]:
            audio_el = f'<audio controls preload="none" src="data:audio/mpeg;base64,{t["audio_b64"]}"></audio>'

        link_el = ""
        if t["url"]:
            link_el = f'<a class="track-link" href="{t["url"]}" target="_blank">source ↗</a>'

        tracks_html += f"""
        <div class="track-card">
          <div class="track-rank {"top" if t["rank"] <= 3 else ""}">{t["rank"]}</div>
          <div class="track-info">
            <h3>{t["title"]}</h3>
            <div class="artist">{t["artist"]}</div>
            {f'<div class="genres">{genre_tags}</div>' if genre_tags else ''}
          </div>
          <div class="track-right">
            <div class="activation-badge">{t["activation"]:.4f}</div>
            {f'<div class="track-audio">{audio_el}</div>' if audio_el else ''}
            {link_el}
          </div>
        </div>"""

    hist_data_js = json.dumps({
        "counts": hist_counts.tolist(),
        "edges": hist_edges.tolist(),
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neuron #{neuron} — SAE Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #060609;
    --surface: #0e0e14;
    --surface2: #151520;
    --border: #252538;
    --border-light: #33334d;
    --text: #d8d8e8;
    --text-dim: #7070a0;
    --text-faint: #4a4a70;
    --accent: #ff6b35;
    --accent-dim: rgba(255, 107, 53, 0.15);
    --cyan: #22d3ee;
    --cyan-dim: rgba(34, 211, 238, 0.12);
    --mono: 'Space Mono', monospace;
    --sans: 'Outfit', sans-serif;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
  }}

  header {{
    padding: 1.5rem 2.5rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(6, 6, 9, 0.92);
  }}

  .logo {{
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--accent);
    letter-spacing: 0.08em;
  }}

  .logo span {{ color: var(--text-dim); font-weight: 400; }}

  .main {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    min-height: calc(100vh - 60px);
  }}

  .neuron-left {{
    border-right: 1px solid var(--border);
    padding: 2rem 2.5rem;
    overflow-y: auto;
  }}

  .neuron-title {{
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--text-faint);
    margin-bottom: 0.3rem;
  }}

  .neuron-title strong {{
    font-size: 2rem;
    color: var(--accent);
    display: block;
    line-height: 1;
    margin-bottom: 0.5rem;
  }}

  .neuron-stats {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1.5rem 0;
  }}

  .n-stat {{
    background: var(--surface2);
    border-radius: 6px;
    padding: 0.75rem;
  }}

  .n-stat .val {{
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 700;
    color: var(--cyan);
  }}

  .n-stat .label {{
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.15rem;
  }}

  .section-label {{
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-faint);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }}

  #hist-canvas {{
    width: 100%;
    height: 120px;
    border-radius: 6px;
    margin-top: 1rem;
  }}

  .neuron-right {{
    padding: 2rem 2.5rem;
    overflow-y: auto;
  }}

  .track-list {{ display: flex; flex-direction: column; gap: 0.6rem; }}

  .track-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    display: grid;
    grid-template-columns: 36px 1fr auto;
    gap: 1rem;
    align-items: center;
  }}

  .track-card:hover {{ border-color: var(--border-light); }}

  .track-rank {{
    font-family: var(--mono);
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-faint);
    text-align: center;
  }}

  .track-rank.top {{ color: var(--accent); }}

  .track-info h3 {{
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 350px;
  }}

  .track-info .artist {{
    font-size: 0.8rem;
    color: var(--text-dim);
  }}

  .track-info .genres {{ margin-top: 0.3rem; }}

  .genre-tag {{
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.6rem;
    background: var(--accent-dim);
    color: var(--accent);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-right: 0.25rem;
  }}

  .track-right {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.4rem;
  }}

  .activation-badge {{
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--cyan);
    background: var(--cyan-dim);
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
  }}

  .track-audio audio {{
    height: 28px;
    width: 200px;
  }}

  .track-link {{
    font-size: 0.7rem;
    color: var(--text-faint);
    text-decoration: none;
  }}
  .track-link:hover {{ color: var(--accent); }}

  @media (max-width: 900px) {{
    .main {{ grid-template-columns: 1fr; }}
    .neuron-left {{ border-right: none; border-bottom: 1px solid var(--border); }}
  }}
</style>
</head>
<body>

<header>
  <div class="logo">NEURON<span> EXPLORER</span></div>
  <div style="font-family:var(--mono);font-size:0.8rem;color:var(--text-dim)">
    Latent size: {latent_size} &nbsp;·&nbsp; Test samples: {activations.shape[0]}
  </div>
</header>

<div class="main">
  <div class="neuron-left">
    <div class="neuron-title">NEURON<strong>#{neuron}</strong></div>
    <div class="neuron-stats">
      <div class="n-stat"><div class="val">{stats["mean"]:.4f}</div><div class="label">Mean</div></div>
      <div class="n-stat"><div class="val">{stats["std"]:.4f}</div><div class="label">Std</div></div>
      <div class="n-stat"><div class="val">{stats["max"]:.4f}</div><div class="label">Max</div></div>
      <div class="n-stat"><div class="val">{stats["min"]:.4f}</div><div class="label">Min</div></div>
      <div class="n-stat"><div class="val">{stats["sparsity"]*100:.1f}%</div><div class="label">Active >0.1</div></div>
      <div class="n-stat"><div class="val">{k}</div><div class="label">Top-k</div></div>
    </div>
    <div class="section-label">Activation Distribution</div>
    <canvas id="hist-canvas"></canvas>
  </div>
  <div class="neuron-right">
    <div class="section-label">Top-{k} Activating Tracks</div>
    <div class="track-list">
      {tracks_html}
    </div>
  </div>
</div>

<script>
const histData = {hist_data_js};

function drawHistogram() {{
  const canvas = document.getElementById('hist-canvas');
  if (!canvas) return;
  const W = canvas.clientWidth * 2;
  const H = canvas.clientHeight * 2;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  const counts = histData.counts;
  const edges = histData.edges;
  const maxCount = Math.max(...counts);
  const pad = 10;
  const barW = (W - 2 * pad) / counts.length;

  for (let i = 0; i < counts.length; i++) {{
    const h = maxCount > 0 ? (counts[i] / maxCount) * (H - 2 * pad) : 0;
    const x = pad + i * barW;
    const y = H - pad - h;
    const t = i / counts.length;
    const r = Math.floor(34 + t * 221);
    const g = Math.floor(211 - t * 104);
    const b = Math.floor(238 - t * 185);
    ctx.fillStyle = `rgba(${{r}},${{g}},${{b}},0.7)`;
    ctx.fillRect(x, y, barW - 1, h);
  }}

  ctx.fillStyle = '#4a4a70';
  ctx.font = `${{Math.round(H * 0.07)}}px Space Mono`;
  ctx.textAlign = 'left';
  ctx.fillText(edges[0].toFixed(2), pad, H - pad + 14);
  ctx.textAlign = 'right';
  ctx.fillText(edges[edges.length - 1].toFixed(2), W - pad, H - pad + 14);
}}

drawHistogram();
</script>
</body>
</html>"""

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({size_mb:.1f} MB)")
    print("Open this file directly in any browser — no server needed.")


if __name__ == "__main__":
    main()