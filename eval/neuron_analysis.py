"""
Generate a paper-ready neuron analysis figure with:
  (a) Activation histogram showing neuron selectivity
  (b) Genre breakdown: top-K vs overall dataset distribution
  (c) Mel-spectrograms of top-10 tracks in a 2x5 grid
  (d) Pairwise spectrogram cosine similarity matrix

Usage:
    python neuron_analysis.py \
        --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
        --neuron 5773 \
        --output neuron_5773_analysis.pdf

    # For FMA large supplement:
    python neuron_analysis.py \
        --dataset orcd/scratch/window1125_layer-1/fma_large_supplement/dataset_with_audio \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis_flat_1e-1_rho08_32 \
        --neuron 5773 \
        --output neuron_5773_analysis.pdf \
        --full
"""

import argparse
import base64
import collections
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import librosa
import librosa.display
from datasets import load_from_disk

GENRE_NAMES = {
    0: "20th Century Classical", 1: "Abstract Hip-Hop", 2: "African", 3: "Afrobeat",
    4: "Alternative Hip-Hop", 5: "Ambient", 6: "Ambient Electronic", 7: "Americana",
    8: "Asia-Far East", 9: "Audio Collage", 10: "Avant-Garde", 11: "Balkan",
    12: "Banter", 13: "Be-Bop", 14: "Big Band/Swing", 15: "Bigbeat",
    16: "Black-Metal", 17: "Bluegrass", 18: "Blues", 19: "Bollywood",
    20: "Brazilian", 21: "Breakbeat", 22: "Breakcore - Hard", 23: "British Folk",
    24: "Celtic", 25: "Chamber Music", 26: "Chill-out", 27: "Chip Music",
    28: "Chiptune", 29: "Choral Music", 30: "Christmas", 31: "Classical",
    32: "Comedy", 33: "Compilation", 34: "Composed Music", 35: "Contemporary Classical",
    36: "Country", 37: "Country & Western", 38: "Cumbia", 39: "Dance",
    40: "Death-Metal", 41: "Deep Funk", 42: "Disco", 43: "Downtempo",
    44: "Drone", 45: "Drum & Bass", 46: "Dubstep", 47: "Easy Listening",
    48: "Easy Listening: Vocal", 49: "Electro-Punk", 50: "Electroacoustic",
    51: "Electronic", 52: "Experimental", 53: "Experimental Pop",
    54: "Fado", 55: "Field Recordings", 56: "Flamenco", 57: "Folk",
    58: "Freak-Folk", 59: "Free-Folk", 60: "Free-Jazz", 61: "French",
    62: "Funk", 63: "Garage", 64: "Glitch", 65: "Gospel", 66: "Goth",
    67: "Grindcore", 68: "Hardcore", 69: "Hip-Hop", 70: "Hip-Hop Beats",
    71: "Holiday", 72: "House", 73: "IDM", 74: "Improv", 75: "Indian",
    76: "Indie-Rock", 77: "Industrial", 78: "Instrumental", 79: "International",
    80: "Interview", 81: "Jazz", 82: "Jazz: Out", 83: "Jazz: Vocal",
    84: "Jungle", 85: "Kid-Friendly", 86: "Klezmer", 87: "Krautrock",
    88: "Latin", 89: "Latin America", 90: "Lo-Fi", 91: "Loud-Rock",
    92: "Lounge", 93: "Metal", 94: "Middle East", 95: "Minimal Electronic",
    96: "Minimalism", 97: "Modern Jazz", 98: "Musical Theater",
    99: "Musique Concrete", 100: "N. Indian Traditional", 101: "Nerdcore",
    102: "New Age", 103: "New Wave", 104: "No Wave", 105: "Noise",
    106: "Noise-Rock", 107: "North African", 108: "Novelty", 109: "Nu-Jazz",
    110: "Old-Time / Historic", 111: "Opera", 112: "Pacific", 113: "Poetry",
    114: "Polka", 115: "Pop", 116: "Post-Punk", 117: "Post-Rock",
    118: "Power-Pop", 119: "Progressive", 120: "Psych-Folk", 121: "Psych-Rock",
    122: "Punk", 123: "Radio", 124: "Radio Art", 125: "Radio Theater",
    126: "Rap", 127: "Reggae - Dancehall", 128: "Reggae - Dub", 129: "Rock",
    130: "Rock Opera", 131: "Rockabilly", 132: "Romany (Gypsy)", 133: "Salsa",
    134: "Shoegaze", 135: "Singer-Songwriter", 136: "Skweee", 137: "Sludge",
    138: "Soul-RnB", 139: "Sound Art", 140: "Sound Collage", 141: "Sound Effects",
    142: "Sound Poetry", 143: "Soundtrack", 144: "South Indian Traditional",
    145: "Space-Rock", 146: "Spanish", 147: "Spoken", 148: "Spoken Weird",
    149: "Spoken Word", 150: "Surf", 151: "Symphony", 152: "Synth Pop",
    153: "Talk Radio", 154: "Tango", 155: "Techno", 156: "Thrash",
    157: "Trip-Hop", 158: "Turkish", 159: "Unclassifiable", 160: "Western Swing",
    161: "Wonky", 162: "hiphop",
}


def decode_mp3_to_audio(audio_b64, sr=22050):
    audio_bytes = base64.b64decode(audio_b64)
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    return audio


def compute_mel_spectrogram(audio, sr=22050, n_mels=128, hop_length=512):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def spectrogram_cosine_similarity(specs):
    min_len = min(s.shape[1] for s in specs)
    vecs = []
    for s in specs:
        flat = s[:, :min_len].flatten()
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        vecs.append(flat)
    vecs = np.stack(vecs)
    return vecs @ vecs.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--neuron", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"neuron_{args.neuron}_analysis.pdf"

    # ── CVPR style ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5,
        "figure.dpi": args.dpi,
        "savefig.dpi": args.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    })

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset, keep_in_memory=False)
    if args.full:
        test_set = dataset
    else:
        splits = dataset.train_test_split(test_size=0.2, seed=args.seed)
        test_val = splits["test"].train_test_split(test_size=0.5, seed=args.seed)
        test_set = test_val["test"]

    print("Loading latent data...")
    activations_data = torch.load(
        os.path.join(args.latent_dir, "latent_activations.pt"), map_location="cpu")
    topk_cache = torch.load(
        os.path.join(args.latent_dir, "top_k_cache.pt"), map_location="cpu")

    activations = activations_data["activations"]
    top_indices = topk_cache["top_indices"]
    top_values = topk_cache["top_values"]
    k = topk_cache["k"]
    neuron = args.neuron
    neuron_acts = activations[:, neuron]

    # ── Collect top-K tracks ──────────────────────────────────────────────
    print(f"Collecting top-{k} tracks for neuron #{neuron}...")
    tracks = []
    topk_genre_counts = collections.Counter()

    for rank in range(k):
        sample_idx = top_indices[neuron][rank].item()
        activation_val = top_values[neuron][rank].item()
        row = test_set[sample_idx]
        genres = row.get("genres", [])
        for g in genres:
            topk_genre_counts[g] += 1
        tracks.append({
            "rank": rank + 1,
            "activation": activation_val,
            "title": row.get("title", "?"),
            "artist": row.get("artist", "?"),
            "genres": genres,
            "audio_b64": row.get("audio_b64"),
        })

    # Overall genre counts
    print("Counting overall genre distribution...")
    overall_genre_counts = collections.Counter()
    for i in range(len(test_set)):
        for g in test_set[i]["genres"]:
            overall_genre_counts[g] += 1

    display_genres = [g for g, _ in topk_genre_counts.most_common()][:15]

    # ── Compute spectrograms (top 10) ─────────────────────────────────────
    n_specs = min(10, k)
    print(f"Computing spectrograms for top {n_specs} tracks...")
    spectrograms = []
    spec_labels = []
    spec_activations = []

    for i in range(n_specs):
        t = tracks[i]
        if t["audio_b64"] is None:
            print(f"  #{t['rank']}: No audio, skipping")
            continue
        try:
            audio = decode_mp3_to_audio(t["audio_b64"])
            S_db = compute_mel_spectrogram(audio)
            spectrograms.append(S_db)
            title_short = t["title"][:18] + ".." if len(t["title"]) > 18 else t["title"]
            spec_labels.append(f"#{t['rank']} {title_short}")
            spec_activations.append(t["activation"])
            print(f"  #{t['rank']}: {t['title']}")
        except Exception as e:
            print(f"  #{t['rank']}: Failed ({e})")

    # Pad to 10 if needed
    while len(spectrograms) < 10:
        spectrograms.append(None)
        spec_labels.append("")
        spec_activations.append(0)

    # ── Similarity matrix ─────────────────────────────────────────────────
    valid_specs = [s for s in spectrograms if s is not None]
    if len(valid_specs) >= 2:
        sim_matrix = spectrogram_cosine_similarity(valid_specs)
        n_v = len(valid_specs)
        mask = ~np.eye(n_v, dtype=bool)
        avg_sim = sim_matrix[mask].mean()
        min_sim = sim_matrix[mask].min()
        max_sim = sim_matrix[mask].max()
    else:
        sim_matrix = None
        avg_sim = min_sim = max_sim = 0.0

    # ── Terminal output ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Neuron #{neuron} — Spectrogram Similarity")
    print(f"{'='*60}")
    print(f"  Tracks compared: {len(valid_specs)}")
    print(f"  Avg pairwise cosine similarity: {avg_sim:.4f}")
    print(f"  Min: {min_sim:.4f}, Max: {max_sim:.4f}")
    if sim_matrix is not None:
        print(f"\n  Pairwise similarity matrix:")
        header = "        " + "  ".join([f"{i+1:>5}" for i in range(len(valid_specs))])
        print(header)
        for i in range(len(valid_specs)):
            row_str = f"  #{i+1:>3}  " + "  ".join(
                [f"{sim_matrix[i,j]:>5.3f}" for j in range(len(valid_specs))])
            print(row_str)

    named_counts = [(GENRE_NAMES.get(g, str(g)), c) for g, c in topk_genre_counts.most_common()]
    genre_str = ", ".join(f"{name}: {count}" for name, count in named_counts)
    print(f"\n  Genre counts: {genre_str}")

    # ── Build figure ──────────────────────────────────────────────────────
    print("\nGenerating figure...")

    has_sim = sim_matrix is not None

    # Layout: 4 rows
    #   Row 0: (a) histogram  |  (b) genre bars         height ~2.0
    #   Row 1: spectrogram row 1 (5 squares)            height ~1.4
    #   Row 2: spectrogram row 2 (5 squares)            height ~1.4
    #   Row 3: (d) similarity matrix                    height ~2.0 (if present)

    n_rows = 3 + (1 if has_sim else 0)
    height_ratios = [2.0, 1.4, 1.4] + ([2.0] if has_sim else [])
    fig_height = sum(height_ratios) + 0.8

    fig = plt.figure(figsize=(7.0, fig_height))
    gs_main = gridspec.GridSpec(n_rows, 1, figure=fig,
                                height_ratios=height_ratios,
                                hspace=0.5)

    # ── Row 0: (a) histogram + (b) genre bars ────────────────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0],
                                              wspace=0.35)

    # (a) Activation histogram
    ax_hist = fig.add_subplot(gs_top[0])
    acts_np = neuron_acts.numpy()
    counts_h, edges_h, patches_h = ax_hist.hist(
        acts_np, bins=50, color="#2171b5", edgecolor="white", linewidth=0.3, zorder=3)

    topk_threshold = top_values[neuron][-1].item()
    for patch, left_edge in zip(patches_h, edges_h[:-1]):
        if left_edge >= topk_threshold:
            patch.set_facecolor("#e34a33")

    ax_hist.axvline(x=topk_threshold, color="#e34a33", linewidth=0.7,
                    linestyle="--", alpha=0.8, zorder=4)
    ax_hist.text(topk_threshold, ax_hist.get_ylim()[1] * 0.92, f" top-{k}",
                 fontsize=5, color="#e34a33", va="top")

    ax_hist.set_xlabel("Activation value")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"(a) Neuron #{neuron} activation distribution",
                      fontweight="bold", loc="left")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.yaxis.grid(True, linewidth=0.2, alpha=0.4, zorder=0)
    ax_hist.set_axisbelow(True)

    stats_text = f"$\\mu$={acts_np.mean():.3f}\n$\\sigma$={acts_np.std():.3f}\nmax={acts_np.max():.3f}"
    ax_hist.text(0.97, 0.95, stats_text, transform=ax_hist.transAxes,
                 fontsize=5, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="0.8", alpha=0.9))

    # (b) Genre comparison
    ax_genre = fig.add_subplot(gs_top[1])
    genre_labels = [GENRE_NAMES.get(g, str(g)) for g in display_genres]
    x = np.arange(len(display_genres))
    bar_w = 0.35

    topk_total = sum(topk_genre_counts.values())
    overall_total = sum(overall_genre_counts.values())
    topk_pcts = [100 * topk_genre_counts.get(g, 0) / topk_total for g in display_genres]
    overall_pcts = [100 * overall_genre_counts.get(g, 0) / overall_total for g in display_genres]

    ax_genre.bar(x - bar_w / 2, topk_pcts, bar_w, color="#e34a33",
                 edgecolor="white", linewidth=0.3, label=f"Top-{k}", zorder=3)
    ax_genre.bar(x + bar_w / 2, overall_pcts, bar_w, color="#bdd7e7",
                 edgecolor="white", linewidth=0.3, label="Dataset", zorder=3)

    ax_genre.set_xticks(x)
    ax_genre.set_xticklabels(genre_labels, rotation=45, ha="right",
                              rotation_mode="anchor", fontsize=5)
    ax_genre.set_ylabel("Percentage (%)")
    ax_genre.set_title(f"(b) Genre: top-{k} vs. dataset",
                      fontweight="bold", loc="left")
    ax_genre.legend(frameon=True, fancybox=False, edgecolor="0.8", fontsize=5)
    ax_genre.spines["top"].set_visible(False)
    ax_genre.spines["right"].set_visible(False)
    ax_genre.yaxis.grid(True, linewidth=0.2, alpha=0.4, zorder=0)
    ax_genre.set_axisbelow(True)

    # ── Rows 1–2: (c) Spectrograms in 2×5 grid ──────────────────────────
    for row_idx in range(2):
        gs_specs = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs_main[1 + row_idx], wspace=0.08)

        for col_idx in range(5):
            spec_idx = row_idx * 5 + col_idx
            ax = fig.add_subplot(gs_specs[col_idx])

            if spec_idx < len(spectrograms) and spectrograms[spec_idx] is not None:
                S_db = spectrograms[spec_idx]
                # Crop to make roughly square: take first N time frames
                # n_mels=128, so take ~128 time frames
                n_frames = min(S_db.shape[1], 128)
                S_crop = S_db[:, :n_frames]

                ax.imshow(S_crop, aspect="auto", origin="lower", cmap="magma",
                          interpolation="nearest")

                # Title: rank and short name
                label = spec_labels[spec_idx]
                ax.set_title(label, fontsize=4.5, pad=2)

                # Activation badge below
                act_val = spec_activations[spec_idx]
                ax.set_xlabel(f"{act_val:.3f}", fontsize=4.5, labelpad=1)
            else:
                ax.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)

        # Row label
        if row_idx == 0:
            fig.text(0.01, 0.5 * (gs_main[1].get_position(fig).y0 +
                                    gs_main[1].get_position(fig).y1),
                     "(c)", fontsize=7, fontweight="bold", va="center",
                     transform=fig.transFigure)

    # ── Row 3: (d) Similarity matrix ─────────────────────────────────────
    if has_sim:
        ax_sim = fig.add_subplot(gs_main[3])
        n_v = len(valid_specs)
        im = ax_sim.imshow(sim_matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")

        for i in range(n_v):
            for j in range(n_v):
                color = "white" if sim_matrix[i, j] > 0.75 or sim_matrix[i, j] < 0.25 else "black"
                weight = "bold" if i == j else "normal"
                ax_sim.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                           fontsize=4.5, color=color, fontweight=weight)

        tick_labels = [f"#{i+1}" for i in range(n_v)]
        ax_sim.set_xticks(range(n_v))
        ax_sim.set_yticks(range(n_v))
        ax_sim.set_xticklabels(tick_labels, fontsize=5)
        ax_sim.set_yticklabels(tick_labels, fontsize=5)
        ax_sim.set_title(
            f"(d) Spectrogram cosine similarity "
            f"(avg={avg_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f})",
            fontweight="bold", loc="left", fontsize=6.5)

        cbar = fig.colorbar(im, ax=ax_sim, fraction=0.015, pad=0.01, shrink=0.9)
        cbar.ax.tick_params(labelsize=5)

    # ── Save ──────────────────────────────────────────────────────────────
    plt.savefig(args.output)
    print(f"\nSaved to {args.output}")

    png_path = args.output.rsplit(".", 1)[0] + ".png"
    plt.savefig(png_path)
    print(f"Saved to {png_path}")
    plt.close()


if __name__ == "__main__":
    main()