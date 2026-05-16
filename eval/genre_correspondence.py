"""
Genre Correspondence Analysis — generates:
  1. A confusion-matrix-style heatmap: x = top neurons, y = genres, cell = avg activation
  2. Genre purity statistics (per-neuron and overall)
  3. Paper-ready figure and terminal summary

Usage:
    python genre_correspondence.py \
        --dataset orcd/scratch/window1125_layer-1/dataset \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
        --output genre_correspondence.pdf

    # For FMA large supplement:
    python genre_correspondence.py \
        --dataset orcd/scratch/window1125_layer-1/fma_large_supplement/dataset \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis_flat_1e-1_rho08_32 \
        --output genre_correspondence.pdf \
        --full
"""

import argparse
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="genre_correspondence.pdf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--top_neurons", type=int, default=50,
                        help="Number of most-active neurons to show on x-axis")
    parser.add_argument("--top_genres", type=int, default=20,
                        help="Number of most-common genres to show on y-axis")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # ── CVPR style ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": args.dpi,
        "savefig.dpi": args.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.4,
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
    print(f"  {len(test_set)} samples")

    print("Loading latent data...")
    activations_data = torch.load(
        os.path.join(args.latent_dir, "latent_activations.pt"), map_location="cpu")
    topk_cache = torch.load(
        os.path.join(args.latent_dir, "top_k_cache.pt"), map_location="cpu")

    activations = activations_data["activations"]
    latent_size = activations_data["latent_size"]
    top_indices = topk_cache["top_indices"]
    top_values = topk_cache["top_values"]
    k = topk_cache["k"]

    # ── Build sample → genres mapping ─────────────────────────────────────
    print("Building genre index...")
    sample_genres = []
    overall_genre_counts = collections.Counter()
    for i in range(len(test_set)):
        genres = test_set[i]["genres"]
        sample_genres.append(genres)
        for g in genres:
            overall_genre_counts[g] += 1

    # Top genres by overall frequency
    top_genre_ids = [g for g, _ in overall_genre_counts.most_common(args.top_genres)]
    genre_labels = [GENRE_NAMES.get(g, str(g)) for g in top_genre_ids]
    n_genres = len(top_genre_ids)

    # Dataset baseline: probability of each genre
    total_genre_tags = sum(overall_genre_counts.values())
    baseline_pcts = {g: 100.0 * overall_genre_counts[g] / total_genre_tags for g in top_genre_ids}

    # ── Select top neurons by mean activation ─────────────────────────────
    mean_per_neuron = activations.mean(dim=0)
    top_neuron_indices = mean_per_neuron.argsort(descending=True)[:args.top_neurons].tolist()

    # ── Build heatmap matrix + compute purity ─────────────────────────────
    print(f"Analyzing {args.top_neurons} neurons × {n_genres} genres...")

    # heatmap[genre_row, neuron_col] = fraction of that neuron's top-K belonging to that genre
    heatmap = np.zeros((n_genres, args.top_neurons))
    purities = []  # per-neuron purity (max genre fraction)

    for col, neuron_idx in enumerate(top_neuron_indices):
        genre_counts = collections.Counter()
        total_tags = 0

        for rank in range(k):
            sample_idx = top_indices[neuron_idx][rank].item()
            for g in sample_genres[sample_idx]:
                genre_counts[g] += 1
                total_tags += 1

        if total_tags == 0:
            purities.append(0.0)
            continue

        for row, genre_id in enumerate(top_genre_ids):
            heatmap[row, col] = genre_counts.get(genre_id, 0) / total_tags

        # Purity = fraction of top-K tags belonging to the most common genre
        most_common_count = genre_counts.most_common(1)[0][1]
        purities.append(most_common_count / total_tags)

    avg_purity = 100 * np.mean(purities)

    # Dataset baseline purity: if you randomly sample K tracks, what's the
    # expected purity? It's dominated by the most common genre.
    baseline_purity = max(baseline_pcts.values())

    # ── Also compute purity for ALL neurons (not just top N) ──────────────
    print("Computing purity across all active neurons...")
    all_purities = []
    n_active = 0

    for neuron_idx in range(latent_size):
        # Skip dead neurons
        if mean_per_neuron[neuron_idx] < 0.01:
            continue
        n_active += 1

        genre_counts = collections.Counter()
        total_tags = 0
        for rank in range(k):
            sample_idx = top_indices[neuron_idx][rank].item()
            for g in sample_genres[sample_idx]:
                genre_counts[g] += 1
                total_tags += 1

        if total_tags > 0:
            most_common_count = genre_counts.most_common(1)[0][1]
            all_purities.append(most_common_count / total_tags)

        if (neuron_idx + 1) % 5000 == 0:
            print(f"  {neuron_idx + 1}/{latent_size}")

    overall_avg_purity = 100 * np.mean(all_purities)

    # ── Terminal output ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Genre Correspondence Summary")
    print(f"{'='*60}")
    print(f"  Total neurons: {latent_size}")
    print(f"  Active neurons (mean > 0.01): {n_active}")
    print(f"  Top-K: {k}")
    print(f"  Dataset baseline purity: {baseline_purity:.1f}%")
    print(f"  Avg purity (top {args.top_neurons} neurons): {avg_purity:.1f}%")
    print(f"  Avg purity (all {n_active} active neurons): {overall_avg_purity:.1f}%")

    print(f"\n  Interpretation:")
    if overall_avg_purity < baseline_purity * 1.5:
        print(f"  → Neurons are NOT dominated by single genres (purity {overall_avg_purity:.1f}%")
        print(f"    vs baseline {baseline_purity:.1f}%), indicating cross-genre features.")
    else:
        print(f"  → Some neurons show genre specificity (purity {overall_avg_purity:.1f}%")
        print(f"    vs baseline {baseline_purity:.1f}%).")

    # Purity distribution
    purity_arr = np.array(all_purities) * 100
    print(f"\n  Purity distribution (all active neurons):")
    print(f"    Mean:   {purity_arr.mean():.1f}%")
    print(f"    Median: {np.median(purity_arr):.1f}%")
    print(f"    Std:    {purity_arr.std():.1f}%")
    print(f"    Min:    {purity_arr.min():.1f}%")
    print(f"    Max:    {purity_arr.max():.1f}%")

    # ── Paper sentence ────────────────────────────────────────────────────
    print(f"\n  *** For the paper: ***")
    print(f"  \"We observe an average genre purity of {overall_avg_purity:.1f}%, compared")
    print(f"  to a dataset baseline of {baseline_purity:.1f}%, indicating that most neurons")
    print(f"  encode cross-genre features.\"")

    # ── Build figure ──────────────────────────────────────────────────────
    print("\nGenerating figure...")

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[3.5, 1.8],
                           width_ratios=[4, 1],
                           hspace=0.45, wspace=0.08)

    # ── (a) Heatmap ──────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 0])

    im = ax_heat.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0,
                         interpolation="nearest")

    ax_heat.set_xticks(range(0, args.top_neurons, 5))
    ax_heat.set_xticklabels([str(top_neuron_indices[i]) for i in range(0, args.top_neurons, 5)],
                             rotation=90, fontsize=4.5)
    ax_heat.set_yticks(range(n_genres))
    ax_heat.set_yticklabels(genre_labels, fontsize=5.5)
    ax_heat.set_xlabel("Neuron index (sorted by mean activation)", fontsize=6.5)
    ax_heat.set_title("(a) Genre fraction in top-$K$ activating samples per neuron",
                      fontweight="bold", loc="left", fontsize=7)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01, shrink=0.9)
    cbar.set_label("Fraction", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)

    # ── (b) Dataset baseline (right of heatmap) ──────────────────────────
    ax_base = fig.add_subplot(gs[0, 1])

    baseline_vals = [baseline_pcts.get(g, 0) for g in top_genre_ids]
    y = np.arange(n_genres)
    ax_base.barh(y, baseline_vals, color="#bdd7e7", edgecolor="white", linewidth=0.3)
    ax_base.set_yticks([])
    ax_base.set_xlabel("%", fontsize=6)
    ax_base.set_title("Dataset\nbaseline", fontweight="bold", fontsize=6, loc="center")
    ax_base.spines["top"].set_visible(False)
    ax_base.spines["right"].set_visible(False)
    ax_base.spines["left"].set_visible(False)
    ax_base.set_ylim(ax_heat.get_ylim())
    ax_base.invert_yaxis()

    # ── (c) Purity histogram ─────────────────────────────────────────────
    ax_purity = fig.add_subplot(gs[1, :])

    counts_p, edges_p, patches_p = ax_purity.hist(
        purity_arr, bins=50, color="#2171b5", edgecolor="white", linewidth=0.3, zorder=3)

    # Mark baseline
    ax_purity.axvline(x=baseline_purity, color="#e34a33", linewidth=1,
                      linestyle="--", zorder=4, label=f"Dataset baseline ({baseline_purity:.1f}%)")
    # Mark mean purity
    ax_purity.axvline(x=overall_avg_purity, color="#2ca02c", linewidth=1,
                      linestyle="-", zorder=4, label=f"Mean purity ({overall_avg_purity:.1f}%)")

    ax_purity.set_xlabel("Genre purity (%)")
    ax_purity.set_ylabel("Neuron count")
    ax_purity.set_title("(b) Distribution of genre purity across all active neurons",
                        fontweight="bold", loc="left")
    ax_purity.legend(frameon=True, fancybox=False, edgecolor="0.8", fontsize=5.5, loc="upper right")
    ax_purity.spines["top"].set_visible(False)
    ax_purity.spines["right"].set_visible(False)
    ax_purity.yaxis.grid(True, linewidth=0.2, alpha=0.4, zorder=0)
    ax_purity.set_axisbelow(True)

    # ── Save ──────────────────────────────────────────────────────────────
    plt.savefig(args.output)
    print(f"\nSaved to {args.output}")

    png_path = args.output.rsplit(".", 1)[0] + ".png"
    plt.savefig(png_path)
    print(f"Saved to {png_path}")
    plt.close()


if __name__ == "__main__":
    main()