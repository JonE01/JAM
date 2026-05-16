"""
Generate paper-ready genre distribution histograms for train/val/test splits.

Usage:
    python genre_histograms.py \
        --dataset orcd/scratch/window1125_layer-1/dataset \
        --output genre_distributions.pdf

Produces a 3-panel figure suitable for CVPR paper inclusion.
"""

import argparse
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datasets import load_from_disk

# Full genre ID → name mapping
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
    51: "Electronic", 52: "Europe", 53: "Experimental", 54: "Experimental Pop",
    55: "Fado", 56: "Field Recordings", 57: "Flamenco", 58: "Folk",
    59: "Freak-Folk", 60: "Free-Folk", 61: "Free-Jazz", 62: "French",
    63: "Funk", 64: "Garage", 65: "Glitch", 66: "Gospel", 67: "Goth",
    68: "Grindcore", 69: "Hardcore", 70: "Hip-Hop", 71: "Hip-Hop Beats",
    72: "Holiday", 73: "House", 74: "IDM", 75: "Improv", 76: "Indian",
    77: "Indie-Rock", 78: "Industrial", 79: "Instrumental", 80: "International",
    81: "Interview", 82: "Jazz", 83: "Jazz: Out", 84: "Jazz: Vocal",
    85: "Jungle", 86: "Kid-Friendly", 87: "Klezmer", 88: "Krautrock",
    89: "Latin", 90: "Latin America", 91: "Lo-Fi", 92: "Loud-Rock",
    93: "Lounge", 94: "Metal", 95: "Middle East", 96: "Minimal Electronic",
    97: "Minimalism", 98: "Modern Jazz", 99: "Musical Theater",
    100: "Musique Concrete", 101: "N. Indian Traditional", 102: "Nerdcore",
    103: "New Age", 104: "New Wave", 105: "No Wave", 106: "Noise",
    107: "Noise-Rock", 108: "North African", 109: "Novelty", 110: "Nu-Jazz",
    111: "Old-Time / Historic", 112: "Opera", 113: "Pacific", 114: "Poetry",
    115: "Polka", 116: "Pop", 117: "Post-Punk", 118: "Post-Rock",
    119: "Power-Pop", 120: "Progressive", 121: "Psych-Folk", 122: "Psych-Rock",
    123: "Punk", 124: "Radio", 125: "Radio Art", 126: "Radio Theater",
    127: "Rap", 128: "Reggae - Dancehall", 129: "Reggae - Dub", 130: "Rock",
    131: "Rock Opera", 132: "Rockabilly", 133: "Romany (Gypsy)", 134: "Salsa",
    135: "Shoegaze", 136: "Singer-Songwriter", 137: "Skweee", 138: "Sludge",
    139: "Soul-RnB", 140: "Sound Art", 141: "Sound Collage", 142: "Sound Effects",
    143: "Sound Poetry", 144: "Soundtrack", 145: "South Indian Traditional",
    146: "Space-Rock", 147: "Spanish", 148: "Spoken", 149: "Spoken Weird",
    150: "Spoken Word", 151: "Surf", 152: "Symphony", 153: "Synth Pop",
    154: "Talk Radio", 155: "Tango", 156: "Techno", 157: "Thrash",
    158: "Trip-Hop", 159: "Turkish", 160: "Unclassifiable", 161: "Western Swing",
    162: "Wonky", 163: "hiphop",
}


def count_genres(dataset):
    """Count genre occurrences across all samples (each sample can have multiple genres)."""
    counts = collections.Counter()
    for i in range(len(dataset)):
        for g in dataset[i]["genres"]:
            counts[g] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="genre_distributions.pdf")
    parser.add_argument("--top_n", type=int, default=25,
                        help="Show only the top N genres by total count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Load and split
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset)
    splits = dataset.train_test_split(test_size=0.2, seed=args.seed)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=args.seed)

    split_data = {
        "Train": splits["train"],
        "Validation": test_val["train"],
        "Test": test_val["test"],
    }

    # Count genres per split
    print("Counting genres...")
    split_counts = {}
    for name, ds in split_data.items():
        split_counts[name] = count_genres(ds)
        print(f"  {name}: {len(ds)} samples, {sum(split_counts[name].values())} genre tags")

    # Determine top N genres by total count across all splits
    total_counts = collections.Counter()
    for counts in split_counts.values():
        total_counts.update(counts)

    top_genres = [g for g, _ in total_counts.most_common(args.top_n)]

    # Map IDs to names
    genre_labels = [GENRE_NAMES.get(g, str(g)) for g in top_genres]

    # ── Plotting ──────────────────────────────────────────────────────────

    # CVPR-style settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": args.dpi,
        "savefig.dpi": args.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    })

    colors = {
        "Train": "#2171b5",
        "Validation": "#6baed6",
        "Test": "#bdd7e7",
    }

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    x = np.arange(len(top_genres))
    bar_width = 0.7

    for ax, (split_name, counts) in zip(axes, split_counts.items()):
        values = [counts.get(g, 0) for g in top_genres]
        bars = ax.bar(x, values, width=bar_width, color=colors[split_name],
                      edgecolor="white", linewidth=0.3, zorder=3)

        ax.set_ylabel("Count")
        ax.set_title(f"{split_name} ({len(split_data[split_name]):,} samples)",
                     fontweight="bold", loc="left", pad=4)

        # Light grid behind bars
        ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Y-axis formatting
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # X-axis labels on bottom subplot only
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(genre_labels, rotation=55, ha="right",
                              rotation_mode="anchor")
    axes[-1].set_xlabel("")

    # Add "(showing top N)" note
    if args.top_n < len(total_counts):
        fig.text(0.99, 0.01,
                 f"Showing top {args.top_n} of {len(total_counts)} genres by total count",
                 ha="right", va="bottom", fontsize=6, fontstyle="italic", color="0.5")

    # Save
    plt.savefig(args.output)
    print(f"\nSaved to {args.output}")

    # Also save PNG version
    png_path = args.output.rsplit(".", 1)[0] + ".png"
    plt.savefig(png_path)
    print(f"Saved to {png_path}")

    # Print summary table
    print(f"\nTop {args.top_n} genres (total across all splits):")
    print(f"{'Genre':<25} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("-" * 60)
    for g in top_genres:
        name = GENRE_NAMES.get(g, str(g))
        t = split_counts["Train"].get(g, 0)
        v = split_counts["Validation"].get(g, 0)
        te = split_counts["Test"].get(g, 0)
        print(f"{name:<25} {t:>7} {v:>7} {te:>7} {t+v+te:>7}")


if __name__ == "__main__":
    main()