"""
Print genre counts for the top-k activating songs of a given neuron.

Usage:
    python neuron_genres.py \
        --dataset orcd/scratch/window1125_layer-1/dataset \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
        --neuron 5773

    # For the FMA large supplement (use --full to skip train/test split):
    python neuron_genres.py \
        --dataset orcd/scratch/window1125_layer-1/fma_large_supplement/dataset \
        --latent_dir orcd/scratch/window1125_layer-1/latent_analysis_flat_1e-1_rho08_32 \
        --neuron 5773 \
        --full
"""

import argparse
import collections
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--neuron", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true",
                        help="Use the full dataset instead of the test split")
    args = parser.parse_args()

    # Load dataset
    dataset = load_from_disk(args.dataset)
    if args.full:
        test_set = dataset
    else:
        splits = dataset.train_test_split(test_size=0.2, seed=args.seed)
        test_val = splits["test"].train_test_split(test_size=0.5, seed=args.seed)
        test_set = test_val["test"]

    # Load top-k cache
    topk_cache = torch.load(
        os.path.join(args.latent_dir, "top_k_cache.pt"),
        map_location="cpu",
    )
    top_indices = topk_cache["top_indices"]
    top_values = topk_cache["top_values"]
    k = topk_cache["k"]

    neuron = args.neuron
    genre_counts = collections.Counter()

    print(f"Neuron #{neuron} — Top {k} activating tracks:\n")

    for rank in range(k):
        sample_idx = top_indices[neuron][rank].item()
        activation = top_values[neuron][rank].item()
        row = test_set[sample_idx]
        title = row.get("title", "?")
        artist = row.get("artist", "?")
        genres = row.get("genres", [])
        genre_names = [GENRE_NAMES.get(g, str(g)) for g in genres]

        for g in genres:
            genre_counts[g] += 1

        print(f"  #{rank+1} ({activation:.4f}): \"{title}\" by {artist} [{', '.join(genre_names)}]")

    # Print genre summary
    named_counts = [(GENRE_NAMES.get(g, str(g)), c) for g, c in genre_counts.most_common()]
    summary = ", ".join(f"{name}: {count}" for name, count in named_counts)
    print(f"\nGenre counts: {summary}")
    print(f"Total genre tags: {sum(genre_counts.values())} across {k} tracks")


if __name__ == "__main__":
    main()