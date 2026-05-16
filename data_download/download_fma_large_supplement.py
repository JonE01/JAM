"""
Download 10,000 songs from FMA-large that aren't in FMA-medium,
split into two halves:
  - 5,000 tracks matching the genre distribution of the test set
  - 5,000 tracks balanced across the top 20 most popular genres (250 each)

Then process them through MERT and save a dataset compatible with generate_latents.py.

Usage:
    python download_fma_large_supplement.py \
        --mert_dataset orcd/scratch/window1125_layer-1/dataset \
        --output_dir orcd/scratch/window1125_layer-1/fma_large_supplement \
        --hf_token YOUR_TOKEN

Step 1: Run this script to build the supplemental dataset.
Step 2: Run generate_latents.py with --dataset pointing to the output.
"""

import argparse
import os
import collections
import random

import numpy as np
import torch
import torchaudio.transforms as T
from datasets import load_dataset, load_from_disk, Dataset as HFDataset, Audio
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import librosa
import io


# ── Audio decoding (same as your training pipeline) ───────────────────────

def decode_audio(item):
    audio_data = item["audio"]
    if isinstance(audio_data, dict):
        if "bytes" in audio_data and audio_data["bytes"] is not None:
            audio_bytes = audio_data["bytes"]
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            return {"array": audio_array, "sample_rate": sr}
        elif "array" in audio_data:
            return {"array": np.array(audio_data["array"]), "sample_rate": audio_data.get("sampling_rate", 44100)}
    raise ValueError("Cannot decode audio")


# ── Top 20 most popular genres ────────────────────────────────────────────
# Selected based on prevalence across FMA and general music datasets.
# These are broad, well-represented genres that give good coverage.

TOP_20_GENRES = [
    51,   # Electronic
    53,   # Experimental
    130,  # Rock
    70,   # Hip-Hop
    58,   # Folk
    116,  # Pop
    79,   # Instrumental
    31,   # Classical
    82,   # Jazz
    5,    # Ambient
    123,  # Punk
    94,   # Metal
    139,  # Soul-RnB
    36,   # Country
    18,   # Blues
    77,   # Indie-Rock
    106,  # Noise
    91,   # Lo-Fi
    136,  # Singer-Songwriter
    63,   # Funk
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert_dataset", type=str, required=True,
                        help="Path to the existing MERT dataset (to find medium URLs and test split)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the supplemental dataset")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=1125)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_total", type=int, default=10000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    target_per_half = args.target_total // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load existing MERT dataset and get medium URLs ─────────────────
    print("Loading existing MERT dataset...")
    mert_ds = load_from_disk(args.mert_dataset)
    medium_urls = set(mert_ds["url"])
    print(f"  {len(medium_urls)} medium URLs to exclude")

    # Get test split genre distribution
    splits = mert_ds.train_test_split(test_size=0.2, seed=args.seed)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=args.seed)
    test_set = test_val["test"]

    # Count genres in test set
    test_genre_counts = collections.Counter()
    for i in range(len(test_set)):
        for g in test_set[i]["genres"]:
            test_genre_counts[g] += 1

    total_test_genres = sum(test_genre_counts.values())
    test_genre_dist = {g: c / total_test_genres for g, c in test_genre_counts.items()}

    print(f"\n  Test set genre distribution (top 15):")
    for g, c in test_genre_counts.most_common(15):
        print(f"    {g}: {c} ({test_genre_dist[g]*100:.1f}%)")

    # Compute targets for first half (proportional to test set)
    test_half_targets = {}
    for g, frac in test_genre_dist.items():
        count = max(1, int(frac * target_per_half))
        test_half_targets[g] = count

    # Normalize so total = target_per_half
    current_total = sum(test_half_targets.values())
    if current_total > target_per_half:
        scale = target_per_half / current_total
        test_half_targets = {g: max(1, int(c * scale)) for g, c in test_half_targets.items()}

    # Compute targets for second half (balanced top 20)
    per_genre_balanced = target_per_half // len(TOP_20_GENRES)  # 250 each
    balanced_targets = {g: per_genre_balanced for g in TOP_20_GENRES}

    print(f"\n  First half targets (test-proportional): {sum(test_half_targets.values())} tracks across {len(test_half_targets)} genres")
    print(f"  Second half targets (balanced): {sum(balanced_targets.values())} tracks across {len(balanced_targets)} genres ({per_genre_balanced} each)")
    print(f"\n  Top 20 genres for balanced half:")
    for g in TOP_20_GENRES:
        print(f"    {g}: {per_genre_balanced}")

    # ── 2. Load FMA large (streaming to avoid downloading everything) ─────
    print("\nLoading FMA-large (streaming)...")
    fma_large = load_dataset(
        "benjamin-paine/free-music-archive-large",
        split="train",
        streaming=True,
        token=hf_token,
    )
    # Can't use cast_column on streaming datasets, so disable audio decoding:
    fma_large = fma_large.cast_column("audio", Audio(decode=False))
    # ── 3. Iterate and collect matching tracks ────────────────────────────
    print("Scanning FMA-large for matching tracks...")

    test_half_collected = collections.Counter()
    balanced_half_collected = collections.Counter()
    test_half_items = []
    balanced_half_items = []
    skipped = 0
    scanned = 0

    for item in fma_large:
        scanned += 1
        if scanned % 5000 == 0:
            t_count = len(test_half_items)
            b_count = len(balanced_half_items)
            print(f"  Scanned {scanned}, collected: {t_count} test-proportional + {b_count} balanced = {t_count + b_count}")

        # Skip if already in medium
        url = item.get("url", "")
        if url in medium_urls:
            skipped += 1
            continue

        genres = item.get("genres", [])
        if not genres:
            continue

        # Check if this track fits the test-proportional half
        if len(test_half_items) < target_per_half:
            for g in genres:
                if g in test_half_targets and test_half_collected[g] < test_half_targets[g]:
                    test_half_items.append(item)
                    for gg in genres:
                        test_half_collected[gg] += 1
                    break

        # Check if this track fits the balanced half
        if len(balanced_half_items) < target_per_half:
            for g in genres:
                if g in balanced_targets and balanced_half_collected[g] < balanced_targets[g]:
                    balanced_half_items.append(item)
                    for gg in genres:
                        balanced_half_collected[gg] += 1
                    break

        # Stop when both halves are full
        if len(test_half_items) >= target_per_half and len(balanced_half_items) >= target_per_half:
            break

    print(f"\n  Scanned: {scanned}")
    print(f"  Skipped (in medium): {skipped}")
    print(f"  Test-proportional half: {len(test_half_items)}")
    print(f"  Balanced half: {len(balanced_half_items)}")

    all_items = test_half_items + balanced_half_items
    print(f"  Total collected: {len(all_items)}")

    if len(all_items) == 0:
        print("ERROR: No tracks collected. Check dataset name and token.")
        return

    # ── 4. Load MERT model ────────────────────────────────────────────────
    print("\nLoading MERT model...")
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert_model.eval()

    # ── 5. Process through MERT and save ──────────────────────────────────
    print(f"\nProcessing {len(all_items)} tracks through MERT...")

    KEEP_COLS = [
        "title", "url", "artist", "composer", "lyricist", "publisher",
        "genres", "tags", "released", "language", "listens", "artist_url",
        "artist_website", "album_title", "album_url", "license", "copyright",
        "explicit", "instrumental", "allow_commercial_use", "allow_derivatives",
        "require_attribution", "require_share_alike",
    ]

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume support
    existing_ckpts = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_") and f.endswith(".parquet")
    ]) if os.path.exists(checkpoint_dir) else []
    start_idx = 0
    checkpoint_idx = len(existing_ckpts)
    if existing_ckpts:
        # Count samples in existing checkpoints
        for ckpt_file in existing_ckpts:
            ckpt_ds = HFDataset.from_parquet(os.path.join(checkpoint_dir, ckpt_file))
            start_idx += len(ckpt_ds)
        print(f"  Resuming from sample {start_idx}, checkpoint {checkpoint_idx}")

    mert_outputs = []
    metadata_rows = []
    processed = 0
    failed = 0

    for i, item in enumerate(all_items):
        if i < start_idx:
            continue

        # Decode audio
        try:
            decoded = decode_audio(item)
            audio = decoded["array"]
            sr = decoded["sample_rate"]

            if sr != processor.sampling_rate:
                resampler = T.Resample(sr, processor.sampling_rate)
                audio_tensor = resampler(torch.tensor(audio).float())
            else:
                audio_tensor = torch.tensor(audio).float()

            # Truncate to 30s
            max_samples = processor.sampling_rate * 30
            audio_tensor = audio_tensor[:max_samples]

            # Pad to window_size equivalent
            target_len = processor.sampling_rate * 30
            if audio_tensor.shape[0] < target_len:
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, target_len - audio_tensor.shape[0]))

        except Exception as e:
            failed += 1
            continue

        # MERT forward pass
        try:
            with torch.no_grad():
                audio_input = audio_tensor.unsqueeze(0).to(device)
                outputs = mert_model(audio_input, output_hidden_states=True)
                hidden = outputs.hidden_states[args.layer_idx]  # [1, T, 768]

                T_len = hidden.shape[1]
                if T_len >= args.window_size:
                    hidden = hidden[:, :args.window_size, :]
                else:
                    hidden = torch.nn.functional.pad(hidden, (0, 0, 0, args.window_size - T_len))

                mert_outputs.append(hidden[0].cpu().numpy().tobytes())

            # Collect metadata
            available_cols = [c for c in KEEP_COLS if c in item]
            row = {col: item[col] for col in available_cols}
            metadata_rows.append(row)
            processed += 1

        except Exception as e:
            failed += 1
            continue

        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(all_items)} (failed: {failed})")

        # Checkpoint
        if len(mert_outputs) >= args.checkpoint_every:
            available_cols = list(metadata_rows[0].keys())
            ckpt_dict = {"MERT_output": mert_outputs}
            for col in available_cols:
                ckpt_dict[col] = [r.get(col) for r in metadata_rows]
            ckpt_ds = HFDataset.from_dict(ckpt_dict)
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{checkpoint_idx:04d}.parquet")
            ckpt_ds.to_parquet(ckpt_path)
            print(f"  Saved checkpoint {checkpoint_idx} ({len(mert_outputs)} samples)")
            mert_outputs = []
            metadata_rows = []
            checkpoint_idx += 1

    # Save remaining
    if mert_outputs:
        available_cols = list(metadata_rows[0].keys())
        ckpt_dict = {"MERT_output": mert_outputs}
        for col in available_cols:
            ckpt_dict[col] = [r.get(col) for r in metadata_rows]
        ckpt_ds = HFDataset.from_dict(ckpt_dict)
        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{checkpoint_idx:04d}.parquet")
        ckpt_ds.to_parquet(ckpt_path)
        print(f"  Saved final checkpoint ({len(mert_outputs)} samples)")

    print(f"\n  Total processed: {processed}")
    print(f"  Total failed: {failed}")

    # ── 6. Combine checkpoints ────────────────────────────────────────────
    print("\nCombining checkpoints...")
    import glob
    from datasets import concatenate_datasets

    parquet_files = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_*.parquet")))
    partial_datasets = [HFDataset.from_parquet(f) for f in parquet_files]
    final_dataset = concatenate_datasets(partial_datasets)

    # Deduplicate by URL
    if "url" in final_dataset.column_names:
        urls_seen = set()
        unique_indices = []
        for i, url in enumerate(final_dataset["url"]):
            if url not in urls_seen:
                urls_seen.add(url)
                unique_indices.append(i)
        before = len(final_dataset)
        final_dataset = final_dataset.select(unique_indices)
        after = len(final_dataset)
        if before != after:
            print(f"  Removed {before - after} duplicates")

    final_path = os.path.join(args.output_dir, "dataset")
    final_dataset.save_to_disk(final_path)
    print(f"  Saved {len(final_dataset)} samples -> {final_path}")
    print(f"  Columns: {final_dataset.column_names}")

    # Clean up checkpoints
    import shutil
    shutil.rmtree(checkpoint_dir)
    print("  Checkpoints deleted.")

    # ── 7. Genre summary ──────────────────────────────────────────────────
    print(f"\nGenre distribution in final dataset:")
    genre_counts = collections.Counter()
    for i in range(len(final_dataset)):
        for g in final_dataset[i]["genres"]:
            genre_counts[g] += 1
    for g, c in genre_counts.most_common(30):
        print(f"  {g}: {c}")

    print("\nDone.")


if __name__ == "__main__":
    main()