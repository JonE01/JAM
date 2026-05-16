"""
Merge FMA-medium audio into a pre-computed MERT output dataset.

Matches rows by the 'url' column (unique per track in FMA).
Saves audio as base64-encoded MP3 strings so they can be played
directly in a browser via a data-URI:
    <audio src="data:audio/mpeg;base64,{row['audio_b64']}">

Usage:
    python merge_audio_into_mert.py \
        --mert_dataset /scratch/joneba/mert_medium/window1125_layer-1/dataset \
        --output_dir   /scratch/joneba/mert_medium/window1125_layer-1/dataset_with_audio \
        --hf_token     YOUR_TOKEN
"""

import argparse
import base64
import os

from datasets import load_dataset, load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert_dataset", type=str, required=True,
                        help="Path to the saved MERT output dataset on disk")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the merged dataset")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--fma_dataset", type=str,
                        default="benjamin-paine/free-music-archive-medium",
                        help="HuggingFace dataset ID for FMA")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    # ── 1. Load the MERT dataset (preserves row order) ─────────────────────
    print("Loading MERT dataset...")
    mert_ds = load_from_disk(args.mert_dataset)
    print(f"  {len(mert_ds)} rows, columns: {mert_ds.column_names}")

    if "url" not in mert_ds.column_names:
        raise ValueError("MERT dataset must have a 'url' column for matching")

    # ── 2. Load FMA medium ─────────────────────────────────────────────────
    print("Loading FMA medium dataset (this may take a while)...")
    from datasets import Audio
    fma_ds = load_dataset(
        args.fma_dataset,
        split="train",
        token=hf_token,
        revision="main",
    ).cast_column("audio", Audio(decode=False))
    print(f"  {len(fma_ds)} rows")

    # ── 3. Build a URL → audio bytes lookup ────────────────────────────────
    print("Building URL -> audio index...")
    url_to_idx = {}
    for i in range(len(fma_ds)):
        url = fma_ds[i]["url"]
        url_to_idx[url] = i

    # ── 4. Match and extract audio as base64 MP3 ───────────────────────────
    print("Matching audio to MERT rows...")
    audio_b64_list = []
    matched = 0
    missing = 0

    for i in range(len(mert_ds)):
        url = mert_ds[i]["url"]
        fma_idx = url_to_idx.get(url)

        if fma_idx is not None:
            audio_bytes = fma_ds[fma_idx]["audio"]["bytes"]
            audio_b64_list.append(base64.b64encode(audio_bytes).decode("ascii"))
            matched += 1
        else:
            audio_b64_list.append(None)
            missing += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(mert_ds)} rows ({matched} matched, {missing} missing)")

    print(f"Done: {matched} matched, {missing} missing out of {len(mert_ds)}")

    # ── 5. Add the audio column (preserves row order) ──────────────────────
    mert_ds = mert_ds.add_column("audio_b64", audio_b64_list)

    # ── 6. Save ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    mert_ds.save_to_disk(args.output_dir)
    print(f"Saved merged dataset to {args.output_dir}")
    print(f"  {len(mert_ds)} rows, columns: {mert_ds.column_names}")

    # ── 7. Verify ordering wasn't changed ──────────────────────────────────
    verify_ds = load_from_disk(args.output_dir)
    for i in range(min(5, len(verify_ds))):
        assert verify_ds[i]["url"] == mert_ds[i]["url"], f"Row {i} order mismatch!"
    print("Row order verified.")


if __name__ == "__main__":
    main()