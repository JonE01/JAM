"""
Download FMA-large, compute MERT hidden-state outputs for every track,
and save a unified dataset containing all metadata columns, MERT outputs,
and base64-encoded audio.

Saves parquet checkpoints every CHECKPOINT_EVERY batches so the job can
be resumed if it gets preempted.

Usage (inside your SLURM job):
    python download_FMA_large.py \
        --output_dir /scratch/joneba/fma_large_mert \
        --hf_token   YOUR_TOKEN          # or set HF_TOKEN env var
        --layer_idx  -1                   # MERT hidden layer to extract
        --window_size 1125                # ~30 s at MERT frame rate
        --batch_size  8
"""

import argparse
import base64
import glob
import io
import os
import shutil

import datasets
import librosa
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import (
    Audio,
    Dataset as HFDataset,
    concatenate_datasets,
    load_dataset,
)
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, Wav2Vec2FeatureExtractor


# ═══════════════════════════════════════════════════════════════════════
# MERT wrapper
# ═══════════════════════════════════════════════════════════════════════
class MERT_model:
    def __init__(self, mert_model, processor, window_size=1125,
                 deterministic=True, layer_idx=-1):
        self.mert_model = mert_model
        self.processor = processor
        self.layer_idx = layer_idx
        self.deterministic = deterministic
        self.window_size = window_size

    def get_hidden_states(self, audio_array):
        audio_array = audio_array.to(next(self.mert_model.parameters()).device)
        with torch.no_grad():
            outputs = self.mert_model(audio_array, output_hidden_states=True)

        hidden = outputs.hidden_states[self.layer_idx]

        T_len = hidden.shape[1]
        if T_len >= self.window_size:
            if self.deterministic:
                hidden = hidden[:, :self.window_size, :]
            else:
                start = torch.randint(0, T_len - self.window_size + 1, (1,)).item()
                hidden = hidden[:, start:start + self.window_size, :]
        else:
            hidden = F.pad(hidden, (0, 0, 0, self.window_size - T_len))

        return hidden  # [batch, window_size, 768]


# ═══════════════════════════════════════════════════════════════════════
# Audio dataset / dataloader helpers
# ═══════════════════════════════════════════════════════════════════════
def decode_audio(item):
    audio_bytes = item["audio"]["bytes"]
    try:
        array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception:
        # Fallback for MP3s that soundfile can't open
        array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True,
                                 res_type='kaiser_fast', backend='audioread')
    item["array"] = array
    item["sample_rate"] = sr
    return item


def collate_fn(batch, max_samples=24000 * 30):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.tensor([]), [], []
    audios, indices = zip(*batch)
    audios = [a[:max_samples] for a in audios]
    max_len = max(a.shape[0] for a in audios)
    padded = torch.stack([F.pad(a, (0, max_len - a.shape[0])) for a in audios])
    return padded, list(indices)


class FMADataset(TorchDataset):
    def __init__(self, hf_dataset, processor, sample_rate=44100):
        self.dataset = hf_dataset
        self.resample_rate = processor.sampling_rate
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            item = decode_audio(item)
        except Exception as e:
            print(f"Skipping clip {idx}: {e}")
            return None
        audio = item["array"]
        sample_rate = item.get("sample_rate", self.sample_rate)
        try:
            if sample_rate != self.resample_rate:
                resampler = T.Resample(sample_rate, self.resample_rate)
                audio_array = resampler(torch.tensor(audio).float())
            else:
                audio_array = torch.tensor(audio).float()
        except Exception as e:
            print(f"Failed clip {idx}: {e}")
            return None
        return audio_array, idx


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════
KEEP_COLS = [
    "title", "url", "artist", "composer", "lyricist", "publisher",
    "genres", "tags", "released", "language", "listens", "artist_url",
    "artist_website", "album_title", "album_url", "license", "copyright",
    "explicit", "instrumental", "allow_commercial_use", "allow_derivatives",
    "require_attribution", "require_share_alike",
]

CHECKPOINT_EVERY = 50  # batches between parquet saves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for outputs (checkpoints + final dataset)")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--fma_dataset", type=str,
                        default="benjamin-paine/free-music-archive-large",
                        help="HuggingFace dataset ID for FMA-large")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--window_size", type=int, default=1125)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    layer_idx = args.layer_idx
    window_size = args.window_size
    batch_size = args.batch_size

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    final_dir = os.path.join(args.output_dir,
                             f"window{window_size}_layer{layer_idx}", "dataset")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── 1. Load MERT model ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mert_model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    ).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    )
    MERT = MERT_model(mert_model, processor,
                      window_size=window_size, layer_idx=layer_idx)

    # ── 2. Load FMA-large ──────────────────────────────────────────────
    print("Loading FMA-large dataset (this may take a while)...")
    fma_dataset = load_dataset(
        args.fma_dataset,
        split="train",
        token=hf_token,
        revision="main",
    ).cast_column("audio", Audio(decode=False))
    print(f"  {len(fma_dataset)} tracks loaded")

    available_cols = [c for c in KEEP_COLS if c in fma_dataset.column_names]
    metadata_table = fma_dataset.select_columns(available_cols)

    # ── 3. Dataloader ──────────────────────────────────────────────────
    torch_ds = FMADataset(fma_dataset, processor)
    loader = DataLoader(torch_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)

    # ── 4. Resume support ──────────────────────────────────────────────
    existing_checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_") and f.endswith(".parquet")
    ]) if os.path.exists(checkpoint_dir) else []
    start_batch = len(existing_checkpoints) * CHECKPOINT_EVERY
    checkpoint_idx = len(existing_checkpoints)
    if start_batch > 0:
        print(f"Resuming from batch {start_batch} "
              f"({len(existing_checkpoints)} existing checkpoints)")

    # ── 5. Process batches ─────────────────────────────────────────────
    mert_outputs = []
    audio_b64_list = []
    metadata_rows = []
    batch_count = 0

    for batch_idx, (audio_array, indices) in enumerate(loader):
        if batch_idx < start_batch:
            continue
        if audio_array.numel() == 0:
            continue

        with torch.no_grad():
            hidden = MERT.get_hidden_states(audio_array)

        for i in range(hidden.shape[0]):
            idx = indices[i]
            # MERT hidden states
            mert_outputs.append(hidden[i].cpu().numpy().tobytes())
            # Audio as base64 MP3
            audio_bytes = fma_dataset[idx]["audio"]["bytes"]
            audio_b64_list.append(
                base64.b64encode(audio_bytes).decode("ascii")
            )
            # Metadata
            metadata_rows.append(metadata_table[idx])

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processed batch {batch_count}, "
                  f"total samples: {len(mert_outputs)}")

        # Checkpoint
        if batch_count % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_dir, checkpoint_idx,
                             mert_outputs, audio_b64_list,
                             metadata_rows, available_cols)
            mert_outputs, audio_b64_list, metadata_rows = [], [], []
            checkpoint_idx += 1

    # Final leftover
    if mert_outputs:
        _save_checkpoint(checkpoint_dir, checkpoint_idx,
                         mert_outputs, audio_b64_list,
                         metadata_rows, available_cols)

    # ── 6. Merge checkpoints into one dataset ──────────────────────────
    print("Merging checkpoints...")
    parquet_files = sorted(glob.glob(
        os.path.join(checkpoint_dir, "ckpt_*.parquet")
    ))
    partial = [HFDataset.from_parquet(f) for f in parquet_files]
    final_dataset = concatenate_datasets(partial)

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
            print(f"Removed {before - after} duplicates")

    os.makedirs(final_dir, exist_ok=True)
    final_dataset.save_to_disk(final_dir)
    print(f"Saved {len(final_dataset)} samples -> {final_dir}")
    print(f"Columns: {final_dataset.column_names}")

    # Clean up checkpoints
    shutil.rmtree(checkpoint_dir)
    print("Checkpoints deleted. Done!")


def _save_checkpoint(checkpoint_dir, checkpoint_idx,
                     mert_outputs, audio_b64_list,
                     metadata_rows, available_cols):
    ckpt_dict = {
        "MERT_output": mert_outputs,
        "audio_b64": audio_b64_list,
    }
    for col in available_cols:
        ckpt_dict[col] = [r[col] for r in metadata_rows]
    ckpt_ds = HFDataset.from_dict(ckpt_dict)
    ckpt_path = os.path.join(checkpoint_dir,
                             f"ckpt_{checkpoint_idx:04d}.parquet")
    ckpt_ds.to_parquet(ckpt_path)
    print(f"Saved checkpoint {checkpoint_idx} "
          f"({len(mert_outputs)} samples) -> {ckpt_path}")


if __name__ == "__main__":
    main()