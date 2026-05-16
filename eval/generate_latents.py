"""
Generate latent activation data for per-neuron analysis.

Creates three files:
  1. latent_activations.pt — dict with:
       - "activations": tensor [n_samples, latent_size] (time-averaged latent per sample)
       - metadata (latent_size, expansion_factor, etc.)
  2. top_k_cache.pt — dict with:
       - "top_indices": tensor [latent_size, k] (top-k sample indices per neuron)
       - "top_values":  tensor [latent_size, k] (corresponding activation values)
  3. latent_full.pt — tensor [n_samples, latent_channels, latent_size]
       (full encoding without averaging over time)

Usage:
    python generate_latents.py \
        --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
        --checkpoint orcd/scratch/window1125_layer-1/runs/initial_test/best_model_8.pt \
        --expansion_factor 8 \
        --output_dir orcd/scratch/window1125_layer-1/latent_analysis \
        --top_k 20

To retrieve top-k tracks for a neuron:

    data = torch.load("latent_analysis/latent_activations.pt")
    cache = torch.load("latent_analysis/top_k_cache.pt")
    dataset = load_from_disk("orcd/scratch/window1125_layer-1/dataset_with_audio")

    splits = dataset.train_test_split(test_size=0.2, seed=42)
    test_set = splits["test"].train_test_split(test_size=0.5, seed=42)["test"]

    neuron = 42
    for i in range(cache["k"]):
        row = test_set[cache["top_indices"][neuron][i].item()]
        print(f"{row['title']} — {cache['top_values'][neuron][i]:.4f}")
"""

import argparse
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import load_from_disk


STRIDE = 25


# ── Model definitions (must match training code) ──────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_size, stride=STRIDE):
        super().__init__()
        self.stride = stride
        self.input_dim = input_dim
        self.features = input_dim[2]
        self.window_size = input_dim[1]
        self.latent_size = latent_size
        self.fully_connected1 = nn.Linear(self.features, latent_size)
        self.conv_1d = nn.Conv1d(self.features, self.features, kernel_size=self.stride, stride=self.stride)

    def forward(self, input):
        transposed = input.transpose(1, 2)
        conv = self.conv_1d(transposed)
        conv = conv.transpose(1, 2)
        fc = self.fully_connected1(conv)
        return torch.sigmoid(fc)


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_size, stride=STRIDE):
        super().__init__()
        self.stride = stride
        self.output_dim = output_dim
        self.features = output_dim[2]
        self.window_size = output_dim[1]
        self.latent_size = latent_size
        self.fully_connected1 = nn.Linear(latent_size, self.features)
        self.inv_conv1d = nn.ConvTranspose1d(self.features, self.features, kernel_size=self.stride, stride=self.stride)

    def forward(self, input):
        lin_output = self.fully_connected1(input)
        lin_output = lin_output.transpose(1, 2)
        inv_conv = self.inv_conv1d(lin_output)
        inv_conv = inv_conv.transpose(1, 2)
        return inv_conv


class SAE(nn.Module):
    def __init__(self, input_dim, latent_size, loss_fn=F.mse_loss, lr=1e-4, l2=0., rho=.05):
        super().__init__()
        self.latent_size = latent_size
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, latent_size)
        self.decoder = Decoder(input_dim, latent_size)
        self.loss_fn = loss_fn
        self.rho = rho

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded


# ── Dataset ───────────────────────────────────────────────────────────────

class MERTPrecomputedDataset(TorchDataset):
    def __init__(self, hf_dataset, window_size, feature_dim=768):
        self.dataset = hf_dataset
        self.window_size = window_size
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tensor = torch.frombuffer(
            bytearray(item["MERT_output"]), dtype=torch.float32
        ).reshape(self.window_size, self.feature_dim)
        return tensor, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--expansion_factor", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=1125)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load dataset and create the same test split ────────────────────
    print(f"Loading dataset from {args.dataset}...")
    full_dataset = load_from_disk(args.dataset)
    print(f"  {len(full_dataset)} rows")

    splits = full_dataset.train_test_split(test_size=0.2, seed=args.seed)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=args.seed)
    test_hf = test_val["test"]
    print(f"  Test split: {len(test_hf)} rows")

    test_ds = MERTPrecomputedDataset(test_hf, args.window_size, args.feature_dim)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── 2. Load trained encoder ───────────────────────────────────────────
    latent_size = args.feature_dim * args.expansion_factor
    input_dim = (args.batch_size, args.window_size, args.feature_dim)
    latent_channels = args.window_size // STRIDE

    model = SAE(input_dim, latent_size)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded model, latent_size={latent_size}, stride={STRIDE}, latent_channels={latent_channels}")

    # Size estimate
    n_test = len(test_hf)
    full_size_gb = n_test * latent_channels * latent_size * 4 / 1e9
    avg_size_gb = n_test * latent_size * 4 / 1e9
    print(f"\n  Estimated output sizes:")
    print(f"    latent_activations.pt (averaged): ~{avg_size_gb:.2f} GB")
    print(f"    latent_full.pt (full):            ~{full_size_gb:.2f} GB")
    print(f"    top_k_cache.pt:                   ~{n_test * args.top_k * 8 / 1e6:.1f} MB")

    # ── 3. Single-pass: collect both averaged and full latents ────────────
    print("\nGenerating latent activations...")
    all_avg = []
    all_full = []

    with torch.no_grad():
        for batch_idx, (data, indices) in enumerate(test_loader):
            data = data.to(device)
            encoded = model.encoder(data)           # [batch, latent_channels, latent_size]
            all_avg.append(encoded.mean(dim=1).cpu())  # [batch, latent_size]
            all_full.append(encoded.cpu())             # [batch, latent_channels, latent_size]

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}, samples: {sum(a.shape[0] for a in all_avg)}")

    activations = torch.cat(all_avg, dim=0)       # [n_samples, latent_size]
    full_latents = torch.cat(all_full, dim=0)      # [n_samples, latent_channels, latent_size]
    print(f"  Averaged activations shape: {activations.shape}")
    print(f"  Full latents shape:         {full_latents.shape}")

    # ── 4. Compute top-k per neuron ───────────────────────────────────────
    print(f"\nComputing top-{args.top_k} activating samples per neuron...")
    top_values, top_indices = activations.topk(args.top_k, dim=0)
    top_values = top_values.t()     # [latent_size, top_k]
    top_indices = top_indices.t()   # [latent_size, top_k]

    # ── 5. Save ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    activations_path = os.path.join(args.output_dir, "latent_activations.pt")
    torch.save({
        "activations": activations,
        "latent_size": latent_size,
        "expansion_factor": args.expansion_factor,
        "window_size": args.window_size,
        "feature_dim": args.feature_dim,
        "stride": STRIDE,
        "latent_channels": latent_channels,
        "n_samples": activations.shape[0],
        "seed": args.seed,
    }, activations_path)
    print(f"  Saved activations -> {activations_path}")

    topk_path = os.path.join(args.output_dir, "top_k_cache.pt")
    torch.save({
        "top_indices": top_indices,
        "top_values": top_values,
        "k": args.top_k,
    }, topk_path)
    print(f"  Saved top-k cache -> {topk_path}")

    full_path = os.path.join(args.output_dir, "latent_full.pt")
    # torch.save(full_latents, full_path)
    # print(f"  Saved full latents -> {full_path}")

    # ── 6. Summary stats ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Samples:            {activations.shape[0]}")
    print(f"  Latent size:        {latent_size}")
    print(f"  Latent channels:    {latent_channels}")
    print(f"  Stride:             {STRIDE}")
    print(f"  Global mean:        {activations.mean():.4f}")
    print(f"  Global std:         {activations.std():.4f}")
    print(f"  Sparsity (>0.1):    {(activations > 0.1).float().mean():.4f}")
    print(f"  Sparsity (>0.5):    {(activations > 0.5).float().mean():.4f}")
    print(f"  Dead neurons (mean < 0.01): {(activations.mean(dim=0) < 0.01).sum().item()} / {latent_size}")
    print(f"  File sizes:")
    print(f"    activations:      {os.path.getsize(activations_path) / 1e6:.1f} MB")
    print(f"    top_k_cache:      {os.path.getsize(topk_path) / 1e6:.1f} MB")
    # print(f"    full latents:     {os.path.getsize(full_path) / 1e9:.2f} GB")

    # ── 7. Verification ──────────────────────────────────────────────────
    print(f"\nTop-{min(5, args.top_k)} for neuron 0:")
    for rank in range(min(5, args.top_k)):
        sample_idx = top_indices[0][rank].item()
        val = top_values[0][rank].item()
        row = test_hf[sample_idx]
        title = row.get("title", "?")
        artist = row.get("artist", "?")
        print(f"  #{rank+1}: \"{title}\" by {artist} — activation: {val:.4f}")

    # Verify full latent matches averaged
    sample_full = full_latents[0]                      # [latent_channels, latent_size]
    sample_avg_from_full = sample_full.mean(dim=0)     # [latent_size]
    sample_avg = activations[0]                        # [latent_size]
    diff = (sample_avg_from_full - sample_avg).abs().max().item()
    print(f"\n  Avg/full consistency check (max diff): {diff:.8f}")

    print("\nDone.")


if __name__ == "__main__":
    main()