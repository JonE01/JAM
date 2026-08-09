# JAM — Music Sparse Autoencoder Project

## What this project is

**Goal:** Train a Sparse Autoencoder (SAE) on the internal activations of **MERT** (`m-a-p/MERT-v1-95M`, a self-supervised music foundation model) so that individual SAE "neurons" become **human-interpretable features of music** — e.g. one neuron might fire strongly on tracks with electric guitar, another on tracks with a certain rhythmic pattern, another on a genre-adjacent quality — even though MERT itself was never trained with any labeled concepts like these.

This is a mechanistic-interpretability project applied to audio, analogous to SAE interpretability work done on LLMs (e.g. Anthropic's "Towards Monosemanticity"), but here the base model is a **music/audio transformer** instead of a language model.

**Core idea:**
1. Run raw audio through frozen, pretrained MERT → get a hidden-state activation tensor per track (`[time_steps, 768]`).
2. Train a small autoencoder (encoder → sparse latent → decoder) to reconstruct that activation tensor.
3. Add a **sparsity penalty** on the latent so that, for any given input, only a few of the (many, overcomplete) latent neurons fire.
4. If it works, each latent neuron becomes selective for a specific, recognizable musical concept. You can verify this by taking a large pool of songs, computing latent activations for all of them, and looking at the **top-K songs that maximally activate a given neuron** — if they share an obvious trait (instrumentation, genre, mood, production style), the neuron is "interpretable."

**The reconstruction ↔ sparsity tradeoff:**
- Reconstruction loss (MSE) = "how accurately does the SAE preserve what MERT knows."
- Sparsity loss (KL-divergence to a target sparsity `rho`, or L1 on latent activations) = "how likely each neuron is to represent one clean, human-graspable concept rather than a smear of many things."
- `expansion_factor` (latent_size = 768 × factor) trades off capacity vs. interpretability — bigger, more overcomplete latents can decompose activations into more/finer concepts.

The eval tooling (`eval/`) exists to actually **look at** what the trained neurons encode: browse top-activating tracks per neuron, check whether neurons correlate with genre labels (a rough, human-labeled proxy for "does this neuron mean something"), and generate figures for write-ups/papers.

---

## High-level pipeline

```
┌─────────────────┐      ┌──────────────────────┐      ┌────────────────────┐      ┌───────────────────┐
│ data_download/   │ ---> │ train/ (or           │ ---> │ eval/               │ ---> │ figures/,           │
│ Get FMA audio,    │      │ sae_files/)          │      │ generate_latents,   │      │ example_neurons/    │
│ run through MERT, │      │ Train SAE(s) on      │      │ browse/analyze      │      │ (paper-ready        │
│ save datasets     │      │ MERT activations     │      │ neurons, genre eval │      │ artifacts)          │
└─────────────────┘      └──────────────────────┘      └────────────────────┘      └───────────────────┘
```

There are **two training modes**, reflected by duplicated/parallel scripts:
- **"Live" training** (`train/SAE_train_large.py`, root `SAE_train_large (1).py`): decodes raw FMA audio and runs it through MERT on-the-fly, per batch, during training. No large intermediate dataset needs to be stored; slower per-step (MERT forward pass every batch) but simpler to set up and scales to FMA-large. This is the **current, actively used** approach (`train/SAE_train_large.py`).
- **"Precomputed" training** (`sae_files/SAE_train.py`, `train/SAE_train.py`, `train/SAE_train.ipynb`): expects a dataset where MERT hidden states have *already* been computed and stored as a byte-serialized column (`MERT_output`), produced ahead of time by `data_download/download_FMA_large.py`. Training is then just SAE forward/backward, no MERT inference in the loop — much faster iteration once the precompute step is done. `sae_files/SAE_train.py` is a cleaner, config-driven (argparse+YAML) version of the same idea; `train/SAE_train.py` is close to identical (duplicate) and `train/SAE_train_large_old.py` / root `SAE_train_large (1).py` are earlier snapshots of the live-training script (safe to ignore/delete — kept only as history).

Both training modes always train **multiple SAE variants in parallel** (different `expansion_factors`, e.g. 4×/16×/32× of the 768-dim MERT space) within a single run, sharing the same MERT-derived batch, so you get several latent-size regimes to compare per run.

---

## Data flow / on-disk artifacts

Understanding this is essential because most eval scripts require several artifacts to already exist on disk, produced in order:

1. **FMA dataset** (HuggingFace, gated — needs `HF_TOKEN`): `benjamin-paine/free-music-archive-large` (or `-medium`). Raw audio (mp3 bytes) + metadata (title, artist, genres as int IDs, tags, url, etc.).
2. **MERT-precomputed dataset** (optional, for "precomputed" training path and most `eval/` scripts): a HuggingFace `datasets` object saved to disk via `save_to_disk`, with columns:
   - `MERT_output`: raw bytes of a `[window_size, 768]` float32 tensor (`.reshape(window_size, 768)` to recover) — the frozen MERT hidden state for that track.
   - metadata columns from FMA (`title`, `artist`, `genres` (list of int IDs), `tags`, `url`, `album_title`, etc.)
   - Produced by `data_download/download_FMA_large.py`.
3. **Dataset-with-audio**: the same dataset but with an added `audio_b64` column (base64-encoded MP3 bytes) so browser-based tools can play the underlying track. Produced by `data_download/merge_audio_into_mert.py` (matches by `url`).
4. **Train/val/test split**: every script that loads one of the datasets above applies the **same deterministic split** for consistency:
   ```python
   splits = dataset.train_test_split(test_size=0.2, seed=42)
   test_val = splits["test"].train_test_split(test_size=0.5, seed=42)
   # splits["train"]      -> 80% train
   # test_val["train"]    -> 10% val
   # test_val["test"]     -> 10% test  (this is what eval scripts analyze by default)
   ```
   Always use `seed=42` when re-deriving a split so it lines up with what a checkpoint was trained/evaluated on. Some scripts accept `--full` to skip splitting entirely (used for supplemental datasets that are entirely held-out).
5. **Trained SAE checkpoint(s)**: `.pt` files saved by the training scripts as `best_model_{expansion_factor}.pt`, containing `model_state`, `optim_state`, optional `scheduler_state`, and `val_loss`.
6. **Latent analysis artifacts** (from `eval/generate_latents.py`), all under one `--output_dir` (referred to as `latent_dir` everywhere else):
   - `latent_activations.pt`: `{"activations": [n_samples, latent_size] time-averaged latents, "latent_size", "expansion_factor", "window_size", "feature_dim", "stride", "latent_channels", "n_samples", "seed"}`
   - `top_k_cache.pt`: `{"top_indices": [latent_size, k], "top_values": [latent_size, k], "k"}` — for each neuron, the indices (into the test split) and activation values of its top-K activating samples.
   - `latent_full.pt`: full time-resolved latents `[n_samples, latent_channels, latent_size]` (code to save this is currently commented out in `generate_latents.py` — only generated if you re-enable it; it can be huge).

Everything downstream in `eval/` reads `latent_activations.pt` + `top_k_cache.pt` + a dataset (with or without audio, depending on whether the tool needs to play/analyze audio) and does NOT need the SAE model itself (except `generate_latents.py`, which needs the checkpoint to produce these caches in the first place).

---

## Environment / setup

- **Requirements**: `requirements.txt` lists `torch`, `torchaudio`, `transformers`, `datasets`, `librosa`, `wandb`, `tqdm`. Training scripts also import `python-dotenv`, `pyyaml`, and (for `eval/`) `matplotlib`, `numpy`.
- **Secrets**: place a `.env` file (loaded via `python-dotenv`) with:
  ```
  HF_TOKEN=<huggingface token, needed for the gated FMA dataset(s)>
  WANDB_TOKEN=<weights & biases API key, needed for training logging>
  ```
- **Compute**: training uses CUDA if available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`), falls back to CPU otherwise. Live-MERT training is GPU-memory-bound — batch sizes are kept small (8–32) because MERT itself runs every batch.
- **Experiment tracking**: all training scripts log to **Weights & Biases** (`wandb`). You must be logged in (`WANDB_TOKEN` in `.env`) and `wandb_project` in config controls which W&B project results land in.
- This appears to have been run on a SLURM/HPC cluster at some point — comments reference `orcd/scratch/...` paths (MIT Engaging/ORCD cluster scratch space) and "inside your SLURM job" in docstrings. Adjust paths for your own environment.

---

## W&B MCP server (agent access to Weights & Biases)

This project logs all training runs to Weights & Biases (see "Experiment tracking" above). A **W&B MCP server** is configured on this machine so an agent (e.g. Claude Code) can query runs/projects/entities directly instead of asking the user to check the W&B dashboard manually.

- **What it is**: a hosted remote MCP server at `https://mcp.withwandb.com/mcp`, added via Claude Code's HTTP transport with a `Authorization: Bearer <WANDB_API_KEY>` header.
- **How it was configured** (already done on this machine, user scope — applies to all projects, not just this one):
  ```powershell
  claude mcp add --transport http wandb https://mcp.withwandb.com/mcp --header "Authorization: Bearer <WANDB_API_KEY>" -s user
  ```
  On this machine the Claude Code binary isn't on PATH (the desktop app bundles its own copy); the full path used was:
  `C:\Users\Gamer\AppData\Roaming\Claude\claude-code\<version>\claude.exe`. If `claude` isn't recognized in PowerShell, locate the bundled binary the same way (check `Get-Process claude | Select Path`, or search `%APPDATA%\Claude\claude-code\`) and invoke it via its full path, or add that directory to PATH.
- **Config location**: stored under the `mcpServers` key for this machine's user scope in `~/.claude.json` (i.e. `C:\Users\Gamer\.claude.json`). The API key header is saved there in plaintext — if it's ever rotated, remove and re-add:
  ```powershell
  claude mcp remove wandb -s user
  claude mcp add --transport http wandb https://mcp.withwandb.com/mcp --header "Authorization: Bearer <NEW_KEY>" -s user
  ```
- **Verify it's working**: `claude mcp list` should show `wandb: https://mcp.withwandb.com/mcp (HTTP) - ✓ Connected`. A **new** Claude Code session/conversation is required after adding or changing it — the MCP tool list is loaded at session start, so an already-running session won't pick it up until restarted.
- **Using it**: once connected, tools for listing entities/projects/runs and querying run metrics/logs become available to the agent directly — ask things like "list my W&B entities" or "show recent runs in the `mert-sae-large` project" instead of manually checking the dashboard. The project's `wandb_project` config field (see `train/config.yaml`, `sae_files/SAE_train.py` `DEFAULT_CONFIG`) tells you which W&B project a given training run's results live in.
- **Note**: this MCP server's API key is separate from the `WANDB_TOKEN` env var in `.env` used by the training scripts' own `wandb.login()` calls — they can share the same key value, but are configured independently (one lives in `.env`, the other in `~/.claude.json` via `claude mcp add`).

---

## Directory-by-directory / file-by-file reference

### `data_download/` — acquire audio + precompute MERT activations

#### `download_FMA_large.py`
- **Purpose**: Download FMA-large from HuggingFace, run every track through frozen MERT, and save one unified on-disk dataset with MERT outputs + base64 audio + metadata. This is the main "build the precomputed dataset" script, with checkpointing so long-running jobs can be resumed if preempted.
- **Inputs (CLI args)**:
  - `--output_dir` (required): root dir for checkpoints + final dataset.
  - `--hf_token` (default: `$HF_TOKEN`)
  - `--fma_dataset` (default `benjamin-paine/free-music-archive-large`)
  - `--layer_idx` (default `-1`, which MERT hidden layer to extract)
  - `--window_size` (default `1125`, ≈30s at MERT's frame rate)
  - `--batch_size` (default `8`)
- **What it does**: loads MERT + FMA-large, decodes/resamples audio per-track, runs MERT forward pass batch-by-batch (frozen, `no_grad`), collects `MERT_output` (raw float32 bytes), `audio_b64` (base64 mp3), and selected metadata columns (`KEEP_COLS`), writes parquet checkpoints every `CHECKPOINT_EVERY=50` batches (resumable — counts existing checkpoint files to figure out where to resume), then merges all parquet checkpoints into one HF `Dataset`, **deduplicates by `url`**, and saves via `save_to_disk` to `{output_dir}/window{window_size}_layer{layer_idx}/dataset`. Deletes the checkpoint dir afterward.
- **Output**: an HF dataset on disk at `{output_dir}/window{window_size}_layer{layer_idx}/dataset` with columns `MERT_output`, `audio_b64`, plus `KEEP_COLS` metadata (title, url, artist, genres, tags, album info, licensing fields, etc.)
- **Run**:
  ```bash
  python data_download/download_FMA_large.py \
      --output_dir /scratch/joneba/fma_large_mert \
      --hf_token YOUR_TOKEN \
      --layer_idx -1 \
      --window_size 1125 \
      --batch_size 8
  ```

#### `download_fma_large_supplement.py`
- **Purpose**: Build a **held-out supplemental eval set** of 10,000 FMA-large tracks that are NOT already in FMA-medium (i.e. genuinely unseen data), used to sanity-check that neuron behavior generalizes beyond the original training/test split. Splits the 10k into two 5k halves: one matching the existing test split's genre distribution, one balanced across the top-20 most common genres (250 tracks/genre).
- **Inputs (CLI args)**:
  - `--mert_dataset` (required): path to the existing MERT-precomputed dataset (used to know which URLs are already "medium" and to read the test split's genre distribution).
  - `--output_dir` (required)
  - `--hf_token`, `--window_size` (1125), `--feature_dim` (768), `--layer_idx` (-1), `--batch_size` (4), `--seed` (42), `--target_total` (10000), `--checkpoint_every` (100)
- **What it does**: streams FMA-large (`streaming=True`, no full download), skips tracks whose `url` is already in the medium set, greedily fills the two target genre distributions, then runs MERT on each collected track (single-item forward pass, not batched), checkpointing to parquet periodically (resumable), concatenates + dedupes by URL, saves to `{output_dir}/dataset`.
- **Output**: an HF dataset compatible with `generate_latents.py` at `{output_dir}/dataset` (has `MERT_output` + metadata, but **no `audio_b64`** column — run `merge_audio_into_mert.py` afterward if you want playable audio in eval tools).
- **Run**:
  ```bash
  python data_download/download_fma_large_supplement.py \
      --mert_dataset orcd/scratch/window1125_layer-1/dataset \
      --output_dir orcd/scratch/window1125_layer-1/fma_large_supplement \
      --hf_token YOUR_TOKEN
  ```
  Then feed the output into `eval/generate_latents.py` (use `--full`, since this dataset has no train/test split semantics of its own — it's entirely held out).

#### `merge_audio_into_mert.py`
- **Purpose**: Add a playable-audio column to an already-precomputed MERT dataset by matching against FMA-medium (or another FMA split) by `url`.
- **Inputs (CLI args)**:
  - `--mert_dataset` (required): path to the saved MERT dataset (must have a `url` column).
  - `--output_dir` (required)
  - `--hf_token` (default `$HF_TOKEN`)
  - `--fma_dataset` (default `benjamin-paine/free-music-archive-medium`)
- **What it does**: loads the MERT dataset, loads the specified FMA split (audio not decoded, kept as raw bytes), builds a `url -> index` lookup over FMA, then for each MERT-dataset row look up matching audio bytes and base64-encode them into a new `audio_b64` column (order-preserving `add_column`), saves to `output_dir`, then re-loads and asserts the first 5 rows still match by `url` (sanity check row order wasn't shuffled).
- **Output**: copy of the input dataset with an added `audio_b64` column, saved via `save_to_disk` to `output_dir`.
- **Run**:
  ```bash
  python data_download/merge_audio_into_mert.py \
      --mert_dataset /scratch/joneba/mert_medium/window1125_layer-1/dataset \
      --output_dir   /scratch/joneba/mert_medium/window1125_layer-1/dataset_with_audio \
      --hf_token     YOUR_TOKEN
  ```

#### `download_FMA_medium.ipynb`
- Notebook variant of the download workflow, for FMA-medium exploration (not read in full detail here; treat as a manual/exploratory counterpart to `download_FMA_large.py`).

---

### `train/` and `sae_files/` — SAE training

All training scripts share the **same model definitions** (`Encoder`/`Decoder`/`RELUEncoder`/`SAE`/`RELU_SAE`) — see "Model architecture" section below for the shared explanation instead of repeating it per file.

#### `train/SAE_train_large.py` ⭐ (current/primary training script — "live" MERT computation)
- **Purpose**: Train SAE(s) directly against **live-computed** MERT hidden states — audio is decoded and pushed through frozen MERT inside the training loop, every batch. No precompute step required; works straight off the raw FMA-large HF dataset.
- **Config** (via `--config path/to/config.yaml`, falls back to `DEFAULT_CONFIG` in-file if omitted; YAML values override defaults): see `train/config.yaml` for a working example. Key fields:
  - `run_name`, `wandb_project`
  - `sae_type`: `"SAE"` or `"RELU_SAE"`
  - `expansion_factors`: list, e.g. `[4, 16, 32]` — trains one SAE per factor, in parallel, per batch.
  - `mert_model_name` (default `m-a-p/MERT-v1-95M`), `layer_idx` (default `-1`)
  - `fma_dataset` (default `benjamin-paine/free-music-archive-large`)
  - `window_size` (default `1125`, ~30s), `batch_size` (kept small, 8–32, since MERT runs live), `lr`, `sparsity_loss_weight`, `rho` (target sparsity for KL-based `SAE`), `epochs`, `cascading_lr` (bool — cosine-anneal LR per model if true)
  - `use_checkpoints` (bool), `checkpoint_dir`
  - `output_dir`: where `best_model_{factor}.pt` and checkpoints are written
  - `resume_from`: path to a `.pt` checkpoint to resume the *first* epoch from (only applies once)
- **Data pipeline**: `FMALiveDataset` decodes each item's raw mp3 bytes via `librosa.load` (with an `audioread` backend fallback), resamples to MERT's expected sample rate (cached per-samplerate `torchaudio.transforms.Resample`), returns `(audio_tensor, genres)`. `audio_collate_fn` truncates each clip to `max_samples=24000*30`, pads to the batch's max length, stacks. `DataLoader`s use `num_workers=4, pin_memory=True`.
- **Model construction**: after loading MERT + one sample batch to determine `input_dim = [batch, window_size, 768]`, builds `models = {str(x): SAEClass(input_dim, 768*x, lr=...) for x in expansion_factors}`.
- **Training loop** (`train()` function): for each batch — compute MERT hidden states with `no_grad`, then for every expansion-factor model: zero grad, forward, compute `sparsity_loss()` and MSE `loss_fn`, backward, step (+ scheduler step if `cascading_lr`). Logs to `wandb` + stdout every 80 batches. Runs **mid-epoch validation** every `val_every` batches (`val_every=700` in `main()`) by calling `test()`, which also saves `best_model_{k}.pt` whenever validation loss improves. Supports resuming from a checkpoint by fast-forwarding batches (`start_batch`).
- **Output**: per-expansion-factor best checkpoints `best_model_{expansion_factor}.pt` in `output_dir`, plus optional periodic full-state checkpoints (`{output_dir}/{checkpoint_dir}/epoch{e}_batch{b}.pt`) if `use_checkpoints: true` (checkpointing is currently invoked once per epoch, not per-batch — the per-batch checkpoint block inside `train()` is commented out/disabled via `if False:`). W&B logs the same metrics.
- **Run**:
  ```bash
  python train/SAE_train_large.py --config train/config.yaml
  ```

#### `train/config.yaml`
- The reference config for `SAE_train_large.py`. Notable current values: `sae_type: RELU_SAE`, `expansion_factors: [4, 16, 32]`, `lr: 1.0e-2`, `batch_size: 32`, `epochs: 3`, `rho: 0.05`, `sparsity_loss_weight: 0.5`, `use_checkpoints: True`, `output_dir: orcd/scratch/sae_train_large`. Edit this file (or copy it) to launch new runs — pass `--config <path>` on the CLI.

#### `train/SAE_train_large_old.py` and root `SAE_train_large (1).py`
- Earlier snapshots of the live-training script. Functionally almost identical to `train/SAE_train_large.py` but with small differences (older librosa fallback args like `res_type='kaiser_fast'`, no `num_workers`/`pin_memory` on DataLoaders, slightly different logging key formatting, per-batch checkpointing left enabled via `if False:`/commented differently, output-dir layout differs — nests under `{output_dir}/runs/{run_name}`). **These are stale duplicates** kept for history; prefer `train/SAE_train_large.py` for all new work. Safe to archive/delete once confirmed unneeded.

#### `train/SAE_train.py` and `sae_files/SAE_train.py` (precomputed-MERT training path)
- **Purpose**: Same SAE training as above, but assumes MERT activations were **already computed** and stored in a dataset's `MERT_output` byte column (via `data_download/download_FMA_large.py`). No MERT model is loaded at all — much faster since there's no per-batch transformer forward pass, only SAE forward/backward.
- **`sae_files/SAE_train.py`** is the "clean" config-driven version (`--config` YAML, `DEFAULT_CONFIG` dict, `load_config()` — same pattern as `SAE_train_large.py`). Config fields include `dataset_dir` (if `None`, derived as `orcd/scratch/window{window_size}_layer{layer_idx}`), plus the same SAE/training hyperparameters (`sae_type`, `expansion_factors`, `lr`, `batch_size` default `512` — much larger than the live-training batch size since there's no MERT compute cost — `sparsity_loss_weight`, `rho`, `epochs`, `cascading_lr`, `resume_from`, `wandb_project`).
- **`train/SAE_train.py`** is functionally the same but with hardcoded config at the bottom of the file (`run_name = "RELU_test_1e-4_05"`, hardcoded `wandb.init(config={...})`) instead of CLI/YAML — i.e. edit-the-script-directly style rather than config-file style. Also missing mid-epoch validation (`test()` is only called once per epoch, after the full training loop) and missing `output_dir`/global step handling refinements present in `sae_files/SAE_train.py`. Treat `sae_files/SAE_train.py` as the more maintained version of this precomputed-path script; `train/SAE_train.py` is a duplicate.
- **Data pipeline**: `MERTPrecomputedDataset.__getitem__` does `torch.frombuffer(bytearray(item["MERT_output"]), dtype=torch.float32).reshape(window_size, feature_dim)` to reconstruct the tensor, returns `(tensor, genres)`. `precomputed_collate_fn` just stacks tensors and passes genres through as a list (no padding needed — all rows are already fixed `window_size`).
- **Run** (`sae_files` version):
  ```bash
  python sae_files/SAE_train.py --config path/to/config.yaml
  ```
  (No committed example config for this script currently exists in the repo — copy `train/config.yaml` and remove/adjust the live-training-only fields like `fma_dataset`/`mert_model_name`, add `dataset_dir` pointing at your precomputed dataset.)

#### `train/SAE_train.ipynb`
- Notebook version of the precomputed-path training script — same model/training code, for interactive/exploratory runs. Treat as equivalent to `sae_files/SAE_train.py` but interactive.

### Model architecture (shared across all training scripts and `eval/generate_latents.py`)

- **`Encoder`**: `Conv1d(768, 768, kernel_size=stride=25, stride=25)` collapses the time axis by a factor of `stride` (1125 timesteps → 45 "latent channels" at default settings), transposed appropriately, then `Linear(768, latent_size)`, then a nonlinearity:
  - Base `Encoder` (used by `SAE`): `sigmoid` — bounded [0,1], smooth, differentiable everywhere. Comment in code: chosen because it "still encodes info from negative inputs" and "doesn't stress small diffs in extreme inputs."
  - `RELUEncoder` (used by `RELU_SAE`): `relu` — unbounded, hard-zero for negative pre-activations, the more standard modern-SAE choice (gives a true L0-style sparsity and additive, interpretable-by-construction latent directions).
- **`Decoder`**: mirror of the encoder — `Linear(latent_size, 768)` then `ConvTranspose1d(768, 768, kernel_size=stride, stride=stride)` to expand the time axis back to `window_size`. Output is unbounded linear (no final activation) since MERT hidden states themselves are unbounded/can be negative.
- **`SAE`** (base class): sigmoid-latent autoencoder.
  - `forward`: encode, record `data_rho` (mean latent activation per neuron, used for the sparsity target), record `L0_loss` (fraction of latents `>0.1`, a "soft L0" proxy since true L0 isn't differentiable), decode.
  - `sparsity_loss()`: **KL divergence** between target sparsity `rho` (a hyperparameter, e.g. 0.05 = "each neuron should be active ~5% of the time on average") and the empirical mean activation `rho_hat` per neuron — the classic sparse-autoencoder (Andrew Ng CS294A-style) sparsity penalty:
    `KL = rho*log(rho/rho_hat) + (1-rho)*log((1-rho)/(1-rho_hat))`, averaged or summed over neurons.
- **`RELU_SAE`** (subclass): ReLU-latent autoencoder, more like a standard dictionary-learning / Anthropic-style SAE.
  - `forward`: encode (ReLU), `L1 = encoded.sum(dim=1)` per-sample summed activation (time already collapsed to latent channels; this sums those channels), `L0_loss` = fraction of latents `>0` (true L0 here since ReLU hard-zeros).
  - `sparsity_loss()`: mean/sum of the per-sample L1 term — the standard L1-sparsity penalty used in most modern SAE interpretability work.
- **Total loss** in both cases: `mse_loss(reconstruction, original) + sparsity_loss_weight * sparsity_loss()`.
- **`SAE_CLASSES = {"SAE": SAE, "RELU_SAE": RELU_SAE}`** — the `sae_type` config string selects which class to instantiate.
- Input shape convention throughout: `input_dim = (batch, window_size, feature_dim)` e.g. `(B, 1125, 768)`. `latent_size = feature_dim * expansion_factor`.

---

### `eval/` — inspect, verify, and present what the SAE learned

#### `eval/generate_latents.py` ⭐ (must run first — produces the caches every other eval script depends on)
- **Purpose**: Run a trained SAE encoder over the **test split** of a precomputed-MERT dataset, save per-sample latent activations (averaged over time and, optionally, full time-resolved) plus a top-K cache identifying which samples maximally activate each neuron.
- **Inputs (CLI args)**:
  - `--dataset` (required): path to a `save_to_disk` dataset with `MERT_output` (bytes) column — same format produced by `download_FMA_large.py`.
  - `--checkpoint` (required): path to a `best_model_{k}.pt` file (must match `--expansion_factor`).
  - `--expansion_factor` (required, int): must correspond to the checkpoint's trained latent size (`latent_size = feature_dim * expansion_factor`).
  - `--output_dir` (required)
  - `--batch_size` (default 64), `--window_size` (default 1125), `--feature_dim` (default 768), `--top_k` (default 20), `--seed` (default 42, must match training split seed)
- **What it does**: loads the dataset, re-derives the standard 80/10/10 split (same `seed`) and takes the test slice, builds a plain `SAE` (note: uses the sigmoid `SAE` class regardless of whether the checkpoint was trained as `RELU_SAE` — see gotcha below), loads `model_state` from the checkpoint, does one forward pass per batch through **only the encoder** (`model.encoder(data)`), collecting both the time-averaged `[batch, latent_size]` and full `[batch, latent_channels, latent_size]` latents, then computes `topk` (per-neuron top-K sample indices/values) over the averaged activations.
- ⚠️ **Gotcha**: the encoder class hardcoded in this script is the plain sigmoid `Encoder`/`SAE`. If your checkpoint was trained as `RELU_SAE` (ReLU latent), loading its `state_dict` into a sigmoid `SAE` will load the weights fine (architecture is identical aside from the activation function) but **the activation function applied during `generate_latents.py`'s forward pass will be sigmoid, not ReLU** — meaning the latent statistics/top-K will not exactly match what was optimized during training. If you trained with `RELU_SAE`, consider patching this script to use `RELU_SAE`/`RELUEncoder` for a faithful reproduction before trusting the interpretability numbers precisely (this is a pre-existing script inconsistency, not user error).
- **Outputs** (written to `--output_dir`):
  - `latent_activations.pt`: `{"activations": Tensor[n, latent_size], "latent_size", "expansion_factor", "window_size", "feature_dim", "stride"=25, "latent_channels", "n_samples", "seed"}`
  - `top_k_cache.pt`: `{"top_indices": Tensor[latent_size, top_k], "top_values": Tensor[latent_size, top_k], "k"}`
  - `latent_full.pt`: full time-resolved latents — **currently not actually written** (the `torch.save`/print lines are commented out at the bottom of the script) despite being computed in memory; re-enable if you need it (large — check the printed size estimate first).
- Also prints summary stats (global mean/std, sparsity at 0.1/0.5 thresholds, dead-neuron count) and a sanity check comparing averaged-from-full vs. directly-averaged activations.
- **Run**:
  ```bash
  python eval/generate_latents.py \
      --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
      --checkpoint orcd/scratch/window1125_layer-1/runs/initial_test/best_model_8.pt \
      --expansion_factor 8 \
      --output_dir orcd/scratch/window1125_layer-1/latent_analysis \
      --top_k 20
  ```

#### `eval/neuron_explorer.py` ⭐ (primary interactive tool — "pick a neuron, listen to its top songs")
- **Purpose**: Local HTTP server + single-page dark-themed UI to browse every SAE neuron: overview grid of top-200 most-active neurons (mean/std/sparsity/max), and a per-neuron detail view with an activation histogram and the top-K activating tracks (title/artist/genre tags, playable inline audio, source link).
- **Inputs (CLI args)**:
  - `--dataset` (required): dataset **with `audio_b64`** column (i.e. output of `merge_audio_into_mert.py`), so tracks can be played back.
  - `--latent_dir` (required): dir containing `latent_activations.pt` + `top_k_cache.pt` from `generate_latents.py`.
  - `--port` (default 9999)
  - `--seed` (default 42)
  - `--full` (bool flag, default `False`): if true, use the entire dataset instead of re-deriving the test split (use when the dataset is already a held-out set, e.g. the FMA-large supplement).
- **What it does**: loads dataset (memory-mapped, `keep_in_memory=False` — safe for very large datasets), loads the two `.pt` caches, precomputes per-neuron stats (mean/std/sparsity/max/min) and all-neuron histograms **once at startup** (can take a while for large `latent_size` — logs progress every 5000 neurons), then serves a `http.server`/`socketserver` app with endpoints:
  - `GET /` — the HTML/JS single-page app
  - `GET /api/overview` — top-200-by-mean-activation neuron summary (JSON)
  - `GET /api/neuron?idx=N` — full detail for neuron N: stats, histogram, top-K tracks with metadata
  - `GET /api/audio?idx=N` — streams the base64-decoded mp3 for a given **sample index** (as an `audio/mpeg` response)
- **Output**: nothing persisted to disk — it's a live server. Browse in a browser; arrow keys / `O` key / neuron-number input navigate.
- **Run**:
  ```bash
  python eval/neuron_explorer.py \
      --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
      --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
      --port 9999
  ```
  Then (if running on a remote/cluster machine) SSH tunnel: `ssh -L 9999:localhost:9999 <host>`, and open `http://localhost:9999` locally.

#### `eval/export_neuron.py`
- **Purpose**: Export a single neuron's explorer page (stats, histogram, top-K tracks with embedded playable audio) as one **self-contained, portable HTML file** — no server needed, embeds audio as base64 data-URIs directly in the file. Good for sharing a specific interesting neuron with someone else or archiving a finding.
- **Inputs (CLI args)**: `--dataset` (required, needs `audio_b64`), `--latent_dir` (required), `--neuron` (required, int), `--output` (default `neuron_{n}.html`), `--seed` (default 42).
- **What it does**: same data loading/derivation as `neuron_explorer.py` but for a single neuron, and instead of serving JSON it renders a complete static HTML document (histogram drawn client-side via inline `<canvas>` JS, tracks rendered directly into the HTML with `<audio src="data:audio/mpeg;base64,...">`).
- **Output**: a single `.html` file (can be large — several MB depending on `top_k` × audio-clip sizes — printed at the end).
- **Run**:
  ```bash
  python eval/export_neuron.py \
      --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
      --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
      --neuron 9704 \
      --output neuron_9704.html
  ```
  (See `example_neurons/neuron_9704.html` for a previously exported example.)

#### `eval/neuron_genres.py`
- **Purpose**: Quick terminal-only inspection — print the top-K activating tracks for one neuron along with their genre tags, plus an aggregate genre-count summary. The lightest-weight way to eyeball "does this neuron mean something."
- **Inputs (CLI args)**: `--dataset` (required, genre metadata only — no audio needed), `--latent_dir` (required), `--neuron` (required, int), `--seed` (default 42), `--full` (flag — use whole dataset, e.g. for supplement sets).
- **What it does**: loads dataset + `top_k_cache.pt` only (no `latent_activations.pt` needed here), re-derives split unless `--full`, prints each of the neuron's top-K tracks with title/artist/activation/genre names (mapped via the in-file `GENRE_NAMES` dict, a hardcoded id→string table of FMA's ~163 genre taxonomy), then a genre-count summary line.
- **Output**: stdout only, no files.
- **Run**:
  ```bash
  python eval/neuron_genres.py \
      --dataset orcd/scratch/window1125_layer-1/dataset \
      --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
      --neuron 5773
  ```

#### `eval/neuron_analysis.py`
- **Purpose**: Generate a single **paper-ready PDF/PNG figure** for one neuron combining: (a) activation histogram with the top-K threshold marked, (b) genre breakdown (top-K vs. whole-dataset baseline), (c) mel-spectrograms of the top-10 activating tracks in a 2×5 grid, (d) a pairwise cosine-similarity matrix between those spectrograms (a proxy for "do the top tracks actually sound acoustically similar, not just share a genre label"). This is the deepest single-neuron analysis tool in the repo.
- **Inputs (CLI args)**: `--dataset` (required, needs `audio_b64` to compute spectrograms), `--latent_dir` (required), `--neuron` (required, int), `--output` (default `neuron_{n}_analysis.pdf`), `--seed` (default 42), `--full` (flag), `--dpi` (default 300).
- **What it does**: loads dataset/caches, collects top-K tracks + genre counts, decodes up to 10 top tracks' audio (`librosa`, 22050 Hz) and computes 128-mel spectrograms (`librosa.feature.melspectrogram` → `power_to_db`), computes flattened-and-normalized pairwise cosine similarity across those spectrograms, prints a similarity matrix + summary stats to stdout, then builds the 4-panel matplotlib figure (CVPR-style rcParams: serif fonts, small sizes, tight bbox) and saves both PDF and PNG.
- **Output**: `{output}.pdf` and matching `.png` (e.g. `figures/neuron_5773_analysis.pdf/.png`, `figures/neuron_11639_analysis.pdf/.png` are pre-existing examples in this repo), plus a detailed stdout report (genre counts, pairwise similarity matrix values).
- **Run**:
  ```bash
  python eval/neuron_analysis.py \
      --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
      --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
      --neuron 5773 \
      --output neuron_5773_analysis.pdf

  # For the FMA-large supplement (fully held-out data):
  python eval/neuron_analysis.py \
      --dataset orcd/scratch/window1125_layer-1/fma_large_supplement/dataset_with_audio \
      --latent_dir orcd/scratch/window1125_layer-1/latent_analysis_flat_1e-1_rho08_32 \
      --neuron 5773 \
      --output neuron_5773_analysis.pdf \
      --full
  ```

#### `eval/genre_correspondence.py`
- **Purpose**: The main **quantitative interpretability metric** across the *whole* latent space (not just one neuron) — computes "genre purity" per neuron (what fraction of a neuron's top-K tracks share its single most common genre tag) and compares the average purity across all active neurons against the dataset's baseline genre concentration. Also produces a genre×neuron heatmap for the most-active neurons. This is the evidence used to argue (or caveat) that neurons encode something beyond "just genre."
- **Inputs (CLI args)**: `--dataset` (required, genre metadata only), `--latent_dir` (required), `--output` (default `genre_correspondence.pdf`), `--seed` (default 42), `--full` (flag), `--top_neurons` (default 50, how many most-active neurons shown on the heatmap x-axis), `--top_genres` (default 20, most-common genres shown on the heatmap y-axis), `--dpi` (default 300).
- **What it does**:
  1. Builds a per-sample genre list and overall genre-frequency counter over the test set (or full set).
  2. Computes "dataset baseline purity" = the max single-genre share of all genre tags (i.e. what purity you'd get by chance / from genre skew alone).
  3. For the top-N most-active neurons (by mean activation): builds a genre-fraction matrix over their top-K tracks, and a per-neuron "purity" = fraction of top-K genre tags belonging to that neuron's single most common genre.
  4. Repeats the purity computation over **all active neurons** (mean activation > 0.01, i.e. excluding dead neurons) for a more representative average.
  5. Prints a full terminal summary including a ready-to-paste "for the paper" sentence comparing overall average purity vs. dataset baseline, plus purity distribution stats (mean/median/std/min/max).
  6. Builds a 3-panel figure: (a) genre-fraction heatmap (top neurons × top genres) with a baseline-genre-frequency bar chart alongside, (b) histogram of purity across all active neurons with baseline and mean lines marked.
- **Output**: `{output}.pdf` and `.png` (e.g. `figures/genre_correspondence.pdf`/`.png`, `figures/genre_correspondence_32.png` present as examples), plus the detailed stdout report/paper-sentence.
- **Interpretation note baked into the script**: if overall purity is not meaningfully above baseline (< 1.5× baseline), it prints "Neurons are NOT dominated by single genres... indicating cross-genre features" — i.e. **low genre purity is treated as evidence that neurons encode something more specific/orthogonal than genre**, not necessarily a failure. This matters for how you should read the metric.
- **Run**:
  ```bash
  python eval/genre_correspondence.py \
      --dataset orcd/scratch/window1125_layer-1/dataset \
      --latent_dir orcd/scratch/window1125_layer-1/latent_analysis \
      --output genre_correspondence.pdf

  # For FMA large supplement:
  python eval/genre_correspondence.py \
      --dataset orcd/scratch/window1125_layer-1/fma_large_supplement/dataset \
      --latent_dir orcd/scratch/window1125_layer-1/latent_analysis_flat_1e-1_rho08_32 \
      --output genre_correspondence.pdf \
      --full
  ```

#### `eval/genre_histograms.py`
- **Purpose**: Pure **dataset description** tool (no SAE/latents involved) — shows genre-tag frequency distributions across the train/val/test splits, useful for a paper's dataset section and for sanity-checking that the split didn't skew genre balance.
- **Inputs (CLI args)**: `--dataset` (required, metadata only), `--output` (default `genre_distributions.pdf`), `--top_n` (default 25), `--seed` (default 42), `--dpi` (default 300).
- **What it does**: loads dataset, derives the standard 80/10/10 split, counts genre-tag occurrences per split (`GENRE_NAMES` id→name mapping again hardcoded in-file — note this file's mapping has slightly different IDs/ordering after index 51 compared to `genre_correspondence.py`'s copy, because an extra "Europe" entry is inserted — **the two `GENRE_NAMES` dicts in this repo are NOT byte-identical**, see gotcha below), determines the top-N genres by total count across all splits, then plots a 3-row stacked bar chart (one row per split) in CVPR paper style, saves PDF+PNG, and prints a full per-genre per-split count table to stdout.
- ⚠️ **Gotcha**: `GENRE_NAMES` is copy-pasted independently into `genre_correspondence.py`, `genre_histograms.py`, `neuron_analysis.py`, and `neuron_genres.py`. Comparing them: `genre_histograms.py`'s dict has an extra `52: "Europe"` entry that shifts all subsequent IDs by one relative to the other three files' dicts (which agree with each other, ending at `162: "hiphop"` vs. `genre_histograms.py`'s `163: "hiphop"`). **If you ever refactor this into a shared module, verify which numbering is actually correct against the FMA/dataset's real genre-ID scheme before trusting labels from either version** — right now `genre_histograms.py` may be mislabeling genres by one ID for ids ≥ 52 relative to the others (or vice versa; this needs to be checked against the source-of-truth FMA genre taxonomy, not just internal consistency).
- **Output**: `{output}.pdf`/`.png` (e.g. `figures/genre_distributions.pdf`/`.png` present as examples), stdout table.
- **Run**:
  ```bash
  python eval/genre_histograms.py \
      --dataset orcd/scratch/window1125_layer-1/dataset \
      --output genre_distributions.pdf
  ```

#### `eval/visualize_mert_row.py`
- **Purpose**: Debugging/QA tool — browse **raw MERT-precomputed dataset rows** directly (not SAE latents) to sanity-check the precompute pipeline: view a track's metadata, play its audio, and see summary stats of its raw MERT hidden-state tensor (min/max/mean/std, and an L2-norm-per-timestep line chart, useful for spotting e.g. padding artifacts or dead/silent regions).
- **Inputs (CLI args)**: `--dataset` (required — a `MERT_output` (+ ideally `audio_b64`) dataset), `--port` (default 9999), `--window_size` (default 1125), `--feature_dim` (default 768).
- **What it does**: loads the dataset, serves an `http.server` app with:
  - `GET /` — dark-themed single-page app
  - `GET /api/random` — random row
  - `GET /api/row?idx=N` — specific row (decodes `MERT_output` bytes → tensor, computes summary stats + per-timestep L2 norm, plus all other non-audio/non-MERT columns as generic metadata display)
  - `GET /api/audio?idx=N` — streams audio if `audio_b64` present
  - Keyboard shortcut `R` = load random row.
- **Output**: nothing persisted — live server only, for interactive inspection.
- **Run**:
  ```bash
  python eval/visualize_mert_row.py \
      --dataset orcd/scratch/window1125_layer-1/dataset_with_audio \
      --port 9999
  ```
  SSH tunnel if remote, open `http://localhost:9999`.

---

### `figures/` and `example_neurons/`
- Pre-generated output artifacts from the `eval/` scripts above, kept in the repo as examples/results:
  - `figures/genre_correspondence.pdf`, `.png`, `genre_correspondence_32.png` — from `genre_correspondence.py`
  - `figures/genre_distributions.pdf`, `.png` — from `genre_histograms.py`
  - `figures/neuron_11639_analysis.pdf`/`.png`, `figures/neuron_20_analysis.pdf`/`.png`, `figures/neuron_5773_analysis.pdf`/`.png` — from `neuron_analysis.py`, for specific neurons of interest (20, 5773, 11639)
  - `example_neurons/neuron_9704.html` — from `export_neuron.py`, a portable single-neuron report for neuron 9704.
- These aren't inputs to anything — they're the project's "results so far." Useful for seeing what a finished analysis looks like before running your own.

---

## Typical end-to-end workflows

### A. Train a new SAE from scratch (live MERT, simplest path)
1. Ensure `.env` has `HF_TOKEN` and `WANDB_TOKEN`.
2. Edit/copy `train/config.yaml` (run name, expansion factors, `sae_type`, hyperparameters, `output_dir`).
3. `python train/SAE_train_large.py --config train/config.yaml`
4. Best checkpoints land at `{output_dir}/best_model_{expansion_factor}.pt`; monitor progress in the configured W&B project.

### B. Train faster by precomputing MERT first
1. `python data_download/download_FMA_large.py --output_dir <dir> --hf_token ...` → produces `{dir}/window{w}_layer{l}/dataset`.
2. Copy/adapt a config with `dataset_dir` pointing at that dataset.
3. `python sae_files/SAE_train.py --config <your_config>.yaml`

### C. Analyze a trained checkpoint's neurons
1. `python eval/generate_latents.py --dataset <precomputed dataset> --checkpoint <best_model_N.pt> --expansion_factor N --output_dir <latent_dir>` (⚠️ verify the sigmoid-vs-ReLU encoder gotcha above if the checkpoint is `RELU_SAE`).
2. (Optional, for audio playback in eval tools) `python data_download/merge_audio_into_mert.py --mert_dataset <dataset> --output_dir <dataset_with_audio>`.
3. Quantitative check: `python eval/genre_correspondence.py --dataset <dataset> --latent_dir <latent_dir> --output genre_correspondence.pdf`
4. Browse interactively: `python eval/neuron_explorer.py --dataset <dataset_with_audio> --latent_dir <latent_dir>`
5. Deep-dive a specific interesting neuron found in step 4: `python eval/neuron_analysis.py --dataset <dataset_with_audio> --latent_dir <latent_dir> --neuron <N> --output neuron_<N>_analysis.pdf`, and/or export it for sharing: `python eval/export_neuron.py --dataset <dataset_with_audio> --latent_dir <latent_dir> --neuron <N>`.

### D. Validate generalization on held-out data
1. `python data_download/download_fma_large_supplement.py --mert_dataset <original dataset> --output_dir <supplement_dir> --hf_token ...`
2. `python eval/generate_latents.py --dataset <supplement_dir>/dataset --checkpoint <best_model_N.pt> --expansion_factor N --output_dir <supplement_latent_dir>` (no split needed here since it's fully held out — but note `generate_latents.py` doesn't currently expose a `--full` flag like the other eval scripts do; it always applies the 80/10/10 split internally, so for a fully-held-out supplement dataset the "test set" it evaluates on will actually only be ~10% of the supplement unless the script is patched to support `--full` as well).
3. Run any of the `eval/` genre/neuron scripts with `--full` and pointing at the supplement dataset + supplement latent dir.

---

## Known inconsistencies / things to check before trusting results (for future agents)

1. **`generate_latents.py` always instantiates the plain sigmoid `SAE`/`Encoder`**, never `RELU_SAE`/`RELUEncoder`, regardless of `--expansion_factor` or how the checkpoint was actually trained. If your checkpoints were trained with `sae_type: RELU_SAE` (the current default in `train/config.yaml`), the weights will load but the activation function during latent generation will silently mismatch training. Check this before trusting `latent_activations.pt`/`top_k_cache.pt` for any RELU_SAE checkpoint, or patch the script to look up `SAE_CLASSES` the way the training scripts do.
2. **`GENRE_NAMES` dicts are duplicated in 4 files and not identical** (`genre_histograms.py` has an extra "Europe" entry at id 52 that the others lack, shifting everything after it by one ID). Don't assume genre labels printed by one script exactly match another's for IDs ≥ 52 without checking against ground truth.
3. **`generate_latents.py` has no `--full` flag** unlike most other `eval/` scripts, so it always re-derives an 80/10/10 split even on datasets meant to be fully held-out (e.g. the FMA-large supplement) — meaning it evaluates on ~10% of the supplement, not all of it, by default. Confirm this is intended or patch it to match the other scripts' `--full` behavior.
4. **`train/SAE_train.py` and `sae_files/SAE_train.py`** are near-duplicates (config-file-driven vs. hardcoded-at-bottom-of-script); **`train/SAE_train_large_old.py`** and root **`SAE_train_large (1).py`** are stale earlier snapshots of `train/SAE_train_large.py`. When making training-code changes, edit `train/SAE_train_large.py` (live path, primary) and `sae_files/SAE_train.py` (precomputed path) — the other files are legacy/duplicated and likely safe to remove once confirmed unused, but don't delete without checking whether any in-flight run still references them.
5. **`latent_full.pt` is computed but not saved** in `generate_latents.py` (the save call is commented out) — if you need full time-resolved latents downstream, re-enable that block (and budget disk space using the size estimate the script prints beforehand — it can be enormous for large expansion factors).
6. The **`window_size`/`stride`=25 convention** (1125 timesteps → 45 latent channels) is hardcoded (`STRIDE = 25` in `eval/generate_latents.py`, `stride=25` passed explicitly in training scripts) — if you ever train with a different `window_size` or stride, make sure every downstream script (`generate_latents.py`'s `STRIDE`, all the `Encoder`/`Decoder` instantiations) is updated consistently.
