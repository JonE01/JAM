import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv
load_dotenv()
import os
import io
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_TOKEN")
import wandb

from torch.utils.data import Dataset as TorchDataset, DataLoader

import librosa
import torchaudio.transforms as T
import time
import argparse
import yaml
import datasets
from datasets import load_dataset, Audio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "run_name": "RELU_test_1e-4_05",
    "sae_type": "RELU_SAE",           # "SAE" or "RELU_SAE"
    "expansion_factors": [4, 16, 32],
    "layer_idx": -1,
    "window_size": 1125,
    "rho": 0.05,
    "lr": 1e-4,
    "batch_size": 8,                  # kept small for live MERT computation
    "sparsity_loss_weight": 0.05,
    "epochs": 50,
    "cascading_lr": False,
    "resume_from": None,
    "wandb_project": "mert-sae",
    "mert_model_name": "m-a-p/MERT-v1-95M",
    "fma_dataset": "benjamin-paine/free-music-archive-large",
    "use_checkpoints": False,
    "checkpoint_dir": "checkpoints",
    "output_dir": "runs",             # where to save model checkpoints / best models
}


def load_config():
    parser = argparse.ArgumentParser(description="Train SAE on MERT outputs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.config is not None:
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        if yaml_cfg:
            cfg.update(yaml_cfg)

    return cfg

# ═══════════════════════════════════════════════════════════════════════
# MERT wrapper (frozen, eval-only)
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
# Live FMA dataset / collate (decodes audio on the fly)
# ═══════════════════════════════════════════════════════════════════════

def decode_audio(item):
    audio_bytes = item["audio"]["bytes"]
    try:
        array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception:
        # Fallback for MP3s that soundfile can't open
        array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True,
                             backend='audioread')
    item["array"] = array
    item["sample_rate"] = sr
    return item


def audio_collate_fn(batch, max_samples=24000*30):
    """Collate raw audio tensors: filter Nones, truncate, pad, stack."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.tensor([]), []

    audios, genres = zip(*batch)
    # Truncate any clip longer than max_samples
    audios = [a[:max_samples] for a in audios]
    max_len = max(a.shape[0] for a in audios)
    padded = torch.stack([
        F.pad(a, (0, max_len - a.shape[0]))
        for a in audios
    ])
    return padded, list(genres)


class FMALiveDataset(TorchDataset):
    """Loads raw FMA audio and resamples to MERT's expected sample rate."""
    def __init__(self, hf_dataset, processor, sample_rate=44100):
        self.dataset = hf_dataset
        self.sample_rate = sample_rate
        self.resample_rate = processor.sampling_rate
        self._resampler_cache = {}

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
                if sample_rate not in self._resampler_cache:
                    self._resampler_cache[sample_rate] = T.Resample(sample_rate, self.resample_rate)
                resampler = self._resampler_cache[sample_rate]
                audio_array = resampler(torch.tensor(audio).float())
            else:
                audio_array = torch.tensor(audio).float()
        except Exception as e:
            print(f"Failed clip {idx}: {e}")
            return None
        return audio_array, item["genres"]

# ═══════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
  def __init__(self, input_dim, latent_size,stride):
    super().__init__() #Sets up important functions from the nn.module class
    # self.input_dimension_tuple = input_dimension_tuple
    self.stride = stride
    self.input_dim = input_dim
    self.features = input_dim[2]
    self.window_size = input_dim[1]
    self.latent_size = latent_size
    self.fully_connected1 = nn.Linear(self.features,latent_size)
    self.conv_1d = nn.Conv1d(self.features,self.features,kernel_size=self.stride,stride=self.stride)

  def forward(self, input):
    #I picked sigmoid bc it
    # Still encodes info from negative inputs
    # Doesn't stress small diffs in extreme inputs
    transposed = input.transpose(1,2)
    conv = self.conv_1d(transposed)
    conv = conv.transpose(1,2)
    fc = self.fully_connected1(conv) #[batch, time_steps/stride, latent_size]
    return torch.sigmoid(fc)

class Decoder(nn.Module):
  def __init__(self, output_dim, latent_size,stride):
    super().__init__()
    self.stride = stride
    # self.input_dimension_tuple = input_dimension_tuple
    self.output_dim = output_dim
    self.features = output_dim[2]
    self.window_size = output_dim[1]
    self.latent_size = latent_size
    self.fully_connected1 = nn.Linear(latent_size,self.features)
    self.inv_conv1d = nn.ConvTranspose1d(self.features,self.features,kernel_size=self.stride,stride=self.stride)


  def forward(self,input):
    #Can't do sigmoid again cause we need negative outputs
    #Can't do tanh cause outputs aren't bounded
    #Linear output allows for unbounded, negative outputs
    lin_output = self.fully_connected1(input) #[batch, time_steps/stride, feature_size]
    lin_output = lin_output.transpose(1,2) #[batch, feature_size, time_steps/stride]
    inv_conv = self.inv_conv1d(lin_output) #[batch, feature_size, time_steps]
    inv_conv = inv_conv.transpose(1,2) #[batch, time_steps, feature_size]
    return inv_conv

class RELUEncoder(Encoder):
    def __init__(self, input_dim, latent_size, stride):
        super().__init__(input_dim, latent_size, stride)

    def forward(self, input):
        #I picked sigmoid bc it
        # Still encodes info from negative inputs
        # Doesn't stress small diffs in extreme inputs
        transposed = input.transpose(1,2)
        conv = self.conv_1d(transposed)
        conv = conv.transpose(1,2)
        fc = self.fully_connected1(conv) #[batch, time_steps/stride, latent_size]
        return torch.relu(fc)

class SAE(nn.Module):
  def __init__(self, input_dim, latent_size, loss_fn=F.mse_loss, lr=1e-4, l2=0., rho=.05):
    super().__init__()
    self.latent_size = latent_size
    self.input_dim = input_dim
    self.encoder = Encoder(input_dim, latent_size, 25)
    self.decoder = Decoder(input_dim, latent_size, 25)
    self.loss_fn = loss_fn
    self.sparsity_loss_val = None
    self.L0_loss = None
    self._loss = None
    self.rho = rho
    self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

  def forward(self,input):
    #This stacks the rows horizontally, which matches to stacking the 768 features for each second side by side
    encoded = self.encoder(input)
    self.data_rho = encoded.mean(dim=(0,1))
    self.L0_loss = (encoded > 0.1).float().mean()
    decoded = self.decoder(encoded)
    return decoded

  def sparsity_loss(self, average = True):
    rho_hat = self.data_rho.clamp(1e-7, 1 - 1e-7)
    dkl = self.rho * torch.log(self.rho / rho_hat) + (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat))
    if average:
      self.sparsity_loss_val = dkl.mean()
    else:
      self.sparsity_loss_val = dkl.sum()

    return self.sparsity_loss_val

  def l0_loss(self):
    return self.L0_loss

class RELU_SAE(SAE):
    def __init__(self, input_dim, latent_size, loss_fn=F.mse_loss, lr=1e-4, l2=0., rho=.05):
        super().__init__(input_dim, latent_size, loss_fn, lr, l2)
        self.encoder = RELUEncoder(input_dim, latent_size, stride=25)
      
    def forward(self,input):
        #This stacks the rows horizontally, which matches to stacking the 768 features for each second side by side
        encoded = self.encoder(input)
        self.L1 = encoded.sum(dim=1).float()
        self.L0_loss = (encoded > 0).float().mean()
        decoded = self.decoder(encoded)
        return decoded
    def sparsity_loss(self, average = True):
        if average:
            self.sparsity_loss_val = self.L1.mean()
        else:
            self.sparsity_loss_val = self.L1.sum()
        return self.sparsity_loss_val

# ═══════════════════════════════════════════════════════════════════════
# SAE class lookup
# ═══════════════════════════════════════════════════════════════════════

SAE_CLASSES = {
    "SAE": SAE,
    "RELU_SAE": RELU_SAE,
}

# ═══════════════════════════════════════════════════════════════════════
# Training / testing (with live MERT computation)
# ═══════════════════════════════════════════════════════════════════════

def train(epoch, models, train_loader, sparsity_loss_weight, model_folder_path, mert,
          log=None, resume_from=None, global_step=0,
          val_loader=None, best_losses=None, val_log=None, val_every=350):
    """
    Args:
        epoch          : current epoch number
        models         : dict of {name: SAE}
        train_loader   : DataLoader yielding (raw_audio, genres)
        sparsity_loss_weight : weight for the sparsity loss
        model_folder_path : where to save outputs
        mert           : MERT_model wrapper (frozen, used to compute hidden states)
        log            : dict of {name: []} to accumulate (loss, sparsity_loss) tuples
        resume_from    : path to a checkpoint file to resume from, or None
        global_step    : monotonically increasing step counter for wandb
        val_loader     : DataLoader for validation (run every val_every batches)
        best_losses    : dict tracking best val loss per model
        val_log        : dict of {name: []} for val metrics
        val_every      : run validation every N batches

    Returns:
        global_step    : updated step counter
    """
    # model_checkpoints = {}
    # checkpoints_path = model_folder_path + "/" + checkpoint_dir
    # os.makedirs(checkpoints_path, exist_ok=True)
    total_batches = 0
    # ── resume from checkpoint ────────────────────────────────────────────────
    start_batch = 0
    if resume_from:
        ckpt = torch.load(resume_from)
        for k, model in models.items():
            if k in ckpt["models"]:
                model.load_state_dict(ckpt["models"][k]["model_state"])
                model.optim.load_state_dict(ckpt["models"][k]["optim_state"])
                if "scheduler_state" in ckpt["models"][k] and ckpt["models"][k]["scheduler_state"] is not None and wandb.config.cascading_lr == True:
                    model.scheduler.load_state_dict(ckpt["models"][k]["scheduler_state"])
        start_batch = ckpt.get("batch_idx", 0)
        print(f"Resumed from {resume_from} at batch {start_batch}")

    train_size = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else "?"
    orig_start_time = time.time()
    for batch_idx, (audio_batch, genre) in enumerate(train_loader):
        if batch_idx < start_batch:
            continue  # fast-forward past already-trained batches
        if audio_batch.numel() == 0:
            continue  # skip empty batches from failed audio decoding

        # ── compute MERT hidden states (frozen, no grad) ──────────────────
        with torch.no_grad():
            data = mert.get_hidden_states(audio_batch)
        # data is now [batch, window_size, 768] on device

        for model_name, model in models.items():
            model.optim.zero_grad()

            output = model(data)

            sparsity_l  = model.sparsity_loss(average=True)
            mse_loss = model.loss_fn(output, data)
            model._loss = mse_loss
            loss   = mse_loss + sparsity_loss_weight*sparsity_l

            loss.backward()
            
            model.optim.step()
            if wandb.config.cascading_lr == True:
                model.scheduler.step()

        # ── logging every n batches ─────────────────────────────────────────
        if batch_idx % 80 == 0:
            log_dict = {"epoch": epoch, "batch": batch_idx}
            for k, m in models.items():
                if type(m) == SAE:
                    sparsity_loss_type = "KL Divergence Loss"
                elif type(m) == RELU_SAE:
                    sparsity_loss_type = "L1 Loss"
                else:
                    sparsity_loss_type = "Sparsity Loss"
                log_dict[f"E={k}/Train MSE Loss"]     = m._loss.item()
                log_dict[f"E={k}/Train {sparsity_loss_type}"] = m.sparsity_loss_val.item()
                log_dict[f"E={k}/Train L0 Loss"]  = m.l0_loss().item()
            wandb.log(log_dict, step=global_step)
            global_step += 1

            line = f"Train Epoch: {epoch} [batch {batch_idx}] Time {time.time()-orig_start_time}\t"
            for k, m in models.items():
                if type(m) == SAE:
                    slt = "KL Divergence Loss"
                elif type(m) == RELU_SAE:
                    slt = "L1 Loss"
                else:
                    slt = "Sparsity Loss"
                line += f"{k} MSE Loss: {m._loss.item():.6f} {slt}: {m.sparsity_loss_val.item()}  "
            print(line)
            orig_start_time = time.time()

        # ── mid-epoch validation every val_every batches ──────────────────
        if val_loader is not None and batch_idx > 0 and batch_idx % val_every == 0:
            print(f"\n── Mid-epoch validation at batch {batch_idx} ──")
            for m_val in models.values():
                m_val.eval()
            global_step = test(models, val_loader, model_folder_path,
                               best_losses, mert, log=val_log, global_step=global_step)
            for m_val in models.values():
                m_val.train()
        #abt 30 mins of work
        # if batch_idx % 35 == 0 and batch_idx > 0:
        # if False:
        #     # ── checkpoint ───────────────────────────────────────────────────
        #     ckpt_path =  checkpoints_path + f"/epoch{epoch}_batch{batch_idx}.pt"
        #     ckpt_data = {
        #         "epoch":     epoch,
        #         "batch_idx": batch_idx,
        #         "models": {
        #             k: {
        #                 "model_state": m.state_dict(),
        #                 "optim_state": m.optim.state_dict(),
        #                 "scheduler_state": m.scheduler.state_dict() if hasattr(m, "scheduler") and wandb.config.cascading_lr == True else None,
        #             }
        #             for k, m in models.items()
        #         },
        #     }
        #     torch.save(ckpt_data, ckpt_path)
        #     wandb.save(ckpt_path)  # syncs the file to the wandb run
        total_batches += 1
    # ── end-of-epoch summary ──────────────────────────────────────────────────
    if log is not None:
        for k, m in models.items():
            log[k].append((m._loss.item(), m.sparsity_loss_val.item()))

    return global_step


def test(models, loader, model_folder_path, best_losses, mert, log=None, global_step=0):
    total_loss = {k: 0. for k in models}
    total_sparsity  = {k: 0. for k in models}
    total_l0 = {k: 0. for k in models}
    n_batches  = 0

    with torch.no_grad():
        for audio_batch, _ in loader:
            if audio_batch.numel() == 0:
                continue
            # ── compute MERT hidden states ────────────────────────────────
            data = mert.get_hidden_states(audio_batch)

            n_batches += 1
            for k, m in models.items():
                out = m(data)
                total_loss[k] += m.loss_fn(out, data).item()
                total_sparsity[k]  += m.sparsity_loss(average=True).item()
                total_l0[k] += m.l0_loss().item()

    log_dict = {}
    for k, m in models.items():
        if type(m) == SAE:
            sparsity_loss_type = "KL Divergence Loss"
        elif type(m) == RELU_SAE:
            sparsity_loss_type = "L1 Loss"
        else:
            sparsity_loss_type = "Sparsity Loss"
        avg_loss = total_loss[k] / n_batches
        avg_sparsity  = total_sparsity[k] / n_batches
        avg_l0 = total_l0[k] / n_batches
        print(f"{k}:  loss: {avg_loss:.6f}    {sparsity_loss_type}: {avg_sparsity:.6f}    l0_loss: {avg_l0:.4f}")
        log_dict[f"E={k}/Validation MSE Loss"]     = avg_loss
        # log_dict[f"E={k}/Validation {sparsity_loss_type}"] = avg_sparsity
        # log_dict[f"E={k}/Validation L0 Loss"]  = avg_l0
        if log is not None:
            log[k].append((avg_loss, avg_sparsity))

        if avg_loss < best_losses[k]:
            best_losses[k] = avg_loss
            best_path = os.path.join(model_folder_path, f"best_model_{k}.pt")
            torch.save({
                "model_state": m.state_dict(),
                "optim_state": m.optim.state_dict(),
                "scheduler_state": m.scheduler.state_dict() if hasattr(m, "scheduler") and wandb.config.cascading_lr == True else None,
                "val_loss": avg_loss,
            }, best_path)
            print(f"  New best for {k}: {avg_loss:.6f} -> {best_path}")

    wandb.log(log_dict, step=global_step)
    global_step += 1
    return global_step

def checkpoint(epoch, model_folder_path, checkpoint_dir, batch_idx, models):
    checkpoints_path = model_folder_path + "/" + checkpoint_dir
    os.makedirs(checkpoints_path, exist_ok=True)
    ckpt_file_path =  checkpoints_path + f"/epoch{epoch}_batch{batch_idx}.pt"
    ckpt_data = {
        "epoch":     epoch,
        "batch_idx": batch_idx,
        "models": {
            k: {
                "model_state": m.state_dict(),
                "optim_state": m.optim.state_dict(),
                "scheduler_state": m.scheduler.state_dict() if hasattr(m, "scheduler") and wandb.config.cascading_lr == True else None,
            }
            for k, m in models.items()
        },
    }
    torch.save(ckpt_data, ckpt_file_path)
    wandb.save(ckpt_file_path)  # syncs the file to the wandb run
# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config()

    run_name = cfg["run_name"]

    # Separate wandb config keys from non-wandb keys
    wandb_config = {
        "layer_idx":             cfg["layer_idx"],
        "window_size":           cfg["window_size"],
        "rho":                   cfg["rho"],
        "lr":                    cfg["lr"],
        "batch_size":            cfg["batch_size"],
        "sparsity_loss_weight":  cfg["sparsity_loss_weight"],
        "epochs":                cfg["epochs"],
        "cascading_lr":          cfg["cascading_lr"],
    }

    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=cfg["wandb_project"],
        name=run_name,
        config=wandb_config,
    )

    # ── 1. Load MERT model (frozen) ───────────────────────────────────────
    print(f"Loading MERT model: {cfg['mert_model_name']}...")
    mert_backbone = AutoModel.from_pretrained(
        cfg["mert_model_name"], trust_remote_code=True
    ).to(device).eval()
    # Freeze all MERT parameters — we only train the SAE
    for param in mert_backbone.parameters():
        param.requires_grad = False
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        cfg["mert_model_name"], trust_remote_code=True
    )
    mert = MERT_model(
        mert_backbone, processor,
        window_size=wandb.config.window_size,
        layer_idx=wandb.config.layer_idx,
    )
    print(f"  MERT loaded on {device}")

    # ── 2. Load FMA dataset ───────────────────────────────────────────────
    print(f"Loading FMA dataset: {cfg['fma_dataset']}...")
    fma_dataset = load_dataset(
        cfg["fma_dataset"],
        split="train",
        token=HF_TOKEN,
        revision="main",
    ).cast_column("audio", Audio(decode=False))
    print(f"  {len(fma_dataset)} tracks loaded")

    # ── 3. Split into train / val / test ──────────────────────────────────
    splits = fma_dataset.train_test_split(test_size=0.2, seed=42)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=42)

    train_ds = FMALiveDataset(splits["train"], processor)
    val_ds   = FMALiveDataset(test_val["train"], processor)
    test_ds  = FMALiveDataset(test_val["test"], processor)

    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size,
                              shuffle=True, collate_fn=audio_collate_fn,num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=wandb.config.batch_size,
                              collate_fn=audio_collate_fn,num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=wandb.config.batch_size,
                              collate_fn=audio_collate_fn,num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ── 4. Determine SAE input shape via a sample MERT forward pass ───────
    sample_audio, _ = next(iter(train_loader))
    with torch.no_grad():
        sample_hidden = mert.get_hidden_states(sample_audio)
    print(f"MERT output shape: {sample_hidden.shape}")

    window_size = wandb.config.window_size
    feature_dim = sample_hidden.shape[-1]   # 768
    input_dim   = sample_hidden.shape       # [batch, window_size, 768]

    print(f"SAE input_dim      : {input_dim}  ({window_size} steps × {feature_dim} features)")

    # ── 5. Build SAE models from config ───────────────────────────────────
    SAEClass = SAE_CLASSES[cfg["sae_type"]]
    expansion_factors = cfg["expansion_factors"]
    models = {
        str(x): SAEClass(input_dim, feature_dim * x, lr=wandb.config.lr).to(device)
        for x in expansion_factors
    }
    train_log = {k: [] for k in models}
    test_log  = {k: [] for k in models}
    best_losses = {k: float("inf") for k in models}
    if wandb.config.cascading_lr == True:
        for k, m in models.items():
            m.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                m.optim, T_max=wandb.config.epochs * len(train_loader)
            )

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    RESUME_FROM = cfg["resume_from"]

    # ── 6. Train ──────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(1, wandb.config.epochs+1):
        for k, m in models.items():
            m.train()
        if bool(cfg["use_checkpoints"]) and epoch > 1:
            model_folder_path = output_dir
            checkpoint(epoch,model_folder_path, cfg["checkpoint_dir"],0,models)

        global_step = train(epoch, models, train_loader, wandb.config.sparsity_loss_weight,
              output_dir, mert, log=train_log, resume_from=RESUME_FROM,
              global_step=global_step,
              val_loader=val_loader, best_losses=best_losses,
              val_log=test_log, val_every=700)
        RESUME_FROM = None  # only resume first epoch
        


if __name__ == "__main__":
    main()
