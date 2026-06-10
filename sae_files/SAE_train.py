from xml.parsers.expat import model

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv
load_dotenv()
import os
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_TOKEN")
import wandb

from torch.utils.data import Dataset as TorchDataset, DataLoader


import time
import argparse
import yaml
from datasets import load_from_disk

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
    "batch_size": 512,
    "sparsity_loss_weight": 0.05,
    "epochs": 50,
    "cascading_lr": False,
    "dataset_dir": None,              # if None, derived from window_size/layer_idx
    "resume_from": None,
    "wandb_project": "mert-sae",
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
# Dataset / collate
# ═══════════════════════════════════════════════════════════════════════

def precomputed_collate_fn(batch):
    tensors, genres = zip(*batch)
    return torch.stack(tensors), list(genres)

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
        return tensor, item["genres"]

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
            self.sparsity_loss_val = self.L1.mean(dim=0)
        else:
            self.sparsity_loss_val = self.L1.sum(dim=0)
        return self.sparsity_loss_val

# ═══════════════════════════════════════════════════════════════════════
# SAE class lookup
# ═══════════════════════════════════════════════════════════════════════

SAE_CLASSES = {
    "SAE": SAE,
    "RELU_SAE": RELU_SAE,
}

# ═══════════════════════════════════════════════════════════════════════
# Training / testing
# ═══════════════════════════════════════════════════════════════════════

def train(epoch, models, train_loader, sparsity_loss_weight, model_folder_path, log=None,
          checkpoint_dir="checkpoints", resume_from=None):
    """
    Args:
        epoch          : current epoch number
        models         : dict of {name: SAE}
        train_loader   : DataLoader yielding (hidden_states, labels)
        sparsity_loss_weight : weight for the sparsity loss
        log            : dict of {name: []} to accumulate (loss, sparsity_loss) tuples
        checkpoint_dir : where to save .pt checkpoints
        resume_from    : path to a checkpoint file to resume from, or None
    """
    model_checkpoints = {}
    checkpoints_path = model_folder_path + "/" + checkpoint_dir
    os.makedirs(checkpoints_path, exist_ok=True)
    total_batches = 0
    # ── resume from checkpoint ────────────────────────────────────────────────
    start_batch = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from)
        for k, model in models.items():
            if k in ckpt["models"]:
                model.load_state_dict(ckpt["models"][k]["model_state"])
                model.optim.load_state_dict(ckpt["models"][k]["optim_state"])
                if "scheduler_state" in ckpt["models"][k] and ckpt["models"][k]["scheduler_state"] is not None and wandb.config.cascading_lr == True:
                    model.scheduler.load_state_dict(ckpt["models"][k]["scheduler_state"])
        start_batch = ckpt.get("batch_idx", 0)  # ← add this back
        print(f"Resumed from {resume_from} at batch {start_batch}")

    train_size = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else "?"
    orig_start_time = time.time()
    for batch_idx, (data, genre) in enumerate(train_loader):
        if batch_idx < start_batch:
            continue  # fast-forward past already-trained batches
        # start_time = time.time()
        data = data.to(device)
        # model_batch_times = {}
        for model_name, model in models.items():
            # model_start_time = time.time()
            
            model.optim.zero_grad()

            # forward_start = time.time()
            output = model(data)
            # output_time = time.time() - forward_start

            sparsity_l  = model.sparsity_loss(average=True)
            mse_loss = model.loss_fn(output, data)
            model._loss = mse_loss
            loss   = mse_loss + sparsity_loss_weight*sparsity_l

            # backprop_start = time.time()
            loss.backward()
            # back_prop_time = time.time() - backprop_start
            
            # model_optim_start = time.time()
            model.optim.step()
            if wandb.config.cascading_lr == True:
                model.scheduler.step()
            # model_optim_time = time.time() - model_optim_start
            # model_batch_times[model_name] = time.time() - model_start_time

        # time_per_batch = time.time() - start_time
        # print(f"Batch: {batch_idx} Time {time_per_batch:.2f} Forward {output_time:.2f} Backprop {back_prop_time:.2f} Optimization {model_optim_time:.2f}")
        # model_timing_str = ""
        # for model_name, t in model_batch_times.items():
            # model_timing_str += f"{model_name}: {t:.2f} "
        # print(model_timing_str)
        # ── logging every n batches ─────────────────────────────────────────
        if type(model) == SAE:
            sparsity_loss_type = "kl_loss"
        elif type(model) == RELU_SAE:
            sparsity_loss_type = "L1_loss"
        else:
            sparsity_loss_type = "sparsity_loss"
        if batch_idx % 5 == 0:
            log_dict = {"epoch": epoch, "batch": batch_idx}
            for k, m in models.items():
                log_dict[f"{k}/loss"]     = m._loss.item()
                log_dict[f"{k}/{sparsity_loss_type}"] = m.sparsity_loss_val.item()
                log_dict[f"{k}/l0_loss"]  = m.l0_loss()
            wandb.log(log_dict)

            line = f"Train Epoch: {epoch} [batch {batch_idx}] Time {time.time()-orig_start_time}\t"
            line += "  ".join([f"{k} loss: {m._loss.item():.6f} {sparsity_loss_type}: {m.sparsity_loss_val.item()}" for k, m in models.items()])
            print(line)
            orig_start_time = time.time()
        #abt 30 mins of work
        # if batch_idx % 35 == 0 and batch_idx > 0:
        if False:
            # ── checkpoint ───────────────────────────────────────────────────
            ckpt_path =  checkpoints_path + f"/epoch{epoch}_batch{batch_idx}.pt"
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
            torch.save(ckpt_data, ckpt_path)
            wandb.save(ckpt_path)  # syncs the file to the wandb run
        total_batches += 1
    # ── end-of-epoch summary ──────────────────────────────────────────────────
    if log is not None:
        for k, m in models.items():
            log[k].append((m._loss.item(), m.sparsity_loss_val.item()))


def test(models, loader, model_folder_path, best_losses, log=None):
    total_loss = {k: 0. for k in models}
    total_sparsity  = {k: 0. for k in models}
    total_l0 = {k: 0. for k in models}
    n_batches  = 0

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            n_batches += 1
            for k, m in models.items():
                out = m(data)
                total_loss[k] += m.loss_fn(out, data).item()
                total_sparsity[k]  += m.sparsity_loss(average=True).item()
                total_l0[k] += m.l0_loss().item()

    log_dict = {}
    for k, m in models.items():
        if type(m) == SAE:
            sparsity_loss_type = "kl_loss"
        elif type(m) == RELU_SAE:
            sparsity_loss_type = "L1_loss"
        else:
            sparsity_loss_type = "sparsity_loss"
        avg_loss = total_loss[k] / n_batches
        avg_sparsity  = total_sparsity[k] / n_batches
        avg_l0 = total_l0[k] / n_batches
        print(f"{k}:  loss: {avg_loss:.6f}    {sparsity_loss_type}: {avg_sparsity:.6f}    l0_loss: {avg_l0:.4f}")
        log_dict[f"{k}/val_loss"]     = avg_loss
        log_dict[f"{k}/val_{sparsity_loss_type}"] = avg_sparsity
        log_dict[f"{k}/val_l0_loss"]  = avg_l0
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

    wandb.log(log_dict)


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

    output_dir = cfg["dataset_dir"]
    if output_dir is None:
        output_dir = f"orcd/scratch/window{wandb.config.window_size}_layer{wandb.config.layer_idx}"
    dataset = load_from_disk(os.path.join(output_dir, "dataset"))

    splits = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=42)

    train_ds = MERTPrecomputedDataset(splits["train"], wandb.config.window_size)
    val_ds   = MERTPrecomputedDataset(test_val["train"], wandb.config.window_size)
    test_ds  = MERTPrecomputedDataset(test_val["test"], wandb.config.window_size)

    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=precomputed_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=wandb.config.batch_size, collate_fn=precomputed_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=wandb.config.batch_size, collate_fn=precomputed_collate_fn)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # grab one example from the FMA training stream to inspect shapes
    sample_audio, sample_genre = next(iter(train_loader))
    single = sample_audio
    print(f"Single sample shape: {single.shape}")

    window_size = wandb.config.window_size
    feature_dim = single.shape[-1]
    input_dim   = single.shape

    print(f"layer output shape : {single.shape}")
    print(f"SAE input_dim      : {input_dim}  ({window_size} steps × {feature_dim} features)")

    # Build models from config
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
            m.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(m.optim, T_max=wandb.config.epochs * len(train_loader))

    data_dir = output_dir
    runs_dir = data_dir + '/runs'
    os.makedirs(runs_dir, exist_ok=True)

    RESUME_FROM = cfg["resume_from"]

    model_folder_path = runs_dir + "/" + run_name
    os.makedirs(model_folder_path, exist_ok=True)

    for epoch in range(1, wandb.config.epochs+1):
        for m in models.values():
            m.train()
        train(epoch, models, train_loader, wandb.config.sparsity_loss_weight, model_folder_path, log=train_log, resume_from=RESUME_FROM, checkpoint_dir="model_checkpoints")
        RESUME_FROM = None  # only resume first epoch

        for m in models.values():
            m.eval()
        test(models, val_loader, model_folder_path, best_losses, log=test_log)


if __name__ == "__main__":
    main()
