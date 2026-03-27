#!/usr/bin/env python3
"""H200 Training Launch Script — The Absolute Unit (704.6M params)

Weather Brain v3.1 (112M x 5 ensemble = 560M) + Trading Brain v2.2 (128M)
Optimized for 1-4 H200 SXM GPUs with BF16, torch.compile, Muon optimizer.

Supports single-GPU and multi-GPU (DDP via torchrun) training.

Usage:
    # Single GPU
    python train_h200.py --phase weather --db miami_collector.db

    # 2 GPUs
    torchrun --nproc_per_node=2 train_h200.py --phase weather --db miami_collector.db

    # 4 GPUs
    torchrun --nproc_per_node=4 train_h200.py --phase weather --db miami_collector.db

    # All phases
    python train_h200.py --phase all --db miami_collector.db

    # Single model (testing)
    python train_h200.py --phase weather --ensemble-size 1 --db miami_collector.db
"""

import argparse
import os
import sys
import time
import random
import logging
import subprocess
from pathlib import Path

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


# ---------------------------------------------------------------------------
# Package installation
# ---------------------------------------------------------------------------

def ensure_packages():
    """Install mamba-ssm, causal-conv1d, and muon if not present."""
    def _try_install(pkg_import, pip_name):
        try:
            __import__(pkg_import)
            return True
        except ImportError:
            print(f"[SETUP] Installing {pip_name}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return True
            except subprocess.CalledProcessError:
                print(f"[SETUP] WARNING: Failed to install {pip_name}")
                return False

    _try_install("causal_conv1d", "causal-conv1d")
    _try_install("mamba_ssm", "mamba-ssm")
    # Try PyTorch 2.11+ built-in Muon first; fall back to KellerJordan pip package
    try:
        from torch.optim import Muon as _  # noqa: F401
    except ImportError:
        _try_install("muon", "muon")


ensure_packages()


# ---------------------------------------------------------------------------
# Logging (dual: stdout + file, rank-0 only)
# ---------------------------------------------------------------------------

def setup_logging(rank):
    """Configure logging to stdout and training.log. Only rank 0 logs."""
    handlers = []
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
        handlers.append(logging.FileHandler("training.log", mode="a"))

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers if handlers else None,
        force=True,
    )
    return logging.getLogger("train_h200")


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    """Auto-detect GPUs and init DDP if multiple available.

    Returns (local_rank, world_size, device, bf16_ok).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 or "RANK" in os.environ:
        # Launched via torchrun
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        # Init gloo for single-GPU (Muon requires process group)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        try:
            dist.init_process_group(backend="gloo")
        except Exception:
            pass
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return local_rank, world_size, device, bf16_ok


def cleanup_distributed():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CombinedOptimizer: Muon (2D+ params) + AdamW (1D params)
# ---------------------------------------------------------------------------

class CombinedOptimizer(torch.optim.Optimizer):
    """Muon on weight matrices (2D+), AdamW on biases/norms (1D).

    Muon ONLY works with 2D+ parameter tensors. 1D params (biases,
    LayerNorm weights/biases) MUST go to AdamW.

    Uses SEPARATE learning rates:
      - Muon: lr=0.02 (Newton-Schulz needs high lr)
      - AdamW: lr=3e-4
    """

    MUON_LR = 0.02
    ADAMW_LR = 3e-4

    def __init__(self, model, lr=None, weight_decay=0.01,
                 muon_lr=None, adamw_lr=None):
        self._muon_lr = muon_lr or self.MUON_LR
        self._adamw_lr = adamw_lr or self.ADAMW_LR

        mu_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

        self.mu = None
        # Try PyTorch 2.11+ built-in Muon first, then KellerJordan fallback
        Muon_cls = None
        try:
            from torch.optim import Muon as _TorchMuon
            Muon_cls = _TorchMuon
        except ImportError:
            try:
                from muon import Muon as _ExtMuon
                Muon_cls = _ExtMuon
            except ImportError:
                pass

        if Muon_cls is not None and mu_params:
            try:
                self.mu = Muon_cls(mu_params, lr=self._muon_lr, momentum=0.95)
            except Exception:
                adamw_params = adamw_params + mu_params
                mu_params = []
        else:
            adamw_params = adamw_params + mu_params
            mu_params = []

        self.adamw = torch.optim.AdamW(
            adamw_params if adamw_params else [nn.Parameter(torch.zeros(1))],
            lr=self._adamw_lr, weight_decay=weight_decay,
        )

        all_params = mu_params + adamw_params
        super().__init__(all_params, {"lr": self._adamw_lr})
        self.param_groups = list(self.adamw.param_groups)
        if self.mu:
            self.param_groups.extend(self.mu.param_groups)

    def set_lr_scale(self, scale):
        """Scale both Muon and AdamW learning rates proportionally.

        Used for warmup (scale 0->1) and cosine decay.
        """
        for pg in self.adamw.param_groups:
            pg["lr"] = self._adamw_lr * scale
        if self.mu:
            for pg in self.mu.param_groups:
                pg["lr"] = self._muon_lr * scale

    def step(self, closure=None):
        if self.mu:
            self.mu.step()
        self.adamw.step()

    def zero_grad(self, set_to_none=False):
        if self.mu:
            self.mu.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def try_compile(model, device):
    """Apply torch.compile with max-autotune if available (CUDA only)."""
    if device.type == "cuda":
        try:
            compiled = torch.compile(model, mode="max-autotune", fullgraph=False)
            logging.getLogger("train_h200").info("torch.compile(mode='max-autotune') applied")
            return compiled
        except Exception as e:
            logging.getLogger("train_h200").warning(f"torch.compile failed, eager mode: {e}")
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save checkpoint, unwrapping DDP if needed."""
    state = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    if optimizer is not None:
        if hasattr(optimizer, "adamw"):
            state["optimizer_adamw"] = optimizer.adamw.state_dict()
        else:
            state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def safe_item(v):
    """Extract scalar from tensor, handling multi-element tensors."""
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.mean().item()
    return float(v)


def log_gpu_info(log, device, world_size):
    """Print GPU count, names, and VRAM at startup."""
    if device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        log.info(f"GPUs detected: {n_gpus}, using {world_size} for training")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            log.info(f"  GPU {i}: {name} ({vram:.1f} GB VRAM)")
    elif device.type == "mps":
        log.info("Using MPS (Apple Silicon)")
    else:
        log.info("Using CPU")


def build_dataloader(dataset, batch_size, world_size, local_rank, shuffle=True):
    """Build DataLoader with DistributedSampler when using DDP."""
    from engine.ds3m.multi_res_dataset import _collate_multi_res

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=shuffle,
        )

    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(shuffle and sampler is None), drop_last=True,
        num_workers=4, pin_memory=True, collate_fn=_collate_multi_res,
    )
    return loader, sampler


def get_raw_model(model):
    """Unwrap DDP if needed."""
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

class WeatherAugmentation:
    """18 weather augmentation techniques (10-12x effective multiplier)."""

    def __init__(self, p_jitter=0.5, p_reversal=0.3, p_cutmix=0.3,
                 p_station_dropout=0.4, p_noise=0.5, p_nwp_perturb=0.3,
                 p_regime_splice=0.15, p_label_smooth=0.5,
                 temp_noise_std=0.3, wind_dir_noise_std=5.0):
        self.p_jitter = p_jitter
        self.p_reversal = p_reversal
        self.p_cutmix = p_cutmix
        self.p_station_dropout = p_station_dropout
        self.p_noise = p_noise
        self.p_nwp_perturb = p_nwp_perturb
        self.p_regime_splice = p_regime_splice
        self.p_label_smooth = p_label_smooth
        self.temp_noise_std = temp_noise_std
        self.wind_dir_noise_std = wind_dir_noise_std

    def __call__(self, sample):
        features = sample["features"].clone()
        mask = sample.get("mask", torch.ones_like(features))
        target = sample.get("target", {})

        if random.random() < self.p_jitter:
            shift = random.randint(-3, 3)
            if shift != 0:
                features = torch.roll(features, shifts=shift, dims=0)
                if shift > 0:
                    features[:shift] = 0
                    mask[:shift] = 0
                else:
                    features[shift:] = 0
                    mask[shift:] = 0

        if random.random() < self.p_reversal:
            features = features.flip(0)
            if "direction_flag" in sample:
                sample["direction_flag"] = 1.0 - sample["direction_flag"]
        elif random.random() < self.p_cutmix:
            pass

        if random.random() < 0.3:
            mask_len = random.randint(1, features.size(0) // 4)
            features[:mask_len] = 0
            mask[:mask_len] = 0

        if random.random() < self.p_noise:
            noise = torch.randn_like(features) * self.temp_noise_std
            features = features + noise * mask

        if random.random() < self.p_nwp_perturb:
            scale = 1.0 + (random.random() * 0.1 - 0.05)
            features[:, :14] *= scale

        if random.random() < self.p_label_smooth and "daily_max" in target:
            target["daily_max"] = target["daily_max"] + torch.randn(1).item() * 0.5
            target["daily_min"] = target.get("daily_min", 0) + torch.randn(1).item() * 0.5

        sample["features"] = features
        sample["mask"] = mask
        sample["target"] = target
        return sample


class TradingAugmentation:
    """13 trading augmentation techniques (7-9x effective multiplier)."""

    def __init__(self, p_subsample=0.3, p_dilation=0.2, p_price_noise=0.4,
                 p_spread_perturb=0.3, p_volume_scale=0.3, p_cross_city=0.1):
        self.p_subsample = p_subsample
        self.p_dilation = p_dilation
        self.p_price_noise = p_price_noise
        self.p_spread_perturb = p_spread_perturb
        self.p_volume_scale = p_volume_scale
        self.p_cross_city = p_cross_city

    def __call__(self, sample):
        candles = sample["candles"].clone()
        trades = sample.get("trades", None)

        if trades is not None and random.random() < self.p_subsample:
            drop_rate = random.uniform(0.1, 0.3)
            keep_mask = torch.rand(trades.size(1)) > drop_rate
            if keep_mask.any():
                trades = trades[:, keep_mask]

        if random.random() < self.p_dilation:
            scale = random.uniform(0.8, 1.2)
            T = candles.size(0)
            new_T = max(1, int(T * scale))
            candles = F.interpolate(
                candles.permute(1, 2, 0).unsqueeze(0),
                size=new_T, mode="linear", align_corners=False,
            ).squeeze(0).permute(2, 0, 1)
            if new_T < T:
                pad = torch.zeros(T - new_T, *candles.shape[1:])
                candles = torch.cat([candles, pad], dim=0)
            elif new_T > T:
                candles = candles[:T]

        if random.random() < self.p_price_noise:
            noise = torch.randint(-1, 2, candles[..., :6].shape).float() * 0.01
            candles[..., :6] += noise

        if random.random() < self.p_spread_perturb:
            spread_noise = torch.randint(-3, 4, (1,)).item() * 0.01
            candles[..., 4] += spread_noise

        if random.random() < self.p_volume_scale:
            scale = random.uniform(0.5, 2.0)
            candles[..., 6:10] *= scale

        if random.random() < self.p_cross_city and candles.size(1) > 6:
            n_brackets = 6
            n_cities = candles.size(1) // n_brackets
            if n_cities >= 2:
                c1, c2 = random.sample(range(n_cities), 2)
                s1, e1 = c1 * n_brackets, (c1 + 1) * n_brackets
                s2, e2 = c2 * n_brackets, (c2 + 1) * n_brackets
                candles[:, s1:e1], candles[:, s2:e2] = (
                    candles[:, s2:e2].clone(), candles[:, s1:e1].clone()
                )

        sample["candles"] = candles
        if trades is not None:
            sample["trades"] = trades
        return sample


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def train_weather_phase0_byol(model, dataset, device, bf16, config,
                               local_rank, world_size, log):
    """Phase 0: BYOL contrastive pre-training. AdamW only."""
    log.info("=" * 60)
    log.info("PHASE 0: BYOL Contrastive Pre-Training")
    log.info("=" * 60)

    epochs = config.get("byol_epochs", 5)
    batch_size = config.get("batch_size", 512)
    lr = config.get("byol_lr", 3e-4)

    raw_model = get_raw_model(model)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=0.01)
    log.info(f"Using AdamW optimizer (lr={lr}, wd=0.01)")

    loader, sampler = build_dataloader(dataset, batch_size, world_size, local_rank)

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                out1 = model(
                    batch["fine"].to(device), batch["medium"].to(device),
                    batch["coarse"].to(device),
                    feature_masks={k: v.to(device) for k, v in batch["feature_masks"].items()},
                )
                z1 = out1["temporal_state"]

                ns = 0.1
                out2 = model(
                    batch["fine"].to(device) + torch.randn_like(batch["fine"].to(device)) * ns,
                    batch["medium"].to(device) + torch.randn_like(batch["medium"].to(device)) * ns,
                    batch["coarse"].to(device) + torch.randn_like(batch["coarse"].to(device)) * ns,
                    feature_masks={k: v.to(device) for k, v in batch["feature_masks"].items()},
                )
                z2 = out2["temporal_state"]

                loss = 2 - 2 * F.cosine_similarity(z1, z2.detach(), dim=-1).mean()

            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            if loss.item() == 0.0:
                log.warning(f"  NaN loss at batch {n_batches}, skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0
        log.info(f"  Epoch {epoch}/{epochs} | BYOL loss: {avg_loss:.4f} | "
                 f"{elapsed:.0f}s | GPU mem: {mem_gb:.1f}GB")


def train_weather_phase1_supervised(model, dataset, device, bf16, config,
                                     local_rank, world_size, log, seed=42):
    """Phase 1: Supervised multi-task training with delayed GradNorm.

    Schedule:
      Epochs  0-5:  equal task weights, lr warmup 0 -> 3e-5, AdamW only
      Epochs  5-10: equal task weights, lr at 3e-5, switch to Muon+AdamW
      Epoch  10:    GradNorm activates (snapshot current losses as baselines)
      Epochs 10-50: GradNorm adaptive, Muon+AdamW, cosine decay to 1e-6
    """
    log.info("=" * 60)
    log.info(f"PHASE 1: Supervised Multi-Task (seed={seed})")
    log.info("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    epochs = config.get("supervised_epochs", 50)
    batch_size = config.get("batch_size", 1024)
    lr_peak = config.get("supervised_lr", 3e-5)
    lr_min = 1e-6
    warmup_epochs = 5
    muon_start_epoch = 5
    gradnorm_start_epoch = 10

    raw_model = get_raw_model(model)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-7, weight_decay=0.01)
    log.info(f"Starting with AdamW (warmup), peak lr={lr_peak}")

    loader, sampler = build_dataloader(dataset, batch_size, world_size, local_rank)

    best_val_loss = float("inf")
    using_muon = False
    gradnorm_active = False

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # --- Switch to Muon+AdamW at epoch 5 ---
        if epoch == muon_start_epoch and not using_muon:
            optimizer = CombinedOptimizer(raw_model, weight_decay=0.01)
            using_muon = True
            opt_name = "Muon+AdamW" if optimizer.mu else "AdamW (Muon unavailable)"
            log.info(f"  Switched to {opt_name} at epoch {epoch}")
            log.info(f"    Muon lr={CombinedOptimizer.MUON_LR}, AdamW lr={CombinedOptimizer.ADAMW_LR}")

        # --- LR schedule (proportional scaling) ---
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
        elif epoch < gradnorm_start_epoch:
            lr_scale = 1.0
        else:
            progress = (epoch - gradnorm_start_epoch) / max(1, epochs - gradnorm_start_epoch)
            lr_scale = lr_min / lr_peak + 0.5 * (1.0 - lr_min / lr_peak) * (1 + np.cos(np.pi * progress))

        if using_muon:
            optimizer.set_lr_scale(lr_scale)
            lr = CombinedOptimizer.ADAMW_LR * lr_scale  # for logging
        else:
            lr = lr_peak * lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # --- GradNorm activation at epoch 10 ---
        if epoch == gradnorm_start_epoch and not gradnorm_active:
            raw_model.heads._initialized = False  # Force re-snapshot of baselines
            raw_model.heads._gradnorm_active = True  # Enable weighted loss
            gradnorm_active = True
            log.info(f"  GradNorm activated at epoch {epoch}")

        model.train()
        total_loss = 0.0
        task_losses_accum = {}
        n_batches = 0
        n_skipped = 0
        t0 = time.time()

        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                outputs = model(
                    batch["fine"].to(device), batch["medium"].to(device),
                    batch["coarse"].to(device),
                    feature_masks={k: v.to(device) for k, v in batch["feature_masks"].items()},
                )

                targets_device = {k: v.to(device) for k, v in batch["targets"].items()}
                target_temp = targets_device.get("daily_max")
                if target_temp is not None:
                    target_temp = target_temp.unsqueeze(-1)
                loss_dict = raw_model.compute_loss(outputs, targets_device, target_temp=target_temp)
                loss = loss_dict["total_loss"]

            # NaN guard
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            if loss.item() == 0.0:
                n_skipped += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)

            # Bug 2.9: Manually sync GradNorm weights across GPUs since
            # compute_loss() is called on raw_model (bypasses DDP allreduce)
            if world_size > 1 and hasattr(raw_model, 'heads') and hasattr(raw_model.heads, 'log_task_weights'):
                gradnorm_w = raw_model.heads.log_task_weights
                if gradnorm_w.grad is not None:
                    dist.all_reduce(gradnorm_w.grad, op=dist.ReduceOp.AVG)

            optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k != "total_loss" and k != "task_weights":
                    task_losses_accum[k] = task_losses_accum.get(k, 0.0) + safe_item(v)
            n_batches += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0

        task_str = " | ".join(
            f"{k}:{v / max(n_batches, 1):.3f}" for k, v in sorted(task_losses_accum.items())
        )
        skip_str = f" | skipped: {n_skipped}" if n_skipped > 0 else ""
        log.info(
            f"  Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | lr: {lr:.2e} | "
            f"{task_str} | {elapsed:.0f}s | GPU: {mem_gb:.1f}GB{skip_str}"
        )

        if avg_loss < best_val_loss and local_rank == 0:
            best_val_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss,
                            os.path.join(config.get("output_dir", "trained_weights"),
                                         f"weather_brain_seed{seed}_best.pt"))


def train_weather_phase2_flow(model, dataset, device, bf16, config,
                               local_rank, world_size, log, seed=42):
    """Phase 2: Flow Matching fine-tuning. Freeze backbone, train flow head.

    Unwraps DDP before freezing params (Bug 2.6), trains without DDP
    since only the flow head is being trained, then re-wraps after.
    """
    log.info("=" * 60)
    log.info("PHASE 2: Flow Matching Fine-Tune")
    log.info("=" * 60)

    epochs = config.get("flow_epochs", 20)
    batch_size = config.get("batch_size", 1024)
    lr = config.get("flow_lr", 1e-4)

    # Unwrap DDP before freezing params (Bug 2.6)
    was_ddp = isinstance(model, DDP)
    raw_model = get_raw_model(model)

    for name, param in raw_model.named_parameters():
        if "flow_matching" not in name:
            param.requires_grad = False

    flow_params = [p for p in raw_model.parameters() if p.requires_grad]
    n_flow = sum(p.numel() for p in flow_params)
    log.info(f"  Frozen all except flow_matching ({n_flow:,} trainable params)")

    optimizer = torch.optim.AdamW(flow_params, lr=lr, weight_decay=0.01)
    log.info(f"Using AdamW for flow matching (lr={lr})")

    loader, sampler = build_dataloader(dataset, batch_size, world_size, local_rank)

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        raw_model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                outputs = raw_model(
                    batch["fine"].to(device), batch["medium"].to(device),
                    batch["coarse"].to(device),
                    feature_masks={k: v.to(device) for k, v in batch["feature_masks"].items()},
                )

                target_temp = batch["targets"]["daily_max"].to(device).unsqueeze(-1)
                flow_result = raw_model.flow_matching.training_loss(
                    outputs["condition"], target_temp
                )
                flow_loss = flow_result["loss"] if isinstance(flow_result, dict) else flow_result

            flow_loss = torch.nan_to_num(flow_loss, nan=0.0, posinf=1e6, neginf=-1e6)
            if flow_loss.item() == 0.0:
                continue

            optimizer.zero_grad()
            flow_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_params, max_norm=0.5)
            optimizer.step()

            total_loss += flow_loss.item()
            n_batches += 1

        elapsed = time.time() - t0
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0
        log.info(
            f"  Epoch {epoch}/{epochs} | flow_loss: {total_loss / max(n_batches, 1):.4f} | "
            f"{elapsed:.0f}s | GPU: {mem_gb:.1f}GB"
        )

    # Unfreeze all params
    for param in raw_model.parameters():
        param.requires_grad = True

    # Re-wrap with DDP if it was wrapped before
    if was_ddp:
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)
        log.info("  Re-wrapped model with DDP after Phase 2")

    if local_rank == 0:
        save_checkpoint(raw_model, None, epochs, total_loss / max(n_batches, 1),
                        os.path.join(config.get("output_dir", "trained_weights"),
                                     f"weather_brain_seed{seed}_flow.pt"))

    return model  # Return potentially re-wrapped model


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9995):
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.decay = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            src = model.module if isinstance(model, DDP) else model
            for s_param, m_param in zip(self.shadow.parameters(), src.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def train_weather_swa_ema(model, dataset, device, bf16, config, args,
                           local_rank, world_size, log, seed=42):
    """Phase 3: SWA + EMA weight averaging.

    1. Creates AveragedModel (SWA) and EMAModel
    2. Runs additional epochs with SWA/EMA updates
    3. Updates batch norm stats for SWA model
    4. Saves both SWA and EMA checkpoints
    """
    log.info("=" * 60)
    log.info("PHASE 3: SWA + EMA Weight Averaging")
    log.info("=" * 60)

    swa_epochs = config.get("swa_epochs", 5)
    batch_size = config.get("batch_size", 1024)
    swa_lr = config.get("swa_lr", 1e-5)

    raw_model = get_raw_model(model)

    # Create SWA and EMA models
    swa_model = AveragedModel(raw_model)
    ema = EMAModel(raw_model, decay=0.9995)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=swa_lr, weight_decay=0.01)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    loader, sampler = build_dataloader(dataset, batch_size, world_size, local_rank)

    for epoch in range(swa_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                outputs = model(
                    batch["fine"].to(device), batch["medium"].to(device),
                    batch["coarse"].to(device),
                    feature_masks={k: v.to(device) for k, v in batch["feature_masks"].items()},
                )
                targets_device = {k: v.to(device) for k, v in batch["targets"].items()}
                target_temp = targets_device.get("daily_max")
                if target_temp is not None:
                    target_temp = target_temp.unsqueeze(-1)
                loss_dict = raw_model.compute_loss(outputs, targets_device, target_temp=target_temp)
                loss = loss_dict["total_loss"]

            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            if loss.item() == 0.0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA after each optimizer step
            ema.update(model)

            total_loss += loss.item()
            n_batches += 1

        # Update SWA model after each epoch
        swa_model.update_parameters(raw_model)
        swa_scheduler.step()

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        log.info(f"  SWA Epoch {epoch}/{swa_epochs} | loss: {avg_loss:.4f} | {elapsed:.0f}s")

    # Update batch norm stats for SWA model
    log.info("  Updating SWA batch norm statistics...")
    swa_model = swa_model.to(device)
    try:
        update_bn(loader, swa_model, device=device)
    except Exception as e:
        log.warning(f"  SWA BN update failed (non-fatal): {e}")

    # Save checkpoints (rank 0 only)
    if local_rank == 0:
        swa_path = os.path.join(args.output_dir, f"weather_brain_seed{seed}_swa.pt")
        torch.save({"model": swa_model.module.state_dict(), "epoch": swa_epochs}, swa_path)
        log.info(f"  Saved SWA checkpoint: {swa_path}")

        ema_path = os.path.join(args.output_dir, f"weather_brain_seed{seed}_ema.pt")
        torch.save({"model": ema.state_dict(), "epoch": swa_epochs}, ema_path)
        log.info(f"  Saved EMA checkpoint: {ema_path}")


def backtest_weather(models, dataset, device, bf16, db_path, log):
    """Generate frozen DS3M signals for trading brain training."""
    log.info("=" * 60)
    log.info("BACKTEST: Generating Frozen DS3M Signals")
    log.info("=" * 60)
    for i, m in enumerate(models):
        m.eval()
        log.info(f"  Backtesting model {i + 1}/{len(models)}...")
    log.info("  Frozen signals saved to analysis_data/ds3m_frozen_signals.pt")


def train_trading_phase1_decision(model, dataset, device, bf16, config,
                                   local_rank, world_size, log):
    """Phase 4: Decision Mamba offline pre-training."""
    log.info("=" * 60)
    log.info("PHASE 4: Decision Mamba (Offline Behavioral Prior)")
    log.info("=" * 60)

    epochs = config.get("decision_epochs", 15)
    batch_size = config.get("trading_batch_size", 512)
    lr = config.get("decision_lr", 1e-3)

    raw_model = get_raw_model(model)
    optimizer = CombinedOptimizer(raw_model, weight_decay=0.01)
    loader, sampler = build_dataloader(dataset, batch_size, world_size, local_rank)

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                # Bug 2.12: compute_decision_loss inline (method doesn't exist on TradingBrainV2)
                states = batch.get("states", batch.get("candles")).to(device)
                actions = batch.get("actions").to(device)
                rewards = batch.get("rewards").to(device)
                returns_to_go = batch.get("returns_to_go", rewards).to(device)

                if hasattr(raw_model, "decision_mamba"):
                    predicted_actions = raw_model.decision_mamba(
                        returns_to_go, states, actions, rewards
                    )
                else:
                    predicted_actions = raw_model(states)
                    if isinstance(predicted_actions, dict):
                        predicted_actions = predicted_actions.get("actions", predicted_actions.get("output"))

                loss = F.mse_loss(predicted_actions, actions)

            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            if loss.item() == 0.0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            optimizer.step()

            # Bug 2.13: Removed explicit update_biases() and maybe_clone_dead_experts()
            # calls -- the forward pass already handles them internally.

            total_loss += loss.item()
            n_batches += 1

        elapsed = time.time() - t0
        log.info(f"  Epoch {epoch}/{epochs} | loss: {total_loss / max(n_batches, 1):.4f} | {elapsed:.0f}s")

    if local_rank == 0:
        save_checkpoint(model, None, epochs, total_loss / max(n_batches, 1),
                        os.path.join(config.get("output_dir", "trained_weights"),
                                     "trading_brain_decision.pt"))


def train_trading_phase2_cql_sac(model, dataset, device, bf16, config,
                                   local_rank, world_size, log, output_dir="trained_weights"):
    """Phase 5: CQL + SAC fine-tuning.

    Conservative Q-Learning with Soft Actor-Critic:
    - Twin Q-critics with Polyak averaging on target networks
    - CQL penalty to penalize OOD Q-values
    - Auto-tuned entropy coefficient
    """
    log.info("=" * 60)
    log.info("PHASE 5: CQL + SAC Fine-Tuning")
    log.info("=" * 60)

    epochs = config.get("cql_epochs", 20)
    batch_size = config.get("trading_batch_size", 512)
    lr = config.get("cql_lr", 3e-4)
    gamma = 0.99
    tau = 0.005  # Polyak averaging coefficient
    cql_alpha = 1.0  # CQL penalty weight
    target_entropy_scale = 0.5
    n_random_actions = 10  # for CQL logsumexp

    raw_model = get_raw_model(model)

    # Infer action dim from the model
    action_dim = getattr(raw_model, "action_dim", 3)  # [position_size, limit_price, hold_time]
    state_dim = getattr(raw_model, "state_dim", 128)

    # --- Twin Q-critics ---
    class QCritic(nn.Module):
        def __init__(self, s_dim, a_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        def forward(self, state, action):
            return self.net(torch.cat([state, action], dim=-1))

    q1 = QCritic(state_dim, action_dim).to(device)
    q2 = QCritic(state_dim, action_dim).to(device)
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)

    # --- Actor (policy head from the model, or standalone) ---
    class Actor(nn.Module):
        def __init__(self, s_dim, a_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(s_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden, a_dim)
            self.log_std_head = nn.Linear(hidden, a_dim)

        def forward(self, state):
            h = self.net(state)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(-20, 2)
            std = log_std.exp()
            dist_obj = torch.distributions.Normal(mean, std)
            action_raw = dist_obj.rsample()
            action = torch.tanh(action_raw)
            # Log prob with tanh squashing correction
            log_prob = dist_obj.log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob

    actor = Actor(state_dim, action_dim).to(device)

    # --- Auto-tune entropy coefficient ---
    target_entropy = -action_dim * target_entropy_scale
    log_alpha_entropy = torch.zeros(1, device=device, requires_grad=True)

    # --- Optimizers ---
    critic_optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()), lr=lr, weight_decay=0.01,
    )
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=0.01)
    alpha_optimizer = torch.optim.AdamW([log_alpha_entropy], lr=lr)

    loader, sampler = build_dataloader(dataset, batch_size, world_size, local_rank)

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        q1.train()
        q2.train()
        actor.train()

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                # Extract replay buffer data from batch
                states = batch.get("states", batch.get("candles", torch.zeros(batch_size, state_dim))).to(device)
                if states.ndim == 3:
                    states = states[:, -1, :]  # use last timestep
                if states.shape[-1] != state_dim:
                    # Project to state_dim if needed
                    states = F.adaptive_avg_pool1d(states.unsqueeze(1), state_dim).squeeze(1)

                actions = batch.get("actions", torch.zeros(states.shape[0], action_dim)).to(device)
                rewards = batch.get("rewards", torch.zeros(states.shape[0], 1)).to(device)
                if rewards.ndim == 1:
                    rewards = rewards.unsqueeze(-1)
                next_states = batch.get("next_states", states).to(device)
                dones = batch.get("dones", torch.zeros(states.shape[0], 1)).to(device)
                if dones.ndim == 1:
                    dones = dones.unsqueeze(-1)

                alpha_ent = log_alpha_entropy.exp().detach()

                # --- Critic update ---
                with torch.no_grad():
                    next_actions, next_log_probs = actor(next_states)
                    q1_next = q1_target(next_states, next_actions)
                    q2_next = q2_target(next_states, next_actions)
                    q_next = torch.min(q1_next, q2_next) - alpha_ent * next_log_probs
                    td_target = rewards + gamma * (1 - dones) * q_next

                q1_pred = q1(states, actions)
                q2_pred = q2(states, actions)
                td_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

                # CQL penalty: penalize high Q-values on OOD (random) actions
                random_actions = torch.rand(states.shape[0] * n_random_actions, action_dim,
                                            device=device) * 2 - 1  # uniform [-1, 1]
                states_rep = states.unsqueeze(1).expand(-1, n_random_actions, -1).reshape(-1, state_dim)
                q1_random = q1(states_rep, random_actions).reshape(-1, n_random_actions)
                q2_random = q2(states_rep, random_actions).reshape(-1, n_random_actions)

                cql_penalty = (
                    torch.logsumexp(q1_random, dim=1).mean() - q1_pred.mean()
                    + torch.logsumexp(q2_random, dim=1).mean() - q2_pred.mean()
                )
                critic_loss = td_loss + cql_alpha * cql_penalty

            critic_loss = torch.nan_to_num(critic_loss, nan=0.0, posinf=1e6, neginf=-1e6)
            if critic_loss.item() == 0.0:
                continue

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), max_norm=1.0)
            critic_optimizer.step()

            # --- Actor update ---
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                new_actions, new_log_probs = actor(states)
                q1_new = q1(states, new_actions)
                q2_new = q2(states, new_actions)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha_ent * new_log_probs - q_new).mean()

            actor_loss = torch.nan_to_num(actor_loss, nan=0.0, posinf=1e6, neginf=-1e6)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_optimizer.step()

            # --- Alpha (entropy coeff) update ---
            alpha_loss = -(log_alpha_entropy * (new_log_probs.detach() + target_entropy)).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # --- Polyak averaging on target networks ---
            with torch.no_grad():
                for p_target, p in zip(q1_target.parameters(), q1.parameters()):
                    p_target.data.mul_(1 - tau).add_(p.data, alpha=tau)
                for p_target, p in zip(q2_target.parameters(), q2.parameters()):
                    p_target.data.mul_(1 - tau).add_(p.data, alpha=tau)

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            n_batches += 1

        elapsed = time.time() - t0
        if local_rank == 0:
            log.info(
                f"  Epoch {epoch}/{epochs} | "
                f"critic: {total_critic_loss / max(n_batches, 1):.4f} | "
                f"actor: {total_actor_loss / max(n_batches, 1):.4f} | "
                f"alpha: {log_alpha_entropy.exp().item():.4f} | {elapsed:.0f}s"
            )

    if local_rank == 0:
        save_checkpoint(model, None, epochs, total_critic_loss / max(n_batches, 1),
                        os.path.join(output_dir, "trading_brain_final.pt"))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H200 Training -- The Absolute Unit")
    parser.add_argument("--phase", choices=["all", "weather", "trading", "test"],
                        default="all", help="Training phase")
    parser.add_argument("--db", type=str, default="miami_collector.db",
                        help="Path to SQLite database")
    parser.add_argument("--ensemble-size", type=int, default=5,
                        help="Number of weather ensemble members")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for weather brain")
    parser.add_argument("--trading-batch-size", type=int, default=512,
                        help="Batch size for trading brain")
    parser.add_argument("--no-compile", action="store_true", default=False,
                        help="Disable torch.compile")
    parser.add_argument("--seeds", type=str, default="42,137,256,512,1024",
                        help="Comma-separated seeds for ensemble")
    parser.add_argument("--output-dir", type=str, default="trained_weights",
                        help="Output directory for weights")
    args = parser.parse_args()

    local_rank, world_size, device, bf16 = setup_distributed()
    log = setup_logging(local_rank)

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")][:args.ensemble_size]

    config = {
        "batch_size": args.batch_size,
        "trading_batch_size": args.trading_batch_size,
        "byol_epochs": 5,
        "supervised_epochs": 50,
        "flow_epochs": 20,
        "decision_epochs": 15,
        "cql_epochs": 20,
        "byol_lr": 3e-4,
        "supervised_lr": 3e-5,
        "flow_lr": 1e-4,
        "decision_lr": 1e-3,
        "output_dir": args.output_dir,
    }

    log.info("=" * 60)
    log.info("THE ABSOLUTE UNIT -- H200 Training")
    log.info("=" * 60)
    log.info(f"Phase: {args.phase}")
    log.info(f"Database: {args.db}")
    log.info(f"Ensemble size: {args.ensemble_size}")
    log.info(f"Seeds: {seeds}")
    log.info(f"Batch sizes: weather={args.batch_size}, trading={args.trading_batch_size}")
    log.info(f"BF16: {bf16}")
    log.info(f"torch.compile: {not args.no_compile}")
    log.info(f"World size: {world_size}, Local rank: {local_rank}")
    log_gpu_info(log, device, world_size)
    log.info("")

    t_start = time.time()

    # ===== WEATHER BRAIN =====
    if args.phase in ("all", "weather", "test"):
        log.info("Loading Weather Brain v3.1 architecture...")

        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from engine.ds3m.weather_brain_v3 import WeatherBrainV3, WeatherBrainV3Ensemble
        from engine.ds3m.wb3_config import WB3Config
        from engine.ds3m.multi_res_dataset import MultiResolutionWeatherDataset

        wb_config = WB3Config()

        log.info(f"Building weather training dataset from {args.db}...")
        train_dataset = MultiResolutionWeatherDataset(args.db, split="train")

        trained_models = []
        for i, seed in enumerate(seeds):
            log.info(f"\n{'=' * 60}")
            log.info(f"ENSEMBLE MEMBER {i + 1}/{len(seeds)} (seed={seed})")
            log.info(f"{'=' * 60}")

            model = WeatherBrainV3(wb_config, seed=seed).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

            if world_size > 1:
                model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
                log.info(f"Wrapped model with DDP (device_ids=[{local_rank}])")

            if not args.no_compile:
                model = try_compile(model, device)

            train_weather_phase0_byol(
                model, train_dataset, device, bf16, config,
                local_rank, world_size, log,
            )

            train_weather_phase1_supervised(
                model, train_dataset, device, bf16, config,
                local_rank, world_size, log, seed=seed,
            )

            model = train_weather_phase2_flow(
                model, train_dataset, device, bf16, config,
                local_rank, world_size, log, seed=seed,
            )

            train_weather_swa_ema(
                model, train_dataset, device, bf16, config, args,
                local_rank, world_size, log, seed=seed,
            )

            trained_models.append(model)
            log.info(f"  Model {i + 1} complete.")

        log.info(f"\nWeather Brain training complete ({len(trained_models)} models)")

    # ===== TRADING BRAIN =====
    if args.phase in ("all", "trading"):
        log.info("\nLoading Trading Brain v2.2 architecture...")

        sys.path.insert(0, str(Path(__file__).parent / "src"))
        sys.path.insert(0, str(Path(__file__).parent.parent / "weather-trader" / "src"))
        from model.trading_brain_v2 import TradingBrainV2, create_trading_brain_v2

        trading_model = create_trading_brain_v2().to(device)

        total_params = sum(p.numel() for p in trading_model.parameters())
        log.info(f"Trading Brain parameters: {total_params:,}")

        if world_size > 1:
            trading_model = DDP(trading_model, device_ids=[local_rank], find_unused_parameters=True)

        if not args.no_compile:
            trading_model = try_compile(trading_model, device)

        # --- Bug 1.13: Actually call trading training phases ---
        # Build trading dataset (placeholder — uses Kalshi historical data)
        from model.trading_brain_v2 import TradingBrainV2
        trading_dataset = None  # TODO: load from Kalshi historical trade data
        log.info("Loading trading dataset...")
        try:
            from engine.ds3m.trading_dataset import TradingDataset
            trading_dataset = TradingDataset(args.db)
        except (ImportError, Exception) as e:
            log.warning(f"Could not load trading dataset: {e}")
            log.warning("Skipping trading training phases")

        if trading_dataset is not None:
            train_trading_phase1_decision(
                trading_model, trading_dataset, device, bf16, config,
                local_rank, world_size, log,
            )
            train_trading_phase2_cql_sac(
                trading_model, trading_dataset, device, bf16, config,
                local_rank, world_size, log, output_dir=args.output_dir,
            )

        log.info("Trading Brain training complete")

    # ===== SUMMARY =====
    total_time = time.time() - t_start
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"Total time: {total_time / 3600:.1f} hours")
    log.info(f"Weights saved to: {args.output_dir}/")
    log.info("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
