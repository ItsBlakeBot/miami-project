#!/usr/bin/env python3
"""Mini Miami Training Script — clean, single-GPU, no footguns.

No torch.compile (DPF breaks it).
No Muon (port collisions, None grad crashes).
No DDP (single GPU).
Just AdamW with cosine decay, GradNorm at epoch 10, SWA/EMA at the end.

Usage:
    python -m engine.ds3m.mini_miami.train --db miami_collector.db
    python -m engine.ds3m.mini_miami.train --db miami_collector.db --epochs 100

From project root:
    python src/engine/ds3m/mini_miami/train.py --db miami_collector.db
"""

import argparse
import copy
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from engine.ds3m.multi_res_dataset import MultiResolutionWeatherDataset, _collate_multi_res
from engine.ds3m.mini_miami.config import MiniMiamiConfig
from engine.ds3m.mini_miami.model import MiniMiamiWeatherBrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mini_miami")


# ──────────────────────────────────────────────────────────────────
# BYOL (identical to big model, self-contained)
# ──────────────────────────────────────────────────────────────────

class BYOLProjector(nn.Module):
    def __init__(self, d_in, d_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, d_out),
        )

    def forward(self, x):
        return self.net(x)


class BYOLPredictor(nn.Module):
    def __init__(self, d_in, d_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, d_out),
        )

    def forward(self, x):
        return self.net(x)


def byol_augment(fine, medium, coarse, view_id=0):
    """Two different augmentation views."""
    import random
    if view_id == 0:
        shift = random.randint(1, 3) * random.choice([-1, 1])
        ns = 0.05 + random.random() * 0.15
        return (
            torch.roll(fine, shifts=shift, dims=1) + torch.randn_like(fine) * ns,
            torch.roll(medium, shifts=shift, dims=1) + torch.randn_like(medium) * ns,
            coarse + torch.randn_like(coarse) * ns * 0.5,
        )
    else:
        T = fine.shape[1]
        mask_len = random.randint(T // 10, T * 3 // 10)
        start = random.randint(0, T - mask_len)
        fine_aug = fine.clone()
        fine_aug[:, start:start + mask_len, :] = 0
        scale = 0.9 + torch.rand(1, device=fine.device) * 0.2
        return fine_aug * scale, medium * scale, coarse + torch.randn_like(coarse) * 0.15


@torch.no_grad()
def ema_update(online, target, tau):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1 - tau)


def train_byol(model, dataset, device, config, log):
    """Phase 0: BYOL contrastive pre-training."""
    log.info("=" * 50)
    log.info("PHASE 0: BYOL Pre-Training")
    log.info("=" * 50)

    epochs = config.byol_epochs
    d_repr = model.d_model
    batch_size = 16

    projector = BYOLProjector(d_repr).to(device)
    predictor = BYOLPredictor(256).to(device)
    target_model = copy.deepcopy(model)
    target_proj = copy.deepcopy(projector)
    for p in target_model.parameters():
        p.requires_grad = False
    for p in target_proj.parameters():
        p.requires_grad = False

    params = list(model.parameters()) + list(projector.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=config.weight_decay)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate_multi_res, num_workers=0)

    tau_base, total_steps, step = 0.996, epochs * len(loader), 0

    for epoch in range(epochs):
        model.train(); projector.train(); predictor.train()
        total_loss, n = 0.0, 0
        t0 = time.time()

        for batch in loader:
            fine = batch['fine'].to(device)
            medium = batch['medium'].to(device)
            coarse = batch['coarse'].to(device)

            f1, m1, c1 = byol_augment(fine, medium, coarse, 0)
            f2, m2, c2 = byol_augment(fine, medium, coarse, 1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z1 = projector(model(f1, m1, c1)['temporal_state'])
                p1 = predictor(z1)
                z2 = projector(model(f2, m2, c2)['temporal_state'])
                p2 = predictor(z2)

                with torch.no_grad():
                    tz1 = target_proj(target_model(f1, m1, c1)['temporal_state'])
                    tz2 = target_proj(target_model(f2, m2, c2)['temporal_state'])

                loss = (2 - 2 * F.cosine_similarity(p1, tz2.detach(), dim=-1).mean() +
                        2 - 2 * F.cosine_similarity(p2, tz1.detach(), dim=-1).mean()) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            tau = 1.0 - (1.0 - tau_base) * (np.cos(np.pi * step / max(total_steps, 1)) + 1) / 2
            ema_update(model, target_model, tau)
            ema_update(projector, target_proj, tau)
            step += 1

            total_loss += loss.item()
            n += 1

        log.info(f"  Epoch {epoch}/{epochs} | BYOL loss: {total_loss/max(n,1):.4f} | "
                 f"tau: {tau:.5f} | {time.time()-t0:.0f}s")

    del target_model, target_proj, projector, predictor
    torch.cuda.empty_cache()
    log.info("  BYOL complete")


def train_supervised(model, dataset, device, config, log, seed=42):
    """Phase 1: Supervised multi-task training with cosine decay."""
    log.info("=" * 50)
    log.info(f"PHASE 1: Supervised (seed={seed})")
    log.info("=" * 50)

    epochs = config.supervised_epochs
    batch_size = 16
    lr_peak = config.learning_rate
    lr_min = 1e-6
    warmup_epochs = config.warmup_epochs
    gradnorm_start = 10

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_peak, weight_decay=config.weight_decay
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate_multi_res, num_workers=0)

    best_loss = float('inf')
    output_dir = "trained_weights_mini"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # LR schedule
        if epoch < warmup_epochs:
            lr = lr_peak * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            lr = lr_min + 0.5 * (lr_peak - lr_min) * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # GradNorm activation
        if epoch == gradnorm_start:
            model.heads._gradnorm_active = True
            model.heads._initialized = False
            log.info(f"  GradNorm activated at epoch {epoch}")

        model.train()
        total_loss, n_batches, n_skipped = 0.0, 0, 0
        task_accum = {}
        t0 = time.time()

        for batch in loader:
            fine = batch['fine'].to(device)
            medium = batch['medium'].to(device)
            coarse = batch['coarse'].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(fine, medium, coarse)
                targets = {k: v.to(device) for k, v in batch['targets'].items()}
                target_temp = targets.get('daily_max')
                if target_temp is not None:
                    target_temp = target_temp.unsqueeze(-1)
                loss_dict = model.compute_loss(outputs, targets, target_temp=target_temp)
                loss = loss_dict['total_loss']

            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            if loss.item() == 0.0:
                n_skipped += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k in ('total_loss', 'task_weights', '_raw_losses'):
                    continue
                if isinstance(v, dict):
                    continue
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                task_accum[k] = task_accum.get(k, 0.0) + val
            n_batches += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        task_str = " | ".join(f"{k}:{v/max(n_batches,1):.3f}" for k, v in sorted(task_accum.items()))
        skip_str = f" | skipped: {n_skipped}" if n_skipped else ""
        log.info(f"  Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | lr: {lr:.2e} | "
                 f"{task_str} | {elapsed:.0f}s{skip_str}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, os.path.join(output_dir, f"mini_miami_seed{seed}_best.pt"))

    log.info(f"  Best loss: {best_loss:.4f}")


def train_swa_ema(model, dataset, device, config, log, seed=42):
    """Phase 3: SWA + EMA weight averaging."""
    log.info("=" * 50)
    log.info("PHASE 3: SWA + EMA")
    log.info("=" * 50)

    epochs = config.swa_epochs
    batch_size = 16
    swa_lr = 1e-5

    swa_model = AveragedModel(model)
    ema_shadow = copy.deepcopy(model)
    for p in ema_shadow.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=swa_lr, weight_decay=config.weight_decay)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate_multi_res, num_workers=0)

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()

        for batch in loader:
            fine = batch['fine'].to(device)
            medium = batch['medium'].to(device)
            coarse = batch['coarse'].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(fine, medium, coarse)
                targets = {k: v.to(device) for k, v in batch['targets'].items()}
                target_temp = targets.get('daily_max')
                if target_temp is not None:
                    target_temp = target_temp.unsqueeze(-1)
                loss_dict = model.compute_loss(outputs, targets, target_temp=target_temp)
                loss = loss_dict['total_loss']

            loss = torch.nan_to_num(loss, nan=0.0)
            if loss.item() == 0.0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for p_ema, p_model in zip(ema_shadow.parameters(), model.parameters()):
                    p_ema.data.mul_(0.9995).add_(p_model.data, alpha=0.0005)

            total_loss += loss.item()
            n += 1

        swa_model.update_parameters(model)
        swa_scheduler.step()

        log.info(f"  SWA Epoch {epoch}/{epochs} | loss: {total_loss/max(n,1):.4f} | {time.time()-t0:.0f}s")

    # Update BN
    try:
        update_bn(loader, swa_model, device=device)
    except Exception as e:
        log.warning(f"  SWA BN update failed: {e}")

    output_dir = "trained_weights_mini"
    swa_sd = swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict()
    torch.save({'model': swa_sd}, os.path.join(output_dir, f"mini_miami_seed{seed}_swa.pt"))
    torch.save({'model': ema_shadow.state_dict()}, os.path.join(output_dir, f"mini_miami_seed{seed}_ema.pt"))
    log.info(f"  Saved SWA + EMA checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Mini Miami Training")
    parser.add_argument("--db", type=str, default="miami_collector.db")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="42,137,256,512,1024")
    parser.add_argument("--skip-byol", action="store_true")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run only Phase 1 (skip BYOL, flow, SWA)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = MiniMiamiConfig()
    config.supervised_epochs = args.epochs

    log.info(f"Loading dataset from {args.db}...")
    dataset = MultiResolutionWeatherDataset(args.db, split="train")
    log.info(f"Training samples: {len(dataset)}")

    seeds = [int(s) for s in args.seeds.split(",")]

    for i, seed in enumerate(seeds):
        log.info(f"\n{'='*50}")
        log.info(f"ENSEMBLE MEMBER {i+1}/{len(seeds)} (seed={seed})")
        log.info(f"{'='*50}")

        model = MiniMiamiWeatherBrain(config, seed=seed).to(device)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"Parameters: {total:,}")

        if not args.skip_byol and not args.phase1_only:
            train_byol(model, dataset, device, config, log)

        train_supervised(model, dataset, device, config, log, seed=seed)

        if not args.phase1_only:
            train_swa_ema(model, dataset, device, config, log, seed=seed)

        log.info(f"  Member {i+1} complete")

    log.info("\n" + "=" * 50)
    log.info("ALL TRAINING COMPLETE")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
