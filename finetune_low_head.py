#!/usr/bin/env python3
"""Fine-tune the LOW bracket head on settled KXLOWTMIA data.

The main training (train_h200.py) has bracket_loss_low=0.0 because all 103
settled LOW dates fall after the training cutoff (2025-01-01). This script:

1. Loads a trained weather brain checkpoint
2. Freezes everything except head_bracket_probs_low
3. Runs a few epochs on the 103 settled LOW dates
4. Saves the updated checkpoint

Usage:
    python finetune_low_head.py --checkpoint trained_weights/weather_brain_seed42_best.pt \
                                --db miami_collector.db \
                                --epochs 20 \
                                --lr 1e-3

Run this AFTER the main training completes, for each seed.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from engine.ds3m.weather_brain_v3 import WeatherBrainV3
from engine.ds3m.wb3_config import WB3Config
from engine.ds3m.multi_res_dataset import MultiResolutionWeatherDataset, _collate_multi_res

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("finetune_low")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LOW bracket head")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained weather brain checkpoint")
    parser.add_argument("--db", type=str, default="miami_collector.db",
                        help="Path to SQLite database")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of fine-tune epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for LOW head")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (default: overwrites input)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    config = WB3Config()
    model = WeatherBrainV3(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Freeze everything except the LOW bracket head
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze LOW bracket head in multi-task heads
    for name, param in model.heads.head_bracket_probs_low.named_parameters():
        param.requires_grad = True
        log.info(f"  Unfrozen: heads.head_bracket_probs_low.{name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Load dataset with ALL dates (not just train split)
    # We specifically want the val/test dates where LOW settlements exist
    dataset = MultiResolutionWeatherDataset(args.db, split="all")
    log.info(f"Full dataset: {len(dataset)} samples")

    # Filter to only dates with valid LOW bracket targets
    valid_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        bt_low = sample["targets"]["bracket_target_low"]
        if bt_low.item() >= 0:
            valid_indices.append(i)

    log.info(f"Dates with settled LOW brackets: {len(valid_indices)}")

    if len(valid_indices) == 0:
        log.error("No settled LOW bracket data found. Cannot fine-tune.")
        sys.exit(1)

    # Create subset dataset
    subset = torch.utils.data.Subset(dataset, valid_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate_multi_res, num_workers=0)

    # Optimizer — only LOW head params
    optimizer = torch.optim.AdamW(
        model.heads.head_bracket_probs_low.parameters(),
        lr=args.lr, weight_decay=0.01,
    )

    # Fine-tune
    model.train()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total_samples = 0

        for batch in loader:
            fine = batch["fine"].to(device)
            medium = batch["medium"].to(device)
            coarse = batch["coarse"].to(device)
            fm = {k: v.to(device) for k, v in batch["feature_masks"].items()}

            with torch.no_grad():
                # Forward pass through frozen backbone
                out = model(fine, medium, coarse, feature_masks=fm)

            # Only compute LOW bracket loss (with gradients on the head)
            low_logits = model.heads.head_bracket_probs_low(out["temporal_state"])
            bt_low = batch["targets"]["bracket_target_low"].to(device).long()

            valid = bt_low >= 0
            if not valid.any():
                continue

            loss = F.cross_entropy(low_logits[valid], bt_low[valid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Accuracy
            preds = low_logits[valid].argmax(dim=-1)
            correct += (preds == bt_low[valid]).sum().item()
            total_samples += valid.sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        acc = correct / max(total_samples, 1) * 100
        log.info(f"  Epoch {epoch}/{args.epochs} | LOW loss: {avg_loss:.4f} | "
                 f"acc: {acc:.1f}% ({correct}/{total_samples})")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save updated checkpoint
    output_path = args.output or args.checkpoint
    ckpt["model"] = model.state_dict()
    ckpt["low_finetune_epochs"] = args.epochs
    ckpt["low_finetune_best_loss"] = best_loss
    torch.save(ckpt, output_path)
    log.info(f"Saved fine-tuned checkpoint: {output_path}")
    log.info(f"Best LOW loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
