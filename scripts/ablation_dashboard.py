#!/usr/bin/env python3
"""Post-training feature ablation dashboard.

Loads trained WeatherBrainV3 weights and measures validation loss
when each feature group is zeroed out. Produces a ranked report
showing which data sources matter most.

Usage:
    python scripts/ablation_dashboard.py --checkpoint trained_weights/weather_brain_seed42_best.pt --db miami_collector.db
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, 'src')

from engine.ds3m.weather_brain_v3 import WeatherBrainV3
from engine.ds3m.multi_res_dataset import MultiResolutionWeatherDataset, _collate_multi_res
from engine.ds3m.wb3_config import WB3Config
from torch.utils.data import DataLoader

FEATURE_GROUPS = {
    'fine': {
        'HRRR temp/dewp': [0, 1],
        'HRRR wind': [2, 3],
        'HRRR pressure/cape': [4, 5],
        'HRRR cloud/vis': [6, 7],
        'Running max': [8, 9],
        'Time encoding': [10, 11, 12, 13],
        'Lightning': [14, 15],
        'Radar': [16, 17],
    },
    'medium': {
        'ASOS temp/dewp': [0, 1],
        'ASOS wind': [2, 3],
        'ASOS sky/vis/pressure': [4, 5, 6],
        'ASOS humidity': [7],
        'GFS NWP': list(range(8, 16)),
        'Buoy SST': [18, 19],
        'MOS guidance': [20, 21, 22],
        'Time encoding': list(range(23, 30)),
        'Derived': [30, 31],
    },
    'coarse': {
        'GFS temp/dewp/wind': [0, 1, 2, 3],
        'ECMWF temp/dewp/wind': [4, 5, 6, 7],
        'MOS guidance': [8, 9, 10],
        'Time encoding': list(range(11, 18)),
        'ENSO ONI': [18],
        'NAO': [19],
        'PNA': [20],
        'MJO': [21, 22],
        'AMO': [23],
    },
}


@torch.no_grad()
def compute_loss(model, loader, device, bf16=True):
    """Run model over full val set and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        fine = batch['fine'].to(device)
        medium = batch['medium'].to(device)
        coarse = batch['coarse'].to(device)
        masks = {k: v.to(device) for k, v in batch['feature_masks'].items()}
        targets = {k: v.to(device) for k, v in batch['targets'].items()}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            outputs = model(fine, medium, coarse, feature_masks=masks)
            loss = model.compute_loss(outputs, targets)['total_loss']

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_loss_ablated(model, loader, device, branch, indices, bf16=True):
    """Run model with specific feature indices zeroed out in a branch."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        fine = batch['fine'].to(device)
        medium = batch['medium'].to(device)
        coarse = batch['coarse'].to(device)
        masks = {k: v.to(device) for k, v in batch['feature_masks'].items()}
        targets = {k: v.to(device) for k, v in batch['targets'].items()}

        # Zero out the specified feature indices in the target branch
        if branch == 'fine':
            fine = fine.clone()
            fine[:, :, indices] = 0.0
        elif branch == 'medium':
            medium = medium.clone()
            medium[:, :, indices] = 0.0
        elif branch == 'coarse':
            coarse = coarse.clone()
            coarse[:, :, indices] = 0.0

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            outputs = model(fine, medium, coarse, feature_masks=masks)
            loss = model.compute_loss(outputs, targets)['total_loss']

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_ablation(model, loader, device, bf16=True):
    """Run full ablation study across all feature groups."""
    print("Computing baseline loss...")
    t0 = time.time()
    baseline_loss = compute_loss(model, loader, device, bf16)
    print(f"  Baseline loss: {baseline_loss:.6f}  ({time.time() - t0:.1f}s)")
    print()

    results = []

    for branch, groups in FEATURE_GROUPS.items():
        print(f"Ablating {branch} branch ({len(groups)} groups)...")
        for group_name, indices in groups.items():
            t0 = time.time()
            ablated_loss = compute_loss_ablated(
                model, loader, device, branch, indices, bf16
            )
            impact_pct = (ablated_loss - baseline_loss) / baseline_loss * 100
            elapsed = time.time() - t0

            results.append({
                'branch': branch,
                'group': group_name,
                'indices': indices,
                'baseline_loss': baseline_loss,
                'ablated_loss': ablated_loss,
                'impact_pct': impact_pct,
            })
            print(f"  {group_name:30s}  loss={ablated_loss:.6f}  "
                  f"impact={impact_pct:+.2f}%  ({elapsed:.1f}s)")

    return results, baseline_loss


def print_report(results, baseline_loss):
    """Print ranked ablation report."""
    ranked = sorted(results, key=lambda r: -r['impact_pct'])

    print()
    print("=" * 80)
    print("FEATURE ABLATION REPORT")
    print(f"Baseline validation loss: {baseline_loss:.6f}")
    print("=" * 80)
    print(f"{'Rank':<5} {'Branch':<8} {'Feature Group':<30} "
          f"{'Ablated Loss':<14} {'Impact %':<10}")
    print("-" * 80)

    for i, r in enumerate(ranked, 1):
        marker = " ***" if r['impact_pct'] > 5.0 else (
            " **" if r['impact_pct'] > 2.0 else (
                " *" if r['impact_pct'] > 1.0 else ""
            )
        )
        print(f"{i:<5} {r['branch']:<8} {r['group']:<30} "
              f"{r['ablated_loss']:<14.6f} {r['impact_pct']:+8.2f}%{marker}")

    print("-" * 80)
    print("Legend: *** >5% impact  ** >2% impact  * >1% impact")
    print()

    # Summary by branch
    print("Summary by branch:")
    for branch in ['fine', 'medium', 'coarse']:
        branch_results = [r for r in ranked if r['branch'] == branch]
        if branch_results:
            max_impact = max(r['impact_pct'] for r in branch_results)
            avg_impact = sum(r['impact_pct'] for r in branch_results) / len(branch_results)
            print(f"  {branch:8s}  max impact: {max_impact:+.2f}%  "
                  f"avg impact: {avg_impact:+.2f}%")

    # Most and least important
    print()
    print(f"Most important:  {ranked[0]['branch']}/{ranked[0]['group']} "
          f"({ranked[0]['impact_pct']:+.2f}%)")
    print(f"Least important: {ranked[-1]['branch']}/{ranked[-1]['group']} "
          f"({ranked[-1]['impact_pct']:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Feature ablation dashboard for WeatherBrainV3"
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--db', default='miami_collector.db',
                        help='Path to SQLite database')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for evaluation')
    parser.add_argument('--no-bf16', action='store_true',
                        help='Disable BF16 autocast')
    args = parser.parse_args()

    bf16 = not args.no_bf16

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: Running on CPU (will be slow)")

    # Load config and model
    print(f"Loading checkpoint: {args.checkpoint}")
    config = WB3Config()
    model = WeatherBrainV3(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    # Strip DDP 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Load dataset
    print(f"Loading dataset from: {args.db}")
    dataset = MultiResolutionWeatherDataset(args.db)
    print(f"  Dataset size: {len(dataset)} samples")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=4, pin_memory=True,
        collate_fn=_collate_multi_res,
    )

    # Run ablation
    print()
    results, baseline_loss = run_ablation(model, loader, device, bf16)

    # Print report
    print_report(results, baseline_loss)


if __name__ == '__main__':
    main()
