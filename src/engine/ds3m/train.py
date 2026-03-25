"""DS3M v2 training runner.

Usage:
  python -m engine.ds3m.train --phase 1          # Mamba pre-training
  python -m engine.ds3m.train --phase 2          # NSF + DPF fine-tuning
  python -m engine.ds3m.train --phase all        # Full pipeline
  python -m engine.ds3m.train --phase nightly    # Post-settlement nightly
  python -m engine.ds3m.train --graph            # Use GraphMamba (spatial)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from engine.ds3m.mamba_encoder import MambaEncoder, MambaConfig
from engine.ds3m.graph_mamba import GraphMambaEncoder, GraphMambaConfig
from engine.ds3m.diff_particle_filter import DifferentiableParticleFilter, DPFConfig
from engine.ds3m.neural_spline_flow import ConditionalNSF, NSFConfig
from engine.ds3m.training_pipeline import DS3MTrainer

log = logging.getLogger(__name__)


def build_models(device: torch.device, use_graph: bool = False):
    """Instantiate fresh models for training."""
    if use_graph:
        # FULL SEND: GraphMamba with expanded 52-feature vector
        mamba = GraphMambaEncoder(GraphMambaConfig())  # defaults are full send
        d_model = mamba.config.mamba.d_model  # 384
    else:
        mamba = MambaEncoder(MambaConfig())
        d_model = mamba.config.d_model

    dpf = DifferentiableParticleFilter(
        DPFConfig(n_particles=1000, d_latent=64, k_regimes=5),
        d_mamba=d_model,
    )
    nsf = ConditionalNSF(NSFConfig(
        d_condition=19 + 5 + d_model,
        n_transforms=12,
        n_bins=32,
        hidden_dim=192,
        n_hidden_layers=3,
    ))

    mamba.to(device)
    dpf.to(device)
    nsf.to(device)

    n_params = sum(p.numel() for m in [mamba, dpf, nsf] for p in m.parameters())
    log.info(f"Models initialized: {n_params:,} total parameters")
    log.info(f"  {'GraphMamba' if use_graph else 'Mamba'}: {sum(p.numel() for p in mamba.parameters()):,}")
    log.info(f"  DPF:   {sum(p.numel() for p in dpf.parameters()):,}")
    log.info(f"  NSF:   {sum(p.numel() for p in nsf.parameters()):,}")

    return mamba, dpf, nsf


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="DS3M v2 Training")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--phase", default="all", choices=["1", "2", "all", "nightly"])
    parser.add_argument("--epochs1", type=int, default=50, help="Mamba pre-training epochs")
    parser.add_argument("--epochs2", type=int, default=100, help="NSF training epochs")
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="Force device (cpu/mps/cuda)")
    parser.add_argument("--graph", action="store_true", help="Use GraphMamba (spatial attention)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # Ensure output dir
    Path("analysis_data").mkdir(exist_ok=True)

    # Build models
    mamba, dpf, nsf = build_models(device, use_graph=args.graph)

    # Load pre-existing weights if available
    weight_prefix = "graph_" if args.graph else ""
    for name, model in [
        (f"{weight_prefix}mamba_pretrained.pt", mamba),
        ("dpf_live.pt", dpf),
        ("nsf_trained.pt", nsf),
    ]:
        path = Path("analysis_data") / name
        if path.exists():
            try:
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                log.info(f"Loaded existing weights: {name}")
            except Exception as e:
                log.warning(f"Could not load {name}: {e}")

    trainer = DS3MTrainer(mamba, dpf, nsf, args.db, device)

    t0 = time.time()
    results = {}

    if args.phase in ("1", "all"):
        results["mamba"] = trainer.pretrain_mamba(
            epochs=args.epochs1, lr=args.lr1, batch_size=args.batch,
        )

    if args.phase in ("2", "all"):
        results["nsf"] = trainer.train_nsf(
            epochs=args.epochs2, lr=args.lr2, batch_size=args.batch,
        )

    if args.phase == "nightly":
        results = trainer.run_nightly_training()

    elapsed = time.time() - t0
    log.info(f"Training complete in {elapsed:.1f}s")
    for phase, r in results.items():
        log.info(f"  {phase}: {r}")


if __name__ == "__main__":
    main()
