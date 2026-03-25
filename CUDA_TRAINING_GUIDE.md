# CUDA Training Guide — RTX 5070

Instructions for training the DS3M GraphMamba on a Windows PC with NVIDIA RTX 5070.

---

## Prerequisites

- Windows 10/11
- NVIDIA RTX 5070 with latest drivers (https://www.nvidia.com/drivers)
- Python 3.11 or 3.12 (NOT 3.13 — PyTorch CUDA doesn't support it yet)

---

## Step 1: Install Python

Download Python 3.12 from https://www.python.org/downloads/
- Check "Add to PATH" during install
- Verify: `python --version`

---

## Step 2: Transfer Project Files

Files will be transferred via SCP from the Mac. After transfer, you should have:
```
C:\Users\<you>\miami-project\
├── src\engine\ds3m\    ← all the model code
├── miami_collector.db  ← 1.3GB training database
├── pyproject.toml
└── ...
```

---

## Step 3: Install PyTorch with CUDA

Open Command Prompt in the project directory:

```cmd
cd C:\Users\<you>\miami-project

python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

Verify CUDA works:
```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

Should print: `CUDA: True, GPU: NVIDIA GeForce RTX 5070`

---

## Step 4: Run Training

```cmd
python -m engine.ds3m.train --graph --phase all --epochs1 50 --epochs2 100 --batch 64 --device cuda
```

This trains:
- **Phase 1**: GraphMamba pre-training (8L Mamba d=384, 35 stations, 52 features)
- **Phase 2**: NSF fine-tuning on bracket settlements

Expected time on 5070: **2-4 hours total**

### Monitor progress

Training logs print every 5 epochs:
```
Epoch 0: loss=X.XXXX, best=X.XXXX, lr=0.001000
Epoch 5: loss=X.XXXX, best=X.XXXX, lr=0.000XXX
...
```

Weights auto-save to `analysis_data/graph_mamba_pretrained.pt` whenever loss improves.

---

## Step 5: Transfer Trained Weights Back to Mac

After training completes, copy the weights back:

```cmd
scp analysis_data\graph_mamba_pretrained.pt user@<mac-ip>:/Users/blakebot/blakebot/miami-project/analysis_data/
scp analysis_data\nsf_trained.pt user@<mac-ip>:/Users/blakebot/blakebot/miami-project/analysis_data/
```

Or use the same shared folder / USB method.

---

## Architecture Summary (what you're training)

```
12.6M parameters total:

GraphMamba (10.4M):
  └─ 8-layer Mamba2 (d_model=384, d_state=48) — shared across 35 stations
  └─ 4 graph attention layers (6 heads, wind-direction-dynamic edges)
  └─ 35 Florida ASOS stations within 350km of KMIA
  └─ 52-feature input vector per station per timestep

DPF (111K):
  └─ 1000 differentiable particles, d_latent=64
  └─ HDP regime discovery (learnable α, merge/split)

NSF (2.1M):
  └─ 12 coupling layers, 32 spline bins, hidden=192
  └─ Conditional on 408-dim context (384 Mamba + 19 stats + 5 regime)
```

---

## Troubleshooting

**"CUDA out of memory"**: Reduce batch size: `--batch 32` or `--batch 16`

**"torch.cuda not available"**:
- Update NVIDIA drivers
- Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

**Import errors**: Make sure you're in the project root and ran `pip install -e .`

**"No module named engine"**: Run from the `src/` parent: `cd miami-project && python -m engine.ds3m.train ...`

---

## For Claude Code on the PC

If you're running Claude Code on the PC to manage training:

1. Install Claude Code: `npm install -g @anthropic-ai/claude-code`
2. `cd` to the project directory
3. Ask Claude to run the training command above
4. Claude can monitor the training output and transfer weights back when done

Claude Code has full access to the terminal, so it can:
- Start/stop training
- Monitor GPU usage (`nvidia-smi`)
- Check training logs
- Transfer weights back via SCP
- Run smoke tests on the trained model
