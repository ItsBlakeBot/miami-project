# H200 Training Setup Guide

## 1. Connect to VM
```bash
ssh user@<VM_IP>
```

## 2. Clone repos
```bash
git clone git@github.com:ItsBlakeBot/miami-project.git
git clone git@github.com:ItsBlakeBot/weather-trader.git
cd miami-project
```

## 3. Transfer database (from your Mac)
```bash
# On your Mac:
scp miami_collector.db user@<VM_IP>:~/miami-project/
```

## 4. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install scipy numpy muon torchdiffeq
pip install -e .

# Install weather-trader
cd ../weather-trader
pip install -e .
cd ../miami-project
```

## 5. Verify GPU
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"
# Should show: NVIDIA H200 SXM, 141.x GB
```

## 6. Verify model builds
```bash
python -c "
from engine.ds3m.weather_brain_v3 import WeatherBrainV3
from engine.ds3m.wb3_config import WB3Config
m = WeatherBrainV3(WB3Config())
total = sum(p.numel() for p in m.parameters())
print(f'Weather Brain: {total:,} params')
"
```

## 7. Start training
```bash
# Full training (both brains, ~16-18 hours)
python train_h200.py --phase all --db miami_collector.db

# Or weather only first (~6 hours)
python train_h200.py --phase weather --db miami_collector.db

# Then trading (~8 hours)
python train_h200.py --phase trading --db miami_collector.db
```

## 8. Monitor
```bash
# In another terminal:
tail -f training.log

# GPU utilization:
watch -n 1 nvidia-smi
```

## 9. Download weights when done
```bash
# On your Mac:
scp user@<VM_IP>:~/miami-project/trained_weights/*.pt ./analysis_data/
```

## Cost Estimate
- H200 SXM 141GB: ~$4.50/hr
- Training: ~16-18 hours
- Total: ~$72-81
