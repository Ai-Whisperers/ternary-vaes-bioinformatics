# Installation

Complete installation guide for Ternary VAE on all platforms.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Runtime |
| pip | 23.0+ | Package manager |
| Git | 2.30+ | Version control |
| CUDA | 11.8+ | GPU acceleration (optional) |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 32 GB |
| GPU VRAM | 4 GB | 16 GB |
| Storage | 5 GB | 20 GB |
| CPU | 4 cores | 8+ cores |

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Clone repository
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Create virtual environment
python -m venv .venv

# Activate (choose your platform)
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows CMD
.venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Method 2: Conda

```bash
# Clone repository
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Create conda environment
conda create -n ternary-vae python=3.11 -y
conda activate ternary-vae

# Install PyTorch with CUDA (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

### Method 3: Docker

```bash
# Build image
docker build -t ternary-vae .

# Run container
docker run -it --gpus all -v $(pwd):/workspace ternary-vae

# Or use docker-compose
docker-compose up -d
```

## Platform-Specific Instructions

### Windows

```powershell
# Install Python from python.org (check "Add to PATH")

# Open PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Clone and install
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Common Windows Issues**:
- If `pip install` fails with "Microsoft Visual C++ required", install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- For CUDA, install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Clone and install
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note**: macOS does not support NVIDIA CUDA. Use CPU or Apple Silicon MPS:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git -y

# Clone and install
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GPU Setup

### NVIDIA CUDA

1. Check GPU compatibility:
```bash
nvidia-smi
```

2. Install PyTorch with CUDA:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Verify installation:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### CPU-Only Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Development Installation

For contributors:

```bash
# Install with dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests to verify
pytest tests/ -v
```

## Verify Installation

Run the verification script:

```python
# verify_install.py
from src.models import TernaryVAE
from src.config import load_config, EPSILON
from src.geometry import poincare_distance
from src.losses import LossRegistry
import torch

print("Checking imports...")
print(f"  TernaryVAE: OK")
print(f"  EPSILON: {EPSILON}")

print("\nChecking GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

print("\nCreating model...")
model = TernaryVAE(input_dim=19683, latent_dim=16)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nRunning forward pass...")
x = torch.randn(2, 19683)
outputs = model(x)
print(f"  Output shape: {outputs['reconstruction'].shape}")

print("\n Installation verified successfully!")
```

```bash
python verify_install.py
```

## Troubleshooting Installation

### "ModuleNotFoundError: No module named 'src'"

```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### "CUDA out of memory"

```python
# Reduce batch size
config = TrainingConfig(batch_size=32)  # Instead of 128

# Or use gradient checkpointing
model.gradient_checkpointing_enable()
```

### "geoopt not found"

```bash
pip install geoopt
```

### "Permission denied" on Linux

```bash
# Fix permissions
chmod +x scripts/*.py
```

## Next Steps

- [[Quick-Start]] - Run your first experiment
- [[Configuration]] - Customize settings
- [[Training]] - Full training guide

---

*See also: [[Troubleshooting]] for more solutions*
