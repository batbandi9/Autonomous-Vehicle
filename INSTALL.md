# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU-accelerated training

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/batbandi9/Autonomous-Vehicle.git
cd Autonomous-Vehicle
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Package

```bash
pip install -e .
```

## Step-by-Step Installation

If you encounter issues with the quick installation, follow these steps:

### Install Core Dependencies

```bash
# Install gymnasium and highway-env
pip install gymnasium>=0.29.0
pip install highway-env>=1.8.0

# Install Stable Baselines3 for RL algorithms
pip install stable-baselines3>=2.0.0

# Install numpy and scipy
pip install numpy>=1.24.0
pip install scipy>=1.11.0
```

### Install MPC Dependencies

```bash
# Install CVXPY for MPC optimization
pip install cvxpy>=1.4.0
```

### Install Visualization Dependencies

```bash
# Install matplotlib for visualization
pip install matplotlib>=3.7.0
pip install tensorboard>=2.13.0
```

### Install Multi-Agent Dependencies

```bash
# Install PettingZoo for multi-agent support
pip install pettingzoo>=1.24.0
```

### Install Development Tools (Optional)

```bash
# Testing and formatting
pip install pytest>=7.4.0
pip install black>=23.0.0
pip install flake8>=6.0.0
```

## Verify Installation

Run the integration test to verify everything is installed correctly:

```bash
python tests/test_integration.py
```

Expected output:
```
============================================================
Running Integration Tests
============================================================
✓ Package version: 0.1.0
✓ Environment modules: ['ParkingEnv', 'LaneChangeEnv', 'HighwayEnv']
✓ Agent modules: ['E2EAgent', 'HRLAgent']
...
============================================================
All basic integration tests passed! ✓
============================================================
```

## Common Issues

### Issue: `highway-env` not found

**Solution**: Install highway-env separately
```bash
pip install highway-env
```

### Issue: CVXPY installation fails

**Solution**: Install build dependencies first
```bash
# On Ubuntu/Debian
sudo apt-get install gcc g++ python3-dev

# On macOS
xcode-select --install

# Then install CVXPY
pip install cvxpy
```

### Issue: Slow training on CPU

**Solution**: Install PyTorch with CUDA support for GPU acceleration
```bash
# For CUDA 11.8
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Issue: ImportError for stable_baselines3

**Solution**: Install Stable Baselines3
```bash
pip install stable-baselines3[extra]
```

## Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build Docker image
docker build -t autonomous-vehicle .

# Run container
docker run -it --rm -v $(pwd):/workspace autonomous-vehicle bash
```

Note: Docker configuration will be added in future updates.

## Virtual Environment (Recommended)

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Testing Your Installation

Run a quick example to test:

```bash
python examples/train_parking_e2e.py
```

This will start training a parking agent. You should see training progress in the terminal.

## GPU Support

For GPU-accelerated training:

1. Install CUDA (if not already installed)
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify GPU is available:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   ```

## Next Steps

After successful installation:

1. Read the [README](README.md) for an overview
2. Check the [examples](examples/README.md) directory
3. Start with a simple example: `python examples/train_parking_e2e.py`
4. Monitor training with TensorBoard: `tensorboard --logdir=./logs`

## Getting Help

If you encounter issues:

1. Check the [Common Issues](#common-issues) section above
2. Open an issue on GitHub with:
   - Your Python version: `python --version`
   - Your OS information
   - Full error message
   - Steps to reproduce
