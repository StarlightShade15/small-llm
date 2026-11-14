#!/usr/bin/env python3
"""
Setup script for small-lm project.
Installs dependencies and prepares the environment.
"""
import subprocess
import sys
import os
import platform

def run_cmd(cmd, check=True):
    """Run a command and return result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_pytorch(cuda_available):
    """Install PyTorch with appropriate CUDA version."""
    if cuda_available:
        # CUDA 12.1 is widely compatible
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    run_cmd(cmd)

def install_deps():
    """Install all project dependencies."""
    deps = [
        "pyyaml>=6.0",
        "tokenizers>=0.19",
        "datasets>=2.19",
        "accelerate>=0.30",
        "tqdm>=4.66",
        "wandb>=0.17",  # Optional, for logging
        "numpy>=1.24",
        "pandas>=2.1",  # For synthetic data generation
    ]
    for dep in deps:
        run_cmd(f"pip install {dep}", check=False)

def main():
    print("=" * 60)
    print("small-lm Setup")
    print("=" * 60)
    
    # Check Python version
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check/install PyTorch
    print("\n--- Checking PyTorch ---")
    try:
        import torch
        print(f"PyTorch already installed: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA available: Yes ({torch.cuda.get_device_name(0)})")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("CUDA available: No (using CPU)")
    except ImportError:
        print("PyTorch not found. Installing...")
        # Try to detect CUDA before installing
        cuda_available = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            cuda_available = result.returncode == 0
        except FileNotFoundError:
            pass
        install_pytorch(cuda_available)
        import torch
        cuda_available = torch.cuda.is_available()
    
    # Install other dependencies
    print("\n--- Installing dependencies ---")
    install_deps()
    
    # Verify key imports
    print("\n--- Verifying imports ---")
    imports = ["yaml", "tokenizers", "datasets", "accelerate", "tqdm", "numpy", "pandas"]
    for imp in imports:
        try:
            __import__(imp)
            print(f"  {imp}: OK")
        except ImportError as e:
            print(f"  {imp}: FAILED - {e}")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"Device: {'CUDA' if cuda_available else 'CPU'}")
    print("=" * 60)

if __name__ == "__main__":
    main()