#!/bin/bash
# vast.ai "On Start" script - run this in the vast.ai console after instance launches
# This sets up SSH access and prepares the environment

set -e

echo "=== Setting up SSH access ==="

# Install SSH server
apt-get update && apt-get install -y openssh-server git

# Configure SSH
mkdir -p /var/run/sshd
echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config

# Set root password (CHANGE THIS!)
echo 'root:vastai2024' | chpasswd

# Generate host keys if missing
ssh-keygen -A

# Start SSH daemon
/usr/sbin/sshd -D &

echo "=== SSH Ready ==="
echo "Connect with: ssh -p <EXTERNAL_PORT> root@<VAST_IP>"
echo "Password: vastai2024"
echo ""

# Install Python deps
echo "=== Installing dependencies ==="
pip install -r /workspace/small-lm/requirements.txt --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || true

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}')"

echo "=== Ready! ==="
echo "Now SSH in and run:"
echo "  cd /workspace/small-lm"
echo "  python scripts/setup.py"
echo "  python -m src.generation.generate_data --num-facts 20000 --augmentations 3"
echo "  python -m src.tokenizer.tokenizer"
echo "  python -m src.training.train"