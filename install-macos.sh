#!/bin/bash
# UAP Analysis Installation Script for macOS
# ==========================================
# This script installs the UAP Analysis Pipeline with Metal Performance Shaders support

set -e  # Exit on any error

echo "🍎 UAP Analysis Pipeline - macOS Installation"
echo "=============================================="

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required (found $python_version)"
    echo "   Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script is for macOS only"
    echo "   Use install-linux.sh for Linux systems"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📥 Installing base dependencies..."
pip install --upgrade pip

# Install core requirements
pip install -r requirements.txt

# Install macOS-specific requirements with MPS support
echo "🚀 Installing Metal Performance Shaders support..."
pip install -r requirements-macos.txt

# Verify PyTorch MPS support
echo "🧪 Testing Metal Performance Shaders..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.backends.mps.is_available():
    print('✅ Metal Performance Shaders (MPS) is available!')
    device = torch.device('mps')
    x = torch.randn(100, 100, device=device)
    print('✅ MPS test computation successful')
else:
    print('⚠️  MPS not available, will use CPU acceleration')
" || echo "⚠️  PyTorch installation may have issues"

# Test basic functionality
echo "🔬 Running setup test..."
python3 test_setup.py

# Make scripts executable
chmod +x run_analysis.py

echo ""
echo "🎉 Installation complete!"
echo "================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To analyze a video:"
echo "  python run_analysis.py your_video.mp4"
echo ""
echo "To run a quick analysis:"
echo "  python run_analysis.py --quick your_video.mp4"
echo ""
echo "Hardware acceleration status:"
python3 -c "
import sys; sys.path.insert(0, 'src/utils')
try:
    from acceleration import get_acceleration_manager
    accel = get_acceleration_manager()
    info = accel.get_device_info()
    print(f'  Device: {info[\"device_type\"]}')
    print(f'  Backend: {info[\"backend\"]}')
    if info['device_type'] == 'mps':
        print('  🚀 Metal Performance Shaders enabled!')
except:
    print('  ⚠️  Using CPU acceleration')
"
echo ""
echo "For more information, see README.md"