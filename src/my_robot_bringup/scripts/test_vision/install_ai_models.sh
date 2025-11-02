#!/bin/bash
# Install Grounded-Segment-Anything for Tests 2 & 3
# This script automates the setup of AI vision models

set -e  # Exit on error

echo "================================================================"
echo "Installing AI Vision Models for Tests 2 & 3"
echo "================================================================"
echo ""
echo "This will:"
echo "  1. Clone Grounded-Segment-Anything repository"
echo "  2. Install dependencies"
echo "  3. Download model weights (~2-3GB)"
echo "  4. Set up the models in your virtual environment"
echo ""
echo "Requirements:"
echo "  - ~3GB disk space"
echo "  - NVIDIA GPU with CUDA"
echo "  - Active internet connection"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

cd ~/vlm_robot_ws

echo ""
echo "================================================================"
echo "Step 1: Importing Grounded-Segment-Anything repository"
echo "================================================================"

# Check if repos file exists
if [ ! -f "src/tabletop-handybot/tabletop-handybot.repos" ]; then
    echo "Error: tabletop-handybot.repos not found!"
    exit 1
fi

# Import the repository
vcs import . --input src/tabletop-handybot/tabletop-handybot.repos

echo ""
echo "================================================================"
echo "Step 2: Activating virtual environment"
echo "================================================================"

source src/tabletop-handybot/venv/bin/activate

echo ""
echo "================================================================"
echo "Step 3: Installing Grounding DINO"
echo "================================================================"

cd Grounded-Segment-Anything/Grounded-Segment-Anything/GroundingDINO
pip install -e .

echo ""
echo "================================================================"
echo "Step 4: Installing Segment Anything (SAM)"
echo "================================================================"

cd ../segment_anything
pip install -e .

echo ""
echo "================================================================"
echo "Step 5: Downloading model weights"
echo "================================================================"

cd ~/vlm_robot_ws

# Create weights directory
mkdir -p model_weights
cd model_weights

echo "Downloading Grounding DINO weights..."
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

echo "Downloading SAM weights..."
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo ""
echo "================================================================"
echo "Step 6: Installing additional dependencies"
echo "================================================================"

cd ~/vlm_robot_ws
source src/tabletop-handybot/venv/bin/activate

# Run the venv fix script
bash src/my_robot_bringup/scripts/test_vision/fix_venv.sh

echo ""
echo "================================================================"
echo "Step 7: Verifying installation"
echo "================================================================"

python3 << EOF
import sys
try:
    import groundingdino
    print("✓ Grounding DINO installed")
except ImportError as e:
    print("✗ Grounding DINO not found:", e)
    sys.exit(1)

try:
    import segment_anything
    print("✓ SAM installed")
except ImportError as e:
    print("✗ SAM not found:", e)
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print("✗ PyTorch not found:", e)
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print("✗ OpenCV not found:", e)
    sys.exit(1)

print("\n✓ All dependencies installed successfully!")
EOF

echo ""
echo "================================================================"
echo "✓ Installation Complete!"
echo "================================================================"
echo ""
echo "Model weights saved to: ~/vlm_robot_ws/model_weights/"
echo ""
echo "You can now run Tests 2 & 3:"
echo ""
echo "  source src/tabletop-handybot/venv/bin/activate"
echo "  python3 src/my_robot_bringup/scripts/test_vision/test2_object_detection.py"
echo ""
echo "================================================================"
