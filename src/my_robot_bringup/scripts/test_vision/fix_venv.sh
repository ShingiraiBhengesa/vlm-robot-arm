#!/bin/bash
# Fix virtual environment dependencies for vision tests
# This script fixes numpy/opencv compatibility issues

echo "================================================================"
echo "Fixing Virtual Environment Dependencies"
echo "================================================================"

# Activate virtual environment
source src/tabletop-handybot/venv/bin/activate

echo ""
echo "Step 1: Downgrading numpy to <2.0 for compatibility..."
pip install "numpy<2.0" --upgrade

echo ""
echo "Step 2: Installing OpenCV in virtual environment..."
pip install opencv-python opencv-contrib-python

echo ""
echo "Step 3: Installing missing dependencies..."
pip install torchvision supervision

echo ""
echo "Step 4: Verifying installations..."
echo ""
echo "NumPy version:"
python3 -c "import numpy; print(f'  numpy {numpy.__version__}')"

echo "OpenCV version:"
python3 -c "import cv2; print(f'  opencv {cv2.__version__}')"

echo "PyTorch version:"
python3 -c "import torch; print(f'  torch {torch.__version__}')"

echo ""
echo "================================================================"
echo "✓ Virtual environment fixed!"
echo "================================================================"
echo ""
echo "You can now run Tests 2 & 3:"
echo " source src/tabletop-handybot/venv/bin/activate"
echo " python3 src/my_robot_bringup/scripts/test_vision/test2_object_detection.py"
echo ""
