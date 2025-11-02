# Vision System Testing Suite

This directory contains a comprehensive testing suite for the VLM Robot's vision pipeline components. The tests are designed to be run incrementally to verify each component before moving to the next.

## Overview

The vision pipeline consists of:
1. **ZED Camera** - Provides RGB and depth images
2. **Grounding DINO** - Zero-shot object detection
3. **SAM** - Precise object segmentation  
4. **Point Cloud Processing** - Converts depth to 3D coordinates

## Prerequisites

### Hardware
- ZED Mini camera connected via USB 3.0
- NVIDIA GPU with CUDA support (recommended)

### Software Setup
```bash
# 1. Source ROS2 workspace
cd ~/vlm_robot_ws
source /opt/ros/iron/setup.bash
source install/setup.bash

# 2. Activate virtual environment (for Tests 2-3)
source src/tabletop-handybot/venv/bin/activate

# 3. Verify ZED SDK installed
ZED_Diagnostic
```

## Test Sequence

Run tests in this order to ensure each component works before testing integration:

### Test 1: Camera Streaming ✓ (Start Here)

**Purpose**: Verify ZED camera publishes RGB and depth images correctly

**Run:**
```bash
# Terminal 1: Launch ZED camera
ros2 launch zed_wrapper zedm.launch.py

# Terminal 2: Run test
cd ~/vlm_robot_ws
python3 src/my_robot_bringup/scripts/test_vision/test1_camera_streaming.py
```

**Expected Output:**
- ✓ RGB and depth topics active
- Sample images saved to `test_outputs/test1_camera/`
- Frame rate ~15-30 Hz
- Depth range reasonable (e.g., 0.3m - 5.0m)

**Troubleshooting:**
- If no topics: Check `ros2 topic list | grep zed`
- If camera not detected: Run `ZED_Diagnostic`
- Check USB connection: `lsusb | grep 2b03`

---

### Test 2: Object Detection ✓

**Purpose**: Verify Grounding DINO can detect objects from text descriptions

**Run:**
```bash
# Terminal 1: Keep ZED camera running
ros2 launch zed_wrapper zedm.launch.py

# Terminal 2: Run test (with virtual env activated)
cd ~/vlm_robot_ws
source src/tabletop-handybot/venv/bin/activate
python3 src/my_robot_bringup/scripts/test_vision/test2_object_detection.py

# Or test specific objects:
python3 src/my_robot_bringup/scripts/test_vision/test2_object_detection.py --objects "apple,banana,phone"
```

**Expected Output:**
- ✓ Grounding DINO model loads (may take 10-30 seconds first time)
- Objects detected with bounding boxes
- Confidence scores > 0.35
- Annotated images saved to `test_outputs/test2_detection/`

**Tips:**
- Place objects in camera view before running
- Use common object names (cup, bottle, marker, phone, etc.)
- Adjust lighting if confidence scores are low
- If no detections, try:
  - Move objects closer to camera
  - Improve lighting
  - Use more generic object names

---

### Test 3: Segmentation ✓

**Purpose**: Verify SAM generates accurate segmentation masks for detected objects

**Run:**
```bash
# Terminal 1: Keep ZED camera running
ros2 launch zed_wrapper zedm.launch.py

# Terminal 2: Run test (with virtual env activated)
cd ~/vlm_robot_ws
source src/tabletop-handybot/venv/bin/activate
python3 src/my_robot_bringup/scripts/test_vision/test3_segmentation.py

# Or test specific objects:
python3 src/my_robot_bringup/scripts/test_vision/test3_segmentation.py --objects "cup,bottle" --samples 2
```

**Expected Output:**
- ✓ Both Grounding DINO and SAM models load
- Objects detected and segmented
- Multiple visualizations saved:
  - `original_*.jpg` - Raw camera image
  - `bboxes_*.jpg` - With bounding boxes
  - `masks_*.jpg` - With segmentation masks
  - `combined_*.jpg` - Both boxes and masks
  - `mask_*_obj*.png` - Individual mask images
- Output in `test_outputs/test3_segmentation/`

**Tips:**
- SAM requires more GPU memory than Grounding DINO
- Check mask coverage % - should be >70% of bounding box
- Verify masks don't overlap incorrectly
- GPU recommended but CPU works (slower)

---

## Test Results Interpretation

### Success Criteria

**Test 1 (Camera):**
- [x] RGB images 1280x720 or higher
- [x] Depth images same resolution as RGB
- [x] Frame rate ≥ 15 Hz
- [x] Depth range 0.3m - 5.0m

**Test 2 (Detection):**
- [x] Model loads without errors
- [x] Detects at least some target objects
- [x] Confidence scores reasonable (>0.3)
- [x] Bounding boxes align with objects

**Test 3 (Segmentation):**
- [x] Both models load successfully
- [x] Masks generated for detected objects
- [x] Masks align with object boundaries
- [x] Mask coverage >60% of bounding box

### Common Issues

**Issue: "No module named groundingdino"**
```bash
# Solution: Make sure virtual environment is activated
source src/tabletop-handybot/venv/bin/activate
pip install -r src/tabletop-handybot/requirements.txt
```

**Issue: CUDA out of memory**
```bash
# Solution: Close other GPU applications or reduce batch size
# Check GPU usage: nvidia-smi
```

**Issue: Camera images but no detections**
- Try different object names (more generic)
- Improve lighting conditions  
- Move objects closer to camera
- Ensure objects are fully visible

**Issue: Poor segmentation masks**
- Verify detection boxes are accurate first
- Check lighting - SAM works better with good lighting
- Ensure GPU has enough memory

---

## Output Directory Structure

```
test_outputs/
├── test1_camera/
│   ├── rgb_sample.jpg              ← Color image for verification
│   └── depth_sample_colorized.jpg  ← Depth visualization (should show colors)
├── test2_detection/
│   ├── original_sample_0.jpg
│   ├── detection_sample_0.jpg
│   ├── original_sample_1.jpg
│   └── detection_sample_1.jpg
└── test3_segmentation/
    ├── original_0.jpg
    ├── bboxes_0.jpg
    ├── masks_0.jpg
    ├── combined_0.jpg
    ├── mask_0_obj0_cup.png
    └── mask_0_obj1_bottle.png
```

**Note:** Only essential files are saved. Your actual code uses live ROS topics, not these saved images.

---

## Next Steps After Vision Tests Pass

Once all three vision tests pass successfully:

1. **Test Motion Control** (Test 4 - if needed)
   - Verify MoveIt2 motion planning
   - Test arm joint movements
   - Test gripper control

2. **Integration Test** (Test 5 - if needed)
   - Test vision + motion together
   - Test pick operation without AI
   - Test place operation without AI

3. **Full System Test**
   - Test complete AI pipeline
   - Test voice/text commands
   - Test full pick-and-place operations

---

## Performance Benchmarks

Typical processing times on recommended hardware (Intel Xeon W-2123, NVIDIA Quadro P1000):

| Component | Time | Notes |
|-----------|------|-------|
| Grounding DINO (first load) | 15-30s | One-time model loading |
| Grounding DINO (inference) | 0.5-1.0s | Per frame |
| SAM (first load) | 10-20s | One-time model loading |
| SAM (inference) | 0.3-0.8s | Per object |
| Total (detection + segmentation) | 1-3s | For 2-3 objects |

---

## Advanced Usage

### Custom Detection Thresholds

Edit the scripts to adjust detection sensitivity:

```python
# In test2_object_detection.py or test3_segmentation.py
BOX_THRESHOLD = 0.35  # Lower = more detections, more false positives
TEXT_THRESHOLD = 0.25  # Lower = more permissive text matching
NMS_THRESHOLD = 0.8   # Higher = fewer overlapping boxes removed
```

### Testing with Recorded Data

To test without hardware:
```bash
# Record a rosbag
ros2 bag record -o test_data /zedm/zed_node/rgb/image_rect_color /zedm/zed_node/depth/depth_registered

# Play it back
ros2 bag play test_data

# Run tests as normal
```

### Visualization with RViz

```bash
# Launch RViz to visualize camera data
ros2 launch zed_display_rviz2 display_zedm.launch.py
```

---

## Support

If you encounter issues:

1. Check the troubleshooting sections above
2. Review test output logs carefully
3. Verify all prerequisites are met
4. Check GPU/CUDA installation: `nvidia-smi`
5. Verify ZED SDK: `ZED_Diagnostic`

---

## Test Development

To add new tests:

1. Create `testN_description.py` in this directory
2. Follow the structure of existing tests
3. Include clear docstring with usage instructions
4. Save outputs to `test_outputs/testN_*/`
5. Print clear pass/fail results
6. Update this README

---

**Last Updated:** 2025-10-28
**Version:** 1.0.0
