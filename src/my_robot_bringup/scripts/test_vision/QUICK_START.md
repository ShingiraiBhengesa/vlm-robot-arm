# Vision Tests - Quick Start Guide

## ⚠️ IMPORTANT: Read This First

**Test 1 (Camera Hardware) - Ready to Use** ✅
- Verifies ZED camera is working
- Requires only basic ROS2 setup
- **This is the essential test most users need**

**Tests 2 & 3 (AI Models) - Require Additional Setup** ⚠️
- These need Grounded-Segment-Anything installed
- See `SETUP_REQUIRED.md` for details
- Only needed if developing AI vision pipeline

**→ Start with Test 1 to verify your camera works!**

---

## Prerequisites Check

Before running Test 1:
1. ✅ ZED camera is connected (USB 3.0)
2. ✅ ROS2 workspace is sourced  
3. ✅ ZED camera node is running

Before running Tests 2 & 3 (if needed):
4. ✅ Grounded-Segment-Anything repo cloned and set up
5. ✅ AI model weights downloaded
6. ✅ Virtual environment fixed (see `fix_venv.sh`)

**See `SETUP_REQUIRED.md` for full AI setup instructions**

## Running the Tests

### Test 1: Camera Streaming (NO VENV NEEDED)

```bash
cd ~/vlm_robot_ws

# Launch ZED camera (Terminal 1)
ros2 launch zed_wrapper zedm.launch.py

# Run test (Terminal 2) - NO VENV NEEDED
python3 src/my_robot_bringup/scripts/test_vision/test1_camera_streaming.py
```

**Expected:** You should see RGB and depth sample images in `test_outputs/test1_camera/`

---

### Test 2: Object Detection (VENV REQUIRED!)

```bash
cd ~/vlm_robot_ws

# Launch ZED camera (Terminal 1 - if not already running)
ros2 launch zed_wrapper zedm.launch.py

# Run test (Terminal 2) - ACTIVATE VENV FIRST!
source src/tabletop-handybot/venv/bin/activate  # ← IMPORTANT!
python3 src/my_robot_bringup/scripts/test_vision/test2_object_detection.py
```

**Expected:** 
- Model loads (takes 10-30 seconds first time)
- Objects detected with bounding boxes
- Images saved to `test_outputs/test2_detection/`

**Tip:** Place objects (cup, bottle, marker, etc.) in camera view before running

---

### Test 3: Segmentation (VENV REQUIRED!)

```bash
cd ~/vlm_robot_ws

# Launch ZED camera (Terminal 1 - if not already running)
ros2 launch zed_wrapper zedm.launch.py

# Run test (Terminal 2) - ACTIVATE VENV FIRST!
source src/tabletop-handybot/venv/bin/activate  # ← IMPORTANT!
python3 src/my_robot_bringup/scripts/test_vision/test3_segmentation.py
```

**Expected:**
- Both models load
- Objects segmented with masks
- Multiple visualizations saved to `test_outputs/test3_segmentation/`

---

## Common Mistakes

### ❌ Error: "ModuleNotFoundError: No module named 'torch'"

**Problem:** You forgot to activate the virtual environment

**Solution:**
```bash
source src/tabletop-handybot/venv/bin/activate
# Then run the test again
```

### ❌ Error: "No RGB images received"

**Problem:** ZED camera node isn't running

**Solution:**
```bash
# In another terminal:
ros2 launch zed_wrapper zedm.launch.py
```

### ❌ Depth images are all NaN/black

**Problem:** Camera can't compute depth (needs textured surface)

**Solution:**
- Point camera at objects, not blank wall
- Ensure good lighting
- Keep objects 0.3m - 5m from camera

---

## Virtual Environment Cheat Sheet

**When do I need it?**
- ✅ Test 2 (Object Detection) - YES
- ✅ Test 3 (Segmentation) - YES  
- ❌ Test 1 (Camera Streaming) - NO

**How to activate:**
```bash
source src/tabletop-handybot/venv/bin/activate
```

**How to tell if it's active:**
- Your prompt shows `(venv)` at the start
- Example: `(venv) sbhengesa@linux-lab:~/vlm_robot_ws$`

**How to deactivate (when done):**
```bash
deactivate
```

---

## Test Summary

| Test | Purpose | Venv? | Duration |
|------|---------|-------|----------|
| 1 | Camera streaming | ❌ No | ~10 seconds |
| 2 | Object detection | ✅ Yes | ~30 seconds |
| 3 | Segmentation | ✅ Yes | ~45 seconds |

---

## Still Having Issues?

See the full documentation: `README.md` in this directory

Or check:
- System Python packages: `pip list | grep cv2`
- Virtual env packages: `source venv/bin/activate && pip list | grep torch`
- ROS topics: `ros2 topic list | grep zed`
- Camera status: `ZED_Diagnostic`
