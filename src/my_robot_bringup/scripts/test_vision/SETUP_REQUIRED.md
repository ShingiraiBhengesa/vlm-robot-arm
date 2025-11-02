# Important: Tests 2 & 3 Require Additional Setup

## Current Status

**✅ Test 1 (Camera) - Ready to use**
- Test 1 only requires ROS2 and system libraries
- Works without any AI model setup
- This is the primary test to verify your camera hardware

**❌ Tests 2 & 3 (AI Models) - Require Grounded-Segment-Anything Setup**
- These tests need Grounding DINO and SAM AI models
- Requires complex setup with model weights and dependencies
- **Not included in basic ROS2 workspace**

## Why Tests 2 & 3 Don't Work Yet

The error `No module named 'groundingdino'` means you haven't set up the AI vision models yet. This requires:

1. **Grounded-Segment-Anything repository**
   - Clone from: https://github.com/IDEA-Research/Grounded-Segment-Anything
   - Install in the workspace root directory

2. **Download model weights** (~2-3GB)
   - Grounding DINO model weights
   - SAM (Segment Anything) model weights

3. **Build and install the models**
   - Compile GroundingDINO
   - Set up SAM
   - Configure paths

## What You Should Do

### Option 1: Focus on Test 1 (Recommended for Now)

**Test 1 is the essential test** - it verifies your hardware works:
- ✅ Camera connects properly
- ✅ RGB images publish correctly  
- ✅ Depth sensing works
- ✅ ROS2 topics are active

This is sufficient to verify your VLM robot's vision hardware is functional.

```bash
cd ~/vlm_robot_ws

# Terminal 1: Launch camera
ros2 launch zed_wrapper zedm.launch.py

# Terminal 2: Run Test 1
python3 src/my_robot_bringup/scripts/test_vision/test1_camera_streaming.py
```

### Option 2: Set Up Full AI Pipeline (Advanced)

If you need Tests 2 & 3 for object detection/segmentation:

1. **Follow the tabletop-handybot setup guide:**
   ```bash
   cd ~/vlm_robot_ws/src/tabletop-handybot
   cat README.md  # Read the setup instructions
   ```

2. **Key steps include:**
   - Clone Grounded-Segment-Anything
   - Download model weights
   - Install in virtual environment
   - Set up paths correctly

3. **This is a complex setup** that can take 30+ minutes and requires:
   - ~3GB disk space
   - NVIDIA GPU with CUDA
   - Compiling C++/CUDA code
   - Downloading large model files

## Summary

**For basic vision hardware testing:** ✅ Test 1 is ready and sufficient

**For full AI pipeline:** ❌ Tests 2 & 3 require Grounded-Segment-Anything setup first

The test suite was designed to be modular - Test 1 verifies hardware independently of the AI models. Most users only need Test 1 to confirm their camera works correctly.

---

## When to Use Each Test

| Test | Purpose | Setup Required | Typical Use Case |
|------|---------|----------------|------------------|
| **Test 1** | Verify camera hardware | Basic ROS2 | Hardware validation, troubleshooting |
| **Test 2** | Test AI object detection | Full AI setup | Development, AI model testing |
| **Test 3** | Test AI segmentation | Full AI setup | Development, AI model testing |

**Recommendation:** Start with Test 1. Only set up Tests 2 & 3 if you're actively developing/debugging the AI vision pipeline.
