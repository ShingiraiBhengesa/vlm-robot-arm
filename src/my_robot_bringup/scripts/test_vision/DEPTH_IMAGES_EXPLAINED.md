# Understanding Depth Images

## Files Created by Test 1

### 1. `rgb_sample.jpg` ✓ IMPORTANT
**What it is:** Regular color image from the camera  
**Used for:** Object detection, segmentation, visual processing  
**Viewable:** Yes, opens in any image viewer  
**For your project:** This is what Grounding DINO and SAM use

---

### 2. `depth_sample_colorized.jpg` ✓ FOR VERIFICATION ONLY
**What it is:** Human-readable visualization of depth  
**Colors mean:**
- Dark blue/purple = Very close (0.3-1m)
- Blue = Close (1-2m)
- Cyan = Medium (2-3m)  
- Green/Yellow = Far (3-5m)
- Red = Very far (5m+)
- Black = No depth data (NaN or too far)

**Used for:** Visual verification that depth is working  
**For your project:** NOT used in actual code, just for testing  
**Your image shows:** Good depth! Blue wall close, cyan/green areas farther

---

### 3. `depth_sample_mm.png` - WHY IT'S BLACK
**What it is:** 16-bit depth data in millimeters  
**Why it looks black:** 16-bit images need special viewers  
**Viewable:** Only in specialized tools (ImageJ, GIMP with 16-bit support)  
**For your project:** NOT needed - your code uses the raw float data directly

This file IS correct - it just can't be viewed in regular image viewers because:
- Regular viewers expect 8-bit (0-255)
- This is 16-bit (0-65535)  
- Values like 1500 (1.5m) appear as near-black in 8-bit view

---

### 4. `depth_sample_normalized.png` ✓ FOR VERIFICATION
**What it is:** Grayscale depth visualization  
**Used for:** Another way to verify depth is working  
**For your project:** NOT used in code, just for testing

---

### 5. `depth_sample_raw.npy` ✓ IMPORTANT FOR PROJECT
**What it is:** Raw numpy array of depth values in meters (32-bit float)  
**Format:** float32 array, shape (height, width)  
**Values:** Actual distances in meters (e.g., 1.5 = 1.5 meters)  
**Used for:** This is what your code actually uses!  
**For your project:** Your Python code reads depth like this:
```python
depth = cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
# depth is now a float32 array in meters
# Example: depth[100, 200] might be 1.45 (1.45 meters away)
```

---

## What Your Project Actually Uses

Your `tabletop_handybot_node.py` uses:

1. **RGB Image (8-bit color)**
   - For object detection (Grounding DINO)
   - For segmentation masks (SAM)
   - Format: `bgr8` or `rgb8`

2. **Depth Data (32-bit float array)**
   - Raw depth values in meters
   - Used to convert 2D pixel → 3D point cloud
   - Format: `32FC1` (32-bit float, 1 channel)
   - This is what you get from `imgmsg_to_cv2(msg, 'passthrough')`

---

## Summary for Your Project

✅ **Files you need to verify working:**
- `rgb_sample.jpg` - Should show clear color image
- `depth_sample_colorized.jpg` - Should show colors (not all black)
- If colorized shows colors → depth is working!

❌ **Files you can ignore:**
- `depth_sample_mm.png` - Will appear black in normal viewers (this is OK!)
- `depth_sample_normalized.png` - Just another visualization

🔧 **What your code actually uses:**
- ROS depth topic: `/zedm/zed_node/depth/depth_registered`
- Format: 32-bit float array in meters
- Your code converts this to point clouds for pick-and-place

---

## Your Test Result: ✅ PASS

Based on your colorized depth image:
- Camera is working correctly ✓
- Depth sensing is functional ✓  
- Good depth measurements visible ✓
- Ready for Tests 2 and 3 ✓

The black `depth_sample_mm.png` is **normal and expected** - it's a technical limitation of how 16-bit images display, not a problem with your camera!
