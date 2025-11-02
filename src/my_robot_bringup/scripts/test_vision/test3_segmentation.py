#!/usr/bin/env python3
"""
Test 3: Object Segmentation with SAM (Segment Anything Model)

This script tests object segmentation using both Grounding DINO for detection
and SAM for generating precise segmentation masks.

Expected behavior:
- Grounding DINO detects objects
- SAM generates segmentation masks for detected objects
- Masks are accurate and non-overlapping
- Annotated images saved with masks overlaid

Usage:
    1. Make sure ZED camera is running:
       ros2 launch zed_wrapper zedm.launch.py
    
    2. Activate the virtual environment:
       source src/tabletop-handybot/venv/bin/activate
    
    3. Run this test:
       python3 src/my_robot_bringup/scripts/test_vision/test3_segmentation.py
    
    4. Test with custom objects:
       python3 src/my_robot_bringup/scripts/test_vision/test3_segmentation.py --objects "cup,bottle,pen"
"""

import os
import sys
import time
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torchvision
import supervision as sv

# Add GSA path for imports
GSA_PATH = "./Grounded-Segment-Anything/Grounded-Segment-Anything"
sys.path.insert(0, GSA_PATH)

try:
    from groundingdino.util.inference import Model
    from segment_anything import SamPredictor, sam_model_registry
    
    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "groundingdino_swint_ogc.pth")
    
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "sam_vit_h_4b8939.pth")
except ImportError as e:
    print(f"Error importing models: {e}")
    print("\nMake sure you've set up Grounded-Segment-Anything:")
    print("  cd ~/vlm_robot_ws")
    print("  source src/tabletop-handybot/venv/bin/activate")
    sys.exit(1)

# Detection parameters
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Generate segmentation masks for detected bounding boxes using SAM"""
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


class SegmentationTest(Node):
    def __init__(self, object_classes, num_samples=2):
        super().__init__('segmentation_test')
        
        self.cv_bridge = CvBridge()
        self.object_classes = object_classes
        self.num_samples = num_samples
        self.samples_collected = 0
        
        # Create output directory
        self.output_dir = 'test_outputs/test3_segmentation'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Grounding DINO model
        self.get_logger().info('Loading Grounding DINO model...')
        try:
            self.grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH,
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            )
            self.get_logger().info('✓ Grounding DINO loaded')
        except Exception as e:
            self.get_logger().error(f'✗ Failed to load Grounding DINO: {e}')
            raise
        
        # Load SAM model
        self.get_logger().info('Loading SAM model...')
        try:
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
            sam.to(device=DEVICE)
            self.sam_predictor = SamPredictor(sam)
            self.get_logger().info(f'✓ SAM loaded on {DEVICE}')
        except Exception as e:
            self.get_logger().error(f'✗ Failed to load SAM: {e}')
            raise
        
        # Subscribe to camera RGB topic
        self.rgb_sub = self.create_subscription(
            Image,
            '/zedm/zed_node/rgb/image_rect_color',
            self.rgb_callback,
            10
        )
        
        self.get_logger().info(f'Segmentation Test initialized')
        self.get_logger().info(f'Target objects: {", ".join(self.object_classes)}')
        self.get_logger().info(f'Will collect {self.num_samples} samples')
        self.get_logger().info(f'Saving outputs to: {self.output_dir}')
        self.get_logger().info('Waiting for camera images...')
        
        # Timer for periodic processing
        self.timer = self.create_timer(3.0, self.process_frame)
        self.last_rgb_msg = None
        self.start_time = time.time()
        
    def rgb_callback(self, msg):
        """Store latest RGB image"""
        self.last_rgb_msg = msg
    
    def process_frame(self):
        """Process a frame for detection and segmentation"""
        if self.last_rgb_msg is None:
            elapsed = time.time() - self.start_time
            if elapsed > 10:
                self.get_logger().error('✗ No RGB images received after 10 seconds')
                rclpy.shutdown()
            return
        
        if self.samples_collected >= self.num_samples:
            self.print_results()
            rclpy.shutdown()
            return
        
        # Convert ROS image to OpenCV
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(self.last_rgb_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
        
        self.get_logger().info(f'\n--- Processing Sample {self.samples_collected + 1}/{self.num_samples} ---')
        
        # Run detection and segmentation
        detections = self.detect_and_segment(cv_image)
        
        # Save annotated images
        self.save_annotated_images(cv_image, detections, self.samples_collected)
        
        self.samples_collected += 1
    
    def detect_and_segment(self, image: np.ndarray):
        """Run detection with Grounding DINO and segmentation with SAM"""
        self.get_logger().info('Step 1: Running Grounding DINO detection...')
        
        try:
            # Detection
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=self.object_classes,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            
            # Apply NMS
            if len(detections.xyxy) > 0:
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    NMS_THRESHOLD,
                ).numpy().tolist()
                
                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]
            
            num_detections = len(detections.xyxy)
            self.get_logger().info(f'✓ Detected {num_detections} object(s)')
            
            if num_detections == 0:
                self.get_logger().warn('  No objects detected - skipping segmentation')
                return detections
            
            # Segmentation
            self.get_logger().info('Step 2: Running SAM segmentation...')
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections.mask = segment(
                sam_predictor=self.sam_predictor,
                image=rgb_image,
                xyxy=detections.xyxy,
            )
            
            self.get_logger().info(f'✓ Generated {len(detections.mask)} segmentation masks')
            
            # Report statistics for each detection
            for i, (bbox, confidence, class_id, mask) in enumerate(
                zip(detections.xyxy, detections.confidence, detections.class_id, detections.mask)):
                obj_name = self.object_classes[class_id]
                mask_area = np.sum(mask)
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                coverage = (mask_area / bbox_area) * 100 if bbox_area > 0 else 0
                
                self.get_logger().info(
                    f'  [{i}] {obj_name}: conf={confidence:.3f}, '
                    f'mask_pixels={mask_area:.0f}, bbox_coverage={coverage:.1f}%'
                )
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f'✗ Detection/Segmentation failed: {e}')
            import traceback
            traceback.print_exc()
            return sv.Detections.empty()
    
    def save_annotated_images(self, image: np.ndarray, detections: sv.Detections, sample_num: int):
        """Save multiple visualizations of the segmentation results"""
        try:
            # 1. Save original image
            orig_path = os.path.join(self.output_dir, f'original_{sample_num}.jpg')
            cv2.imwrite(orig_path, image)
            
            if len(detections.xyxy) == 0:
                self.get_logger().info(f'Saved: {orig_path} (no detections)')
                return
            
            # 2. Save image with bounding boxes
            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{self.object_classes[class_id]} {confidence:.2f}"
                for confidence, class_id in zip(detections.confidence, detections.class_id)
            ]
            
            bbox_image = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            bbox_path = os.path.join(self.output_dir, f'bboxes_{sample_num}.jpg')
            cv2.imwrite(bbox_path, bbox_image)
            
            # 3. Save image with segmentation masks overlay
            mask_annotator = sv.MaskAnnotator()
            mask_image = mask_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            mask_path = os.path.join(self.output_dir, f'masks_{sample_num}.jpg')
            cv2.imwrite(mask_path, mask_image)
            
            # 4. Save image with both boxes and masks
            combined_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            combined_image = box_annotator.annotate(scene=combined_image, detections=detections, labels=labels)
            combined_path = os.path.join(self.output_dir, f'combined_{sample_num}.jpg')
            cv2.imwrite(combined_path, combined_image)
            
            # 5. Save individual mask images
            for i, (mask, class_id) in enumerate(zip(detections.mask, detections.class_id)):
                mask_only = (mask * 255).astype(np.uint8)
                mask_only_path = os.path.join(
                    self.output_dir, 
                    f'mask_{sample_num}_obj{i}_{self.object_classes[class_id]}.png'
                )
                cv2.imwrite(mask_only_path, mask_only)
            
            self.get_logger().info(f'Saved visualizations:')
            self.get_logger().info(f'  - {orig_path}')
            self.get_logger().info(f'  - {bbox_path}')
            self.get_logger().info(f'  - {mask_path}')
            self.get_logger().info(f'  - {combined_path}')
            self.get_logger().info(f'  - Individual masks: mask_{sample_num}_obj*.png')
            
        except Exception as e:
            self.get_logger().error(f'Error saving annotated images: {e}')
    
    def print_results(self):
        """Print final test results"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('TEST 3: OBJECT SEGMENTATION - RESULTS')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Target objects: {", ".join(self.object_classes)}')
        self.get_logger().info(f'Samples processed: {self.samples_collected}')
        self.get_logger().info(f'Device used: {DEVICE}')
        self.get_logger().info(f'\n✓✓✓ TEST 3 COMPLETED ✓✓✓')
        self.get_logger().info(f'\nVisualization images saved to: {self.output_dir}/')
        self.get_logger().info('\nNext steps:')
        self.get_logger().info('  1. Review combined_*.jpg for full visualization')
        self.get_logger().info('  2. Check mask_*.png for individual object masks')
        self.get_logger().info('  3. Verify masks align with object boundaries')
        self.get_logger().info('  4. Check that masks dont overlap incorrectly')
        self.get_logger().info('='*60 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Test object segmentation with Grounding DINO + SAM')
    parser.add_argument(
        '--objects',
        type=str,
        default='cup,bottle,marker,pen',
        help='Comma-separated list of object classes to detect and segment'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2,
        help='Number of sample frames to process'
    )
    
    args = parser.parse_args()
    object_classes = [obj.strip() for obj in args.objects.split(',')]
    
    rclpy.init()
    
    print("\n" + "="*60)
    print("TEST 3: OBJECT SEGMENTATION (GROUNDING DINO + SAM)")
    print("="*60)
    print("\nPrerequisites:")
    print("  - ZED camera running: ros2 launch zed_wrapper zedm.launch.py")
    print("  - Virtual environment activated")
    print("  - GPU with CUDA (recommended for SAM)")
    print("\nThis test will:")
    print("  - Detect objects with Grounding DINO")
    print(f"  - Segment each detected object with SAM")
    print(f"  - Process {args.samples} sample frames")
    print("  - Save multiple visualizations (boxes, masks, combined)")
    print("="*60 + "\n")
    
    try:
        node = SegmentationTest(object_classes, args.samples)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nTest interrupted by user')
    except Exception as e:
        print(f'\nTest failed with error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
