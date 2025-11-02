#!/usr/bin/env python3
"""
Test 2: Object Detection with Grounding DINO

This script tests zero-shot object detection using Grounding DINO.

Expected behavior:
- Grounding DINO model loads successfully
- Can detect specified objects from camera feed
- Bounding boxes are reasonable
- Confidence scores are reported
- Annotated images saved

Usage:
    1. Make sure ZED camera is running:
       ros2 launch zed_wrapper zedm.launch.py
    
    2. Activate the virtual environment (if needed):
       source src/tabletop-handybot/venv/bin/activate
    
    3. Run this test:
       python3 src/my_robot_bringup/scripts/test_vision/test2_object_detection.py
    
    4. Test with custom objects:
       python3 src/my_robot_bringup/scripts/test_vision/test2_object_detection.py --objects "cup,bottle,pen"
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
    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "groundingdino_swint_ogc.pth")
except ImportError as e:
    print(f"Error importing Grounding DINO: {e}")
    print("\nMake sure you've set up Grounded-Segment-Anything:")
    print("  cd ~/vlm_robot_ws")
    print("  source src/tabletop-handybot/venv/bin/activate")
    sys.exit(1)

# Detection parameters
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


class ObjectDetectionTest(Node):
    def __init__(self, object_classes, num_samples=3):
        super().__init__('object_detection_test')
        
        self.cv_bridge = CvBridge()
        self.object_classes = object_classes
        self.num_samples = num_samples
        self.samples_collected = 0
        
        # Create output directory
        self.output_dir = 'test_outputs/test2_detection'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Grounding DINO model
        self.get_logger().info('Loading Grounding DINO model...')
        try:
            self.grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH,
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            )
            self.get_logger().info('✓ Grounding DINO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'✗ Failed to load Grounding DINO model: {e}')
            raise
        
        # Subscribe to camera RGB topic
        self.rgb_sub = self.create_subscription(
            Image,
            '/zedm/zed_node/rgb/image_rect_color',
            self.rgb_callback,
            10
        )
        
        self.get_logger().info(f'Object Detection Test initialized')
        self.get_logger().info(f'Target objects: {", ".join(self.object_classes)}')
        self.get_logger().info(f'Will collect {self.num_samples} samples')
        self.get_logger().info(f'Saving outputs to: {self.output_dir}')
        self.get_logger().info('Waiting for camera images...')
        
        # Timer for periodic processing
        self.timer = self.create_timer(2.0, self.process_frame)
        self.last_rgb_msg = None
        self.start_time = time.time()
        
    def rgb_callback(self, msg):
        """Store latest RGB image"""
        self.last_rgb_msg = msg
    
    def process_frame(self):
        """Process a frame for object detection"""
        if self.last_rgb_msg is None:
            elapsed = time.time() - self.start_time
            if elapsed > 10:
                self.get_logger().error('✗ No RGB images received after 10 seconds')
                self.get_logger().error('Make sure ZED camera is running!')
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
        
        # Run object detection
        detections = self.detect_objects(cv_image)
        
        # Save annotated image
        self.save_annotated_image(cv_image, detections, self.samples_collected)
        
        self.samples_collected += 1
    
    def detect_objects(self, image: np.ndarray):
        """Run Grounding DINO object detection"""
        self.get_logger().info('Running Grounding DINO detection...')
        
        try:
            # Run detection
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=self.object_classes,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            
            # Apply NMS (Non-Maximum Suppression)
            if len(detections.xyxy) > 0:
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    NMS_THRESHOLD,
                ).numpy().tolist()
                
                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]
            
            # Report results
            num_detections = len(detections.xyxy)
            self.get_logger().info(f'✓ Detected {num_detections} object(s)')
            
            if num_detections > 0:
                for i, (bbox, confidence, class_id) in enumerate(
                    zip(detections.xyxy, detections.confidence, detections.class_id)):
                    obj_name = self.object_classes[class_id]
                    self.get_logger().info(
                        f'  [{i}] {obj_name}: confidence={confidence:.3f}, '
                        f'bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]'
                    )
            else:
                self.get_logger().warn('  No objects detected with current thresholds')
                self.get_logger().info(f'  Try: Lower object names, better lighting, or adjust thresholds')
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f'✗ Detection failed: {e}')
            import traceback
            traceback.print_exc()
            return sv.Detections.empty()
    
    def save_annotated_image(self, image: np.ndarray, detections: sv.Detections, sample_num: int):
        """Save image with detection annotations"""
        try:
            # Create annotated image
            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{self.object_classes[class_id]} {confidence:.2f}"
                for confidence, class_id in zip(detections.confidence, detections.class_id)
            ]
            
            annotated_image = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            
            # Save annotated image
            output_path = os.path.join(
                self.output_dir, 
                f'detection_sample_{sample_num}.jpg'
            )
            cv2.imwrite(output_path, annotated_image)
            self.get_logger().info(f'Saved: {output_path}')
            
            # Also save original for comparison
            orig_path = os.path.join(
                self.output_dir, 
                f'original_sample_{sample_num}.jpg'
            )
            cv2.imwrite(orig_path, image)
            
        except Exception as e:
            self.get_logger().error(f'Error saving annotated image: {e}')
    
    def print_results(self):
        """Print final test results"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('TEST 2: OBJECT DETECTION - RESULTS')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Target objects: {", ".join(self.object_classes)}')
        self.get_logger().info(f'Samples processed: {self.samples_collected}')
        self.get_logger().info(f'Detection thresholds:')
        self.get_logger().info(f'  - Box threshold: {BOX_THRESHOLD}')
        self.get_logger().info(f'  - Text threshold: {TEXT_THRESHOLD}')
        self.get_logger().info(f'  - NMS threshold: {NMS_THRESHOLD}')
        self.get_logger().info(f'\n✓✓✓ TEST 2 COMPLETED ✓✓✓')
        self.get_logger().info(f'\nAnnotated images saved to: {self.output_dir}/')
        self.get_logger().info('\nNext steps:')
        self.get_logger().info('  1. Review detection_sample_*.jpg files')
        self.get_logger().info('  2. Verify bounding boxes are accurate')
        self.get_logger().info('  3. Check confidence scores are reasonable')
        self.get_logger().info('  4. If no detections, try different objects or lighting')
        self.get_logger().info('='*60 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Test Grounding DINO object detection')
    parser.add_argument(
        '--objects',
        type=str,
        default='cup,bottle,marker,pen',
        help='Comma-separated list of object classes to detect'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=3,
        help='Number of sample frames to process'
    )
    
    args = parser.parse_args()
    object_classes = [obj.strip() for obj in args.objects.split(',')]
    
    rclpy.init()
    
    print("\n" + "="*60)
    print("TEST 2: OBJECT DETECTION WITH GROUNDING DINO")
    print("="*60)
    print("\nPrerequisites:")
    print("  - ZED camera running: ros2 launch zed_wrapper zedm.launch.py")
    print("  - Virtual environment activated")
    print("  - Grounded-Segment-Anything setup complete")
    print("\nThis test will:")
    print("  - Load Grounding DINO model")
    print(f"  - Detect objects: {', '.join(object_classes)}")
    print(f"  - Process {args.samples} sample frames")
    print("  - Save annotated images with bounding boxes")
    print("="*60 + "\n")
    
    try:
        node = ObjectDetectionTest(object_classes, args.samples)
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
