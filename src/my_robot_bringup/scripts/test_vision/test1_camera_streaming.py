#!/usr/bin/env python3
"""
Test 1: ZED Camera Streaming Verification

This script tests that the ZED camera is properly publishing RGB and depth images.

Expected behavior:
- ZED camera node starts successfully
- RGB images published to /zedm/zed_node/rgb/image_rect_color
- Depth images published to /zedm/zed_node/depth/depth_registered
- Images saved to test_outputs/ directory

Usage:
    1. Launch ZED camera separately:
       ros2 launch zed_wrapper zedm.launch.py
    
    2. Run this test:
       python3 src/my_robot_bringup/scripts/test_vision/test1_camera_streaming.py
"""

import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraStreamTest(Node):
    def __init__(self):
        super().__init__('camera_stream_test')
        
        self.cv_bridge = CvBridge()
        self.rgb_received = False
        self.depth_received = False
        self.rgb_count = 0
        self.depth_count = 0
        
        # Create output directory
        self.output_dir = 'test_outputs/test1_camera'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subscribe to camera topics
        self.rgb_sub = self.create_subscription(
            Image,
            '/zedm/zed_node/rgb/image_rect_color',
            self.rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/zedm/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )
        
        self.get_logger().info('Camera Stream Test initialized')
        self.get_logger().info(f'Saving outputs to: {self.output_dir}')
        self.get_logger().info('Waiting for camera topics...')
        
        # Timer to check status
        self.timer = self.create_timer(2.0, self.check_status)
        self.start_time = time.time()
        
    def rgb_callback(self, msg):
        """Callback for RGB images"""
        self.rgb_count += 1
        
        if not self.rgb_received:
            self.rgb_received = True
            self.get_logger().info(f'✓ RGB topic active!')
            self.get_logger().info(f'  - Resolution: {msg.width}x{msg.height}')
            self.get_logger().info(f'  - Encoding: {msg.encoding}')
            
            # Save first RGB image
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                output_path = os.path.join(self.output_dir, 'rgb_sample.jpg')
                cv2.imwrite(output_path, cv_image)
                self.get_logger().info(f'  - Saved sample: {output_path}')
            except Exception as e:
                self.get_logger().error(f'Error saving RGB image: {e}')
    
    def depth_callback(self, msg):
        """Callback for depth images"""
        self.depth_count += 1
        
        if not self.depth_received:
            self.depth_received = True
            self.get_logger().info(f'✓ Depth topic active!')
            self.get_logger().info(f'  - Resolution: {msg.width}x{msg.height}')
            self.get_logger().info(f'  - Encoding: {msg.encoding}')
            
            # Save first depth image
            try:
                cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                
                self.get_logger().info(f'  - Depth dtype: {cv_depth.dtype}, shape: {cv_depth.shape}')
                
                # Check for NaN values (common issue with ZED)
                nan_count = np.isnan(cv_depth).sum()
                inf_count = np.isinf(cv_depth).sum()
                zero_count = (cv_depth == 0).sum()
                valid_depth = cv_depth[np.isfinite(cv_depth) & (cv_depth > 0)]
                
                self.get_logger().info(f'  - Total pixels: {cv_depth.size}')
                self.get_logger().info(f'  - NaN pixels: {nan_count} ({100*nan_count/cv_depth.size:.1f}%)')
                self.get_logger().info(f'  - Inf pixels: {inf_count} ({100*inf_count/cv_depth.size:.1f}%)')
                self.get_logger().info(f'  - Zero pixels: {zero_count} ({100*zero_count/cv_depth.size:.1f}%)')
                self.get_logger().info(f'  - Valid pixels: {len(valid_depth)} ({100*len(valid_depth)/cv_depth.size:.1f}%)')
                
                if len(valid_depth) > 0:
                    self.get_logger().info(f'  - Depth range: {valid_depth.min():.3f}m to {valid_depth.max():.3f}m')
                    self.get_logger().info(f'  - Mean depth: {valid_depth.mean():.3f}m')
                else:
                    self.get_logger().warn('  ⚠ No valid depth measurements!')
                    if nan_count > cv_depth.size * 0.9:
                        self.get_logger().warn('  ⚠ Most pixels are NaN - Camera cannot compute depth!')
                        self.get_logger().warn('  → Solutions:')
                        self.get_logger().warn('     1. Point camera at textured surface (not blank wall)')
                        self.get_logger().warn('     2. Ensure adequate lighting')
                        self.get_logger().warn('     3. Keep objects 0.3m - 5m from camera')
                        self.get_logger().warn('     4. Remove lens cap if present')
                
                # Create depth visualization (only keep what's useful for verification)
                if len(valid_depth) > 0:
                    # Create colored depth map for human verification
                    depth_viz = cv_depth.copy()
                    # Replace NaN/invalid with max valid depth for better visualization
                    depth_viz[~np.isfinite(cv_depth)] = valid_depth.max()
                    depth_viz[cv_depth == 0] = valid_depth.max()
                    
                    # Normalize and apply color map
                    depth_normalized = cv2.normalize(depth_viz, None, 0, 255, cv2.NORM_MINMAX)
                    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                    
                    # Make invalid pixels black
                    invalid_mask = (~np.isfinite(cv_depth)) | (cv_depth == 0)
                    depth_colored[invalid_mask] = [0, 0, 0]
                    
                    output_path = os.path.join(self.output_dir, 'depth_sample_colorized.jpg')
                    cv2.imwrite(output_path, depth_colored)
                    
                    self.get_logger().info(f'  - Saved depth visualization: depth_sample_colorized.jpg')
                    self.get_logger().info(f'    (Blue=close, Cyan=medium, Green/Yellow=far, Black=no data)')
                else:
                    self.get_logger().error('  - Cannot create visualization: no valid depth data')
                    
            except Exception as e:
                self.get_logger().error(f'Error saving depth image: {e}')
    
    def check_status(self):
        """Periodically check test status"""
        elapsed = time.time() - self.start_time
        
        if elapsed > 15 and not (self.rgb_received and self.depth_received):
            self.get_logger().warn('Timeout waiting for camera topics')
            if not self.rgb_received:
                self.get_logger().error('✗ RGB topic not receiving data')
            if not self.depth_received:
                self.get_logger().error('✗ Depth topic not receiving data')
            self.print_results()
            rclpy.shutdown()
        
        if self.rgb_received and self.depth_received:
            self.get_logger().info(f'\n--- Status Update (t={elapsed:.1f}s) ---')
            self.get_logger().info(f'RGB messages: {self.rgb_count} (~{self.rgb_count/elapsed:.1f} Hz)')
            self.get_logger().info(f'Depth messages: {self.depth_count} (~{self.depth_count/elapsed:.1f} Hz)')
            
            if elapsed > 5:  # Run for at least 5 seconds after both topics active
                self.print_results()
                rclpy.shutdown()
    
    def print_results(self):
        """Print final test results"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('TEST 1: CAMERA STREAMING - RESULTS')
        self.get_logger().info('='*60)
        
        if self.rgb_received:
            self.get_logger().info('✓ RGB stream: PASS')
        else:
            self.get_logger().error('✗ RGB stream: FAIL')
        
        if self.depth_received:
            self.get_logger().info('✓ Depth stream: PASS')
        else:
            self.get_logger().error('✗ Depth stream: FAIL')
        
        if self.rgb_received and self.depth_received:
            self.get_logger().info('\n✓✓✓ TEST 1 PASSED ✓✓✓')
            self.get_logger().info(f'\nSample images saved to: {self.output_dir}/')
            self.get_logger().info('  - rgb_sample.jpg')
            self.get_logger().info('  - depth_sample.png')
            self.get_logger().info('  - depth_sample_colorized.jpg')
        else:
            self.get_logger().error('\n✗✗✗ TEST 1 FAILED ✗✗✗')
            self.get_logger().error('\nTroubleshooting:')
            self.get_logger().error('  1. Is ZED camera connected? Check: lsusb | grep 2b03')
            self.get_logger().error('  2. Is ZED wrapper running? Check: ros2 node list | grep zed')
            self.get_logger().error('  3. Are topics publishing? Check: ros2 topic list | grep zed')
            self.get_logger().error('  4. Check ZED diagnostics: ZED_Diagnostic')
        
        self.get_logger().info('='*60 + '\n')


def main():
    rclpy.init()
    
    print("\n" + "="*60)
    print("TEST 1: ZED CAMERA STREAMING VERIFICATION")
    print("="*60)
    print("\nMake sure ZED camera is launched first:")
    print("  ros2 launch zed_wrapper zedm.launch.py")
    print("\nThis test will:")
    print("  - Verify RGB and depth topics are publishing")
    print("  - Save sample images")
    print("  - Report image statistics")
    print("="*60 + "\n")
    
    node = CameraStreamTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Test interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
