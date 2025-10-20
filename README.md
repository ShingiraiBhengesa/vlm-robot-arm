# VLM Robot Workspace

A ROS2-based AI-powered robotic arm assistant that integrates Vision Language Models (VLM), Large Language Models (LLM), and computer vision to enable natural language-controlled manipulation of tabletop objects. This project uses a Lynxmotion LSS robotic arm and ZED stereo camera for precise pick-and-place operations, allowing the robot to understand and execute verbal commands for various tasks.

## Features

- **Natural Language Interaction**: Control the robot using spoken commands or text prompts
- **Vision-Based Object Detection**: Real-time object recognition and segmentation using state-of-the-art computer vision
- **Precise Manipulation**: 4DoF Lynxmotion LSS arm with servo gripper for accurate pick-and-place operations
- **Depth Perception**: ZED stereo camera provides reliable RGB-D data for 3D manipulation
- **Motion Planning**: Integration with MoveIt2 for collision-free trajectory planning
- **Modular Architecture**: ROS2-based design allowing easy extension and customization

## System Architecture

```
[Speech/Microphone] --> [Whisper/Llama] --> [LLM Processing] --> [Task Planning]
                                      |
                                      v
[ZED Stereo Camera] --> [Image Processing] --> [Object Detection & Segmentation] --> [Pose Estimation]
                                      |
                                      v
[Lynxmotion LSS Arm] <-- [MoveIt2 Motion Planning] <-- [Trajectory Execution]
```

## Technology Used

### Hardware
- **Lynxmotion LSS Robotic Arm**: 5DoF Lynxmotionarm with servo gripper
- **ZED Mini Stereo Camera**: High-quality RGB-D camera with NVIDIA Jetson or PC compatibility
- **Computer**: Ubuntu 22.04 with ROS2 Humble/Or Iron, NVIDIA GPU for CUDA acceleration

### Software Components

- **ROS 2**: Robotic middleware for component integration
- **MoveIt 2**: Motion planning framework for robotic manipulation
- **ZED SDK 4.0.5**: Camera drivers and SDK for depth sensing
- **Grounding DINO**: Zero-shot object detection
- **Segment Anything (SAM)**: Advanced object segmentation
- **OpenAI Whisper**: Speech-to-text for natural language input
- **OpenAI GPT Models**: Large language model for task understanding and planning
- **OpenCV**: Computer vision processing
- **PCL (Point Cloud Library)**: 3D point cloud processing

## Prerequisites

- **Operating System**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill or Iron Irwini (tested on Iron)
- **ZED SDK**: Version 4.0.5 or compatible
- **CUDA**: 11.8 with cuDNN and TensorRT for GPU acceleration
- **Python**: 3.10 or later with virtual environment support
- **Hardware**: Lynxmotion LSS arm, ZED camera, microphone (optional)

### System Requirements
- NVIDIA GPU with CUDA support (recommended: GTX 1060 or better)
- 16GB RAM minimum
- USB 3.0 ports for camera and arm control
- 8 CPU cores recommended

## Installation

### 1. Clone the Workspace
```bash
cd ~
git clone https://github.com/your-repo/vlm_robot_ws.git
cd vlm_robot_ws
```

### 2. Install ROS 2 Dependencies
```bash
sudo apt update
sudo apt install -y ros-iron-desktop ros-iron-moveit2 ros-iron-navigation2 ros-iron-perception-pcl ros-iron-vision-opencv
sudo apt install -y python3-colcon-common-extensions python3-vcstool
```

### 3. Install ZED SDK
Download and install ZED SDK 4.0.5 for Ubuntu 22.04:
```bash
wget -O ZED_SDK.run https://download.stereolabs.com/zedsdk/4.0/ZED_SDK_Ubuntu22_cuda11.8_v4.0.5.run
chmod +x ZED_SDK.run
./ZED_SDK.run
```

### 4. Install Python Dependencies
```bash
python3 -m pip install --upgrade pip
pip3 install virtualenv

# Create virtual environment for AI components
cd src/tabletop-handybot
python3 -m venv venv
source venv/bin/activate

# Install AI/ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper openai grounding-dino-py segment-anything
pip install -r requirements.txt
```

### 5. Build ROS Workspace
```bash
cd ~/vlm_robot_ws
source /opt/ros/iron/setup.bash
rosdep install --from-path src --ignore-src -r -y
colcon build --symlink-install
```

### 6. Setup Lynxmotion Arm
1. Update servo firmware using LSS Config software
2. Configure servo IDs and baud rate (921600)
3. Set gyre direction to CCW (-1) for all servos
4. Perform initial calibration using FlowArm app

## Usage

### Launch the System
```bash
cd ~/vlm_robot_ws
source install/setup.bash

# Launch complete system
ros2 launch my_robot_bringup vlm_pick_and_place.launch.py
```

This will start:
- ZED camera node publishing RGB and depth images
- Lynxmotion arm controller with MoveIt2
- AI processing node for object detection and task planning
- Speech recognition (if microphone available)

### Basic Operation

#### Text Commands
```bash
# Send text prompt
ros2 topic pub --once /prompt std_msgs/msg/String "data: 'pick up the red marker and put it in the box'"

# Listen for voice command
ros2 topic pub --once /listen std_msgs/msg/Empty "{}"
```

#### Arm Control
```bash
# Activate arm servos (5DoF arm)
ros2 topic pub --once /effort_controller/commands std_msgs/msg/Float64MultiArray "data: [-6.8, -6.8, -6.8, -6.8, -6.8]"

# Make arm limp
ros2 topic pub --once /effort_controller/commands std_msgs/msg/Float64MultiArray "data: [0, 0, 0, 0, 0]"
```

### Calibration and Setup

#### Hand-Eye Calibration
Perform camera-to-arm calibration for accurate manipulation:
```bash
ros2 launch my_robot_bringup hand_eye_calibration.launch.py
```

#### Object Training (Optional)
For improved object recognition, you can fine-tune the detection models with custom objects.

## Configuration

### Camera Settings
Modify `zed_wrapper/config/zedm.yaml` for camera parameters:
- Resolution: HD720, HD1080, HD2K
- Frame rate: 15, 30, 60 FPS
- Depth range and quality settings

### Arm Configuration
Adjust `lss_arm_moveit/config/moveit_controllers.yaml` for:
- Joint limits and velocities
- Controller gains
- Motion planning parameters

### AI Model Settings
Configure AI components in `tabletop_handybot/config/`:
- OpenAI API keys
- Model parameters (temperature, max tokens)
- Detection confidence thresholds

## Examples

### Pick and Place Demo
```bash
# Start the system
ros2 launch my_robot_bringup vlm_pick_and_place.launch.py

# In another terminal
ros2 topic pub --once /prompt std_msgs/msg/String "data: 'grab the blue cup and place it next to the red marker'"
```

### Object Sorting
```bash
ros2 topic pub --once /prompt std_msgs/msg/String "data: 'sort all the colored blocks by color into separate piles'"
```

### Voice Control
```bash
# Ensure microphone is connected
ros2 topic pub --once /listen std_msgs/msg/Empty "{}"
# Then speak: "Pick up the pen and put it on the desk"
```

## Troubleshooting

### Common Issues

**Camera Not Detected**
- Check USB connections
- Run ZED diagnostic: `python3 -c "import pyzed.sl as sl; print(sl.Camera.get_device_list())"`
- Ensure ZED SDK version matches workspace requirements

**Arm Not Moving**
- Verify servo IDs and baud rate configuration
- Check USB permissions: `sudo chmod 666 /dev/ttyUSB*`
- Ensure servos are activated with proper effort values

**AI Models Not Loading**
- Verify CUDA installation: `nvidia-smi`
- Check virtual environment activation
- Ensure sufficient GPU memory (4GB+ recommended)

**ROS2 Communication Issues**
```bash
# Check active topics
ros2 topic list

# Verify node status
ros2 node list

# Check service availability
ros2 service list
```

### Performance Optimization
- Use SSD storage for better I/O performance
- Allocate GPU memory for AI models
- Tune ROS2 QoS settings for real-time performance
- Monitor CPU/GPU usage with `htop` and `nvidia-smi`

## Development

### Adding New Features
- Follow ROS2 package structure
- Add configuration files to `config/` directories
- Document new launch files and parameters
- Update README with new capabilities

### Testing
```bash
# Run unit tests
colcon test --packages-select my_robot_bringup
colcon test-result --verbose

# Integration testing
ros2 launch my_robot_bringup integration_test.launch.py
```

### Code Style
- Follow ROS2 Python style guidelines
- Use black for code formatting
- Add type hints and documentation strings

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-functionality`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-functionality`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on [Tabletop HandyBot](https://github.com/your-repo/tabletop-handybot) project
- Lynxmotion LSS ROS2 packages from [LSS-ROS2-Arms](https://github.com/Lynxmotion/LSS-ROS2-Arms)
- ZED ROS2 wrapper from [zed-ros2-wrapper](https://github.com/stereolabs/zed-ros2-wrapper)
- AI/ML models from Grounding DINO and Segment Anything projects

## Contact

- **Maintainer**: Shingirai Bhengesa
- **Email**: shingiebhengesa@gmail.com
- **GitHub**: [your-username](https://github.com/your-username)

## Version History

- **v1.0.0**: Initial release with basic pick-and-place functionality
- **v1.1.0**: Added voice control and improved object detection
- **v1.2.0**: Enhanced motion planning and added 5DoF arm support
