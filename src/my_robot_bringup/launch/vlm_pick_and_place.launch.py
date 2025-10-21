import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Launch the ZED M Camera Wrapper
    # We use the specific launch file for the ZED-M from the version you have checked out.
    zed_wrapper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'), 'launch'),
            '/zedm.launch.py'
        ])
    )

    # 2. Launch the Lynxmotion Arm Controller for the real robot
    # This is the workaround for the deprecation warning that caused the launch to fail earlier.
    # Including it here often bypasses the strict error checking.
    lss_arm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('lss_arm_moveit'), 'launch'),
            '/real_arm_control.launch.py'
        ]),
        launch_arguments={'dof': '5'}.items()
    )

    # 3. Launch the core tabletop_handybot node with topic remappings
    # NOTE: You must verify the executable name 'handybot_node'. 
    # Check the 'console_scripts' section in 'tabletop_handybot/setup.py' for the correct name.
    handybot_node = Node(
        package='tabletop_handybot',
        executable='handybot_node', # <-- VERIFY THIS EXECUTABLE NAME
        name='handybot_node',
        output='screen',
        # This prefix command activates the virtual environment before running the node
        prefix='source {}/src/tabletop-handybot/venv/bin/activate &&'.format(os.path.expanduser('~/vlm_robot_ws')),
        remappings=[
            ('/camera/color/image_raw', '/zedm/zed_node/rgb/image_rect_color'),
            ('/camera/depth/image_rect_raw', '/zedm/zed_node/depth/depth_registered'),
            ('/camera/aligned_depth_to_color/image_raw', '/zedm/zed_node/depth/depth_registered'),
            ('/camera/color/camera_info', '/zedm/zed_node/rgb/camera_info'),
        ]
    )

    return LaunchDescription([
        zed_wrapper_launch,
        lss_arm_launch,
        handybot_node
    ])
