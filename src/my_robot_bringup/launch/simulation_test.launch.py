import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Launch the ZED M Camera Wrapper (using the correct version-specific file)
    zed_wrapper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'), 'launch'),
            '/zedm.launch.py'
        ])
    )

    # 2. Launch the FAKE Lynxmotion Arm Controller for simulation in RViz
    # This does NOT connect to the physical hardware.
    lss_arm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('lss_arm_moveit'), 'launch'),
            '/fake_arm_control.launch.py' # Using the fake controller
        ]),
        launch_arguments={'dof': '5'}.items()
    )

    # 3. Launch the core tabletop_handybot node with topic remappings
    handybot_node = Node(
        package='tabletop_handybot',
        executable='handybot_node', # VERIFY THIS EXECUTABLE NAME
        name='handybot_node',
        output='screen',
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