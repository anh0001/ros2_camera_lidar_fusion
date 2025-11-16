#!/usr/bin/env python3

"""
Launch RealSense RGB-only stream alongside the intrinsic calibration node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return launch description that brings up RealSense and the calibration node."""
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'pointcloud.enable': 'false',
        }.items(),
    )

    calibration_node = Node(
        package='ros2_camera_lidar_fusion',
        executable='get_intrinsic_camera_calibration',
        name='camera_intrinsics_calibration',
    )

    return LaunchDescription([
        realsense_launch,
        calibration_node,
    ])
