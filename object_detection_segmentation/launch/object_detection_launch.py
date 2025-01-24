from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detection_segmentation',
            executable='object_detection',
            name='object_detection',
            output='screen'
        ),
    ])
