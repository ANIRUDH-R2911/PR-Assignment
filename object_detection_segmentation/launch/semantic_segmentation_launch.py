from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detection_segmentation',
            executable='semantic_segmentation',
            name='semantic_segmentation',
            output='screen'
        ),
    ])
