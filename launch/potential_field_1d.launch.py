from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ── Argument: sim (true = Gazebo, false = real robot) ──────────
    sim_arg = DeclareLaunchArgument(
        'sim',
        default_value='true',
        description='Set to true for Gazebo, false for real robot'
    )
    sim = LaunchConfiguration('sim')

    # ── Gazebo world (only when sim=true) ──────────────────────────
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'launch', 'turtlebot3_empty_world.launch.py')
        ]),
        condition=IfCondition(sim)
    )

    # ── Spawn box obstacle (only when sim=true) ────────────────────
    spawn_box = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'obstacle_box',
            '-file', os.path.join(
                get_package_share_directory('potential_fields_lab'),
                'models', 'obstacle_box', 'model.sdf'),
            '-x', '2.0', '-y', '0.0', '-z', '0.15'
        ],
        output='screen',
        condition=IfCondition(sim)
    )

    # ── Potential field node (always runs) ─────────────────────────
    pf_node = Node(
        package='potential_fields_lab',
        executable='potential_field_1d',
        output='screen'
    )

    return LaunchDescription([
        sim_arg,
        gazebo_launch,
        spawn_box,
        pf_node,
    ])
