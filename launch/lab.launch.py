#!/usr/bin/env python3
"""
Potential Fields Lab — Main Launch File
============================================================
Launches the potential field node and data logger together.

Usage:
  # Run baseline experiment
  ros2 launch potential_fields_lab lab.launch.py

  # Run specific experiment
  ros2 launch potential_fields_lab lab.launch.py config:=exp2_no_damping

  # Override a single parameter on top of a config
  ros2 launch potential_fields_lab lab.launch.py config:=exp1_baseline k_damp:=0.0

Available configs:
  exp1_baseline
  exp2_no_damping
  exp3_weak_repulsion
  exp4_strong_repulsion
  exp5_challenge
============================================================
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():

    # ── Launch Arguments ────────────────────────────────────────────
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='exp1_baseline',
        description='Experiment config name (without .yaml extension)'
    )

    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value=os.path.expanduser('~/pf_logs'),
        description='Directory to save CSV log files'
    )

    # ── Config file path ────────────────────────────────────────────
    config_file = PathJoinSubstitution([
        FindPackageShare('potential_fields_lab'),
        'config',
        [LaunchConfiguration('config'), '.yaml']
    ])

    # ── Potential Field Node ────────────────────────────────────────
    pf_node = Node(
        package='potential_fields_lab',
        executable='potential_field_1d',
        name='potential_field_1d',
        output='screen',
        parameters=[config_file],
        emulate_tty=True,
    )

    # ── Logger Node ─────────────────────────────────────────────────
    logger_node = Node(
        package='potential_fields_lab',
        executable='pf_logger',
        name='pf_logger',
        output='screen',
        parameters=[
            config_file,
            {'log_dir': LaunchConfiguration('log_dir')}
        ],
        emulate_tty=True,
    )

    # ── Info message ────────────────────────────────────────────────
    info = LogInfo(
        msg=[
            '\n',
            '=' * 55, '\n',
            '  POTENTIAL FIELDS LAB\n',
            '  Config : ', LaunchConfiguration('config'), '\n',
            '  Logs   : ', LaunchConfiguration('log_dir'), '\n',
            '=' * 55,
        ]
    )

    return LaunchDescription([
        config_arg,
        log_dir_arg,
        info,
        pf_node,
        logger_node,
    ])
