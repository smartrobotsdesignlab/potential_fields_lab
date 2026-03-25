#!/usr/bin/env python3
"""
Potential Fields 1D — TurtleBot3 + ROS2 Humble
Works on both Gazebo simulation and real TB3 hardware.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class PotentialField1D(Node):
    def __init__(self):
        super().__init__('potential_field_1d')

        # ============================================================
        #   TUNABLE PARAMETERS
        # ============================================================
        self.k_att            = 2.0
        self.goal_distance    = 0.5
        self.k_rep            = 0.8
        self.influence_radius = 1.5
        self.k_damp           = 1.5
        self.max_speed        = 0.3
        self.min_distance     = 0.25
        # ============================================================

        self.current_distance = float('inf')
        self.current_speed    = 0.0

        # Publishers and Subscribers
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, '/pf_debug', 10)
        qos = QoSProfile(
    	   reliability=ReliabilityPolicy.BEST_EFFORT,
           history=HistoryPolicy.KEEP_LAST,
           depth=10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos)

        # Control loop at 10Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('=== Potential Field Node Started ===')
        self.get_logger().info(
            f'k_att={self.k_att} | k_rep={self.k_rep} | '
            f'd0={self.influence_radius}m | k_damp={self.k_damp}')

    def get_front_distance(self, msg):
        ranges = np.array(msg.ranges)
        n      = len(ranges)

        # Index 0 is front on TB3 LDS-01
        # Take ±10% window wrapping around index 0
        delta   = n // 20
        indices = np.arange(-delta, delta + 1) % n
        front   = ranges[indices]

        valid = front[
            np.isfinite(front) &
            (front > msg.range_min) &
            (front < msg.range_max)
        ]

        if len(valid) == 0:
            return float('inf')
        return float(np.min(valid))

    def compute_attractive(self, d):
        """
        Parabolic spring-like attractive force.
        F_att = k_att * (d - goal_distance)
        Positive = move forward, Negative = move backward.
        """
        return self.k_att * (d - self.goal_distance)

    def compute_repulsive(self, d):
        """
        Khatib repulsive force.
        F_rep = -k_rep * (1/d - 1/d0) * (1/d^2)  when d < d0
              = 0                                   when d >= d0
        """
        if d >= self.influence_radius:
            return 0.0
        d = max(d, 0.01)
        return -self.k_rep * (1.0/d - 1.0/self.influence_radius) * (1.0/(d**2))

    def scan_callback(self, msg):
        self.current_distance = self.get_front_distance(msg)

    def control_loop(self):
        d = self.current_distance

        # No valid reading yet — hold position
        if not np.isfinite(d):
            self.get_logger().warn(
                'No valid range reading — holding position',
                throttle_duration_sec=2.0)
            self.stop_robot()
            return

        # Emergency stop
        if d < self.min_distance:
            self.get_logger().warn(f'EMERGENCY STOP: d={d:.3f}m')
            self.stop_robot()
            return

        # Compute forces
        f_att  = self.compute_attractive(d)
        f_rep  = self.compute_repulsive(d)
        f_damp = -self.k_damp * self.current_speed
        f_total = f_att + f_rep + f_damp

        # Velocity update
        new_speed = self.current_speed + 0.1 * f_total
        new_speed = np.clip(new_speed, -self.max_speed, self.max_speed)
        if new_speed < 0 and f_total > -0.1:
            new_speed = 0.0
        self.current_speed = new_speed

        # Publish velocity
        cmd = Twist()
        cmd.linear.x  = float(new_speed)
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        # Publish debug data
        debug = Float32MultiArray()
        debug.data = [
            float(d),
            float(f_att),
            float(f_rep),
            float(f_total),
            float(new_speed)
        ]
        self.debug_pub.publish(debug)

        self.get_logger().info(
            f'd={d:.2f}m | F_att={f_att:.3f} | '
            f'F_rep={f_rep:.3f} | F_total={f_total:.3f} | '
            f'v={new_speed:.3f}m/s')

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.current_speed = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = PotentialField1D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
        node.get_logger().info('Node stopped cleanly.')
    finally:
	node.stop_robot()
        node.destroy_node()
	if rclpy.ok():       
	    rclpy.shutdown()


if __name__ == '__main__':
    main()
