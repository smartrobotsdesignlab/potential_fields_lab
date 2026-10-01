#!/usr/bin/env python3
"""
Fake differential-drive simulator — TurtleBot3 stand-in
============================================================
A tiny ROS2 node that lets the potential-field lab run WITHOUT
a robot and WITHOUT Gazebo. It does exactly what the real robot
does from the field node's point of view:

  subscribes : /cmd_vel   (geometry_msgs/Twist)
  publishes  : /odom      (nav_msgs/Odometry)
               odom -> base_footprint  (TF transform)

It integrates the unicycle model
      x'     = v cos(theta)
      y'     = v sin(theta)
      theta' = w
so the pose that comes back on /odom is a faithful, noise-free
version of how a differential drive would move under the commands
the field node sends.

Run it in one terminal, then run potential_field_2d and rviz2 in
others, exactly as you would against the real robot.

Start pose is the origin (0, 0, 0), same as a freshly booted robot.
============================================================
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class FakeDiffDrive(Node):
    def __init__(self):
        super().__init__('fake_diffdrive')

        # integration rate (Hz). 50 Hz is smooth and cheap.
        self.declare_parameter('rate', 50.0)
        self.rate = self.get_parameter('rate').value

        # pose state, starts at the origin like a booted robot
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # last commanded velocities (held until a new command arrives)
        self.v = 0.0
        self.w = 0.0

        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate, self.update)

        self.get_logger().info(
            f'Fake diff-drive up. Integrating /cmd_vel -> /odom at '
            f'{self.rate:.0f} Hz. Start pose (0, 0, 0).')

    def cmd_callback(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z

    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.0:
            return

        # unicycle integration (Euler; fine at 50 Hz)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta = self._wrap(self.theta + self.w * dt)

        stamp = now.to_msg()
        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        # --- /odom ---
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        self.odom_pub.publish(odom)

        # --- TF odom -> base_footprint ---
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def _wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = FakeDiffDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
