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
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from sensor_msgs.msg import LaserScan
import random


class FakeDiffDrive(Node):
    def __init__(self):
        super().__init__('fake_diffdrive')

        # integration rate (Hz). 50 Hz is smooth and cheap.
        self.declare_parameter('rate', 50.0)
        self.rate = self.get_parameter('rate').value

        # ---- optional simulated LDS-01 (Session 3) ----
        # sim_poles : "x,y; x,y"  pole centres in the world/odom frame [m]
        # sim_walls : "xmin,xmax,ymin,ymax"  rectangular room, "" for none
        self.declare_parameter('sim_poles', '')
        self.declare_parameter('sim_walls', '')
        self.declare_parameter('sim_pole_radius', 0.05)
        self.declare_parameter('sim_scan_noise', True)
        self.poles = self._parse_pairs(self.get_parameter('sim_poles').value)
        w = self.get_parameter('sim_walls').value.strip()
        self.walls = [float(v) for v in w.split(',')] if w else None
        self.pole_r = self.get_parameter('sim_pole_radius').value
        self.scan_noise = self.get_parameter('sim_scan_noise').value

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

        if self.poles or self.walls:
            self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
            self.static_tf = StaticTransformBroadcaster(self)
            st = TransformStamped()
            st.header.stamp = self.get_clock().now().to_msg()
            st.header.frame_id = 'base_footprint'; st.child_frame_id = 'base_scan'
            st.transform.rotation.w = 1.0          # laser at the robot centre in sim
            self.static_tf.sendTransform(st)
            self.scan_timer = self.create_timer(0.2, self.publish_scan)   # 5 Hz like LDS-01
            self.get_logger().info(f'Simulated LDS-01 ON: {len(self.poles)} poles, '
                                   f'walls={self.walls}')

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
    def _parse_pairs(txt):
        out = []
        for chunk in str(txt).split(';'):
            chunk = chunk.strip()
            if chunk:
                x, y = chunk.split(',')
                out.append((float(x), float(y)))
        return out

    def publish_scan(self):
        """360 beams, 1 deg, ray-cast from the current pose against the poles
        and walls. Noise ~ LDS-01 spec (sigma 5 mm < 0.5 m, 1.75 % beyond)."""
        n = 360; inc = 2.0 * math.pi / n
        ranges = []
        for i in range(n):
            a = self.theta + i * inc                    # beam direction in world
            dx, dy = math.cos(a), math.sin(a)
            best = float('inf')
            for (cx, cy) in self.poles:
                ox, oy = cx - self.x, cy - self.y
                b = dx * ox + dy * oy
                disc = b * b - (ox * ox + oy * oy - self.pole_r ** 2)
                if disc >= 0:
                    t = b - math.sqrt(disc)
                    if 0 < t < best:
                        best = t
            if self.walls:
                xmin, xmax, ymin, ymax = self.walls
                for t in ((xmax - self.x) / dx if dx > 1e-9 else None,
                          (xmin - self.x) / dx if dx < -1e-9 else None,
                          (ymax - self.y) / dy if dy > 1e-9 else None,
                          (ymin - self.y) / dy if dy < -1e-9 else None):
                    if t is not None and 0 < t < best:
                        best = t
            if best < float('inf') and self.scan_noise:
                best += random.gauss(0.0, 0.005 if best < 0.5 else 0.0175 * best)
            if not (0.12 <= best <= 3.5):
                best = float('inf')
            ranges.append(best)
        m = LaserScan()
        m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = 'base_scan'
        m.angle_min = 0.0; m.angle_max = 2 * math.pi - inc; m.angle_increment = inc
        m.time_increment = 0.0; m.scan_time = 0.2       # whole scan taken at one pose in sim
        m.range_min = 0.12; m.range_max = 3.5; m.ranges = ranges
        self.scan_pub.publish(m)

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
