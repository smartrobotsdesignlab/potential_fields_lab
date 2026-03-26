#!/usr/bin/env python3
"""
Potential Fields 1D — TurtleBot3 + ROS2 Humble
============================================================
Core robot control node for the Potential Fields Lab.
Do not change anything here. 
 Change parameters using YAML config files in config/
          or via command line arguments.

Author: Ankit Ravankar
============================================================
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String
import json


class PotentialField1D(Node):
    def __init__(self):
        super().__init__('potential_field_1d')

        # ============================================================
        #   DECLARE ROS2 PARAMETERS — change via YAML or command line
        #   Edit config/*.yaml files, not this file
        # ============================================================

        # Attractive field
        self.declare_parameter('k_att', 2.0)
        self.declare_parameter('goal_distance', 0.5)

        # Repulsive field
        self.declare_parameter('k_rep', 0.8)
        self.declare_parameter('influence_radius', 1.5)

        # Damping
        self.declare_parameter('k_damp', 1.5)

        # Safety
        self.declare_parameter('max_speed', 0.3)
        self.declare_parameter('min_distance', 0.25)

        # Experiment metadata
        self.declare_parameter('experiment_name', 'baseline')

        # ============================================================
        #   READ PARAMETERS
        # ============================================================
        self.k_att            = self.get_parameter('k_att').value
        self.goal_distance    = self.get_parameter('goal_distance').value
        self.k_rep            = self.get_parameter('k_rep').value
        self.influence_radius = self.get_parameter('influence_radius').value
        self.k_damp           = self.get_parameter('k_damp').value
        self.max_speed        = self.get_parameter('max_speed').value
        self.min_distance     = self.get_parameter('min_distance').value
        self.experiment_name  = self.get_parameter('experiment_name').value

        # ============================================================
        #   STATE
        # ============================================================
        self.current_distance = float('inf')
        self.current_speed    = 0.0
        self.start_time       = self.get_clock().now()
        self.equilibrium_count = 0
        self.equilibrium_distance = None

        # ============================================================
        #   QoS for real TB3 LiDAR
        # ============================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ============================================================
        #   PUBLISHERS
        # ============================================================
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)

        # Debug topic — all field values in one message
        # [time, distance, f_att, f_rep, f_total, velocity]
        self.debug_pub  = self.create_publisher(
            Float32MultiArray, '/pf_debug', 10)

        # Status topic — human readable experiment status
        self.status_pub = self.create_publisher(
            String, '/pf_status', 10)

        # ============================================================
        #   SUBSCRIBERS
        # ============================================================
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos)

        # ============================================================
        #   CONTROL LOOP — 10 Hz
        # ============================================================
        self.timer = self.create_timer(0.1, self.control_loop)

        # ============================================================
        #   STARTUP LOG
        # ============================================================
        self.get_logger().info('=' * 55)
        self.get_logger().info('  POTENTIAL FIELDS LAB — Node Started')
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'  Experiment  : {self.experiment_name}')
        self.get_logger().info(f'  k_att       : {self.k_att}')
        self.get_logger().info(f'  k_rep       : {self.k_rep}')
        self.get_logger().info(f'  d0 (influence): {self.influence_radius} m')
        self.get_logger().info(f'  goal_distance : {self.goal_distance} m')
        self.get_logger().info(f'  k_damp      : {self.k_damp}')
        self.get_logger().info(f'  max_speed   : {self.max_speed} m/s')
        self.get_logger().info('=' * 55)
        self.get_logger().info('  Waiting for LiDAR scan...')

    # ================================================================
    #   SENSOR PROCESSING
    # ================================================================
    def get_front_distance(self, msg):
        """
        Extract minimum distance in the forward-facing sector.

        TB3 LDS-01: index 0 = front, scan goes counterclockwise.
        We take ±10% of total scan indices around index 0.
        """
        ranges  = np.array(msg.ranges)
        n       = len(ranges)
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

    # ================================================================
    #   POTENTIAL FIELD FUNCTIONS
    # ================================================================
    def compute_attractive(self, d):
        """
        Parabolic (spring-like) attractive potential.

        U_att = 0.5 * k_att * (d - d_goal)^2
        F_att = k_att * (d - d_goal)

        Positive force → move forward (toward goal)
        Negative force → move backward (past goal)
        Zero at d = goal_distance (equilibrium candidate)
        """
        return self.k_att * (d - self.goal_distance)

    def compute_repulsive(self, d):
        """
        Khatib repulsive potential.

        U_rep = 0.5 * k_rep * (1/d - 1/d0)^2    when d < d0
              = 0                                  when d >= d0

        F_rep = -k_rep * (1/d - 1/d0) * (1/d^2)  when d < d0
              = 0                                  when d >= d0

        Always negative → pushes robot away from obstacle.
        Grows rapidly as d approaches 0 (singularity protection at 0.01m).
        """
        if d >= self.influence_radius:
            return 0.0
        d = max(d, 0.01)
        return -self.k_rep * (1.0/d - 1.0/self.influence_radius) \
                           * (1.0/(d**2))

    def compute_potential_energy(self, d):
        """
        Compute potential energy values for logging.
        U_att and U_rep — useful for plotting the energy landscape.
        """
        u_att = 0.5 * self.k_att * (d - self.goal_distance)**2

        if d >= self.influence_radius:
            u_rep = 0.0
        else:
            d_safe = max(d, 0.01)
            u_rep = 0.5 * self.k_rep * \
                    (1.0/d_safe - 1.0/self.influence_radius)**2

        return u_att, u_rep

    # ================================================================
    #   CALLBACKS
    # ================================================================
    def scan_callback(self, msg):
        self.current_distance = self.get_front_distance(msg)

    def control_loop(self):
        d = self.current_distance

        # Time since start
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # No valid reading — hold position
        if not np.isfinite(d):
            self.get_logger().warn(
                'No valid range reading — holding position',
                throttle_duration_sec=2.0)
            self.stop_robot()
            return

        # Emergency stop
        if d < self.min_distance:
            self.get_logger().warn(
                f'EMERGENCY STOP: d={d:.3f}m < min={self.min_distance}m')
            self.stop_robot()
            self._publish_status('EMERGENCY_STOP', d, 0, 0, 0, 0, elapsed)
            return

        # Compute forces
        f_att   = self.compute_attractive(d)
        f_rep   = self.compute_repulsive(d)
        f_damp  = -self.k_damp * self.current_speed
        f_total = f_att + f_rep + f_damp

        # Compute potential energies for logging
        u_att, u_rep = self.compute_potential_energy(d)

        # Velocity update (Euler integration)
        new_speed = self.current_speed + 0.1 * f_total
        new_speed = np.clip(new_speed, -self.max_speed, self.max_speed)

        # Prevent creeping backward unless strongly pushed
        if new_speed < 0 and f_total > -0.1:
            new_speed = 0.0

        self.current_speed = new_speed

        # Detect equilibrium — robot stopped for 10 consecutive cycles
        if abs(new_speed) < 0.005 and abs(f_total) < 0.15:
            self.equilibrium_count += 1
            if self.equilibrium_count == 10:
                self.equilibrium_distance = d
                self.get_logger().info('─' * 55)
                self.get_logger().info(
                    f'  EQUILIBRIUM REACHED at d = {d:.3f} m')
                self.get_logger().info(
                    f'  F_att={f_att:.3f} | F_rep={f_rep:.3f}')
                self.get_logger().info(
                    f'  Theory predicts: solve k_att*(d-d_goal) = '
                    f'k_rep*(1/d - 1/d0)*(1/d^2)')
                self.get_logger().info('─' * 55)
        else:
            self.equilibrium_count = 0

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x  = float(new_speed)
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        # Publish debug data
        # [time, dist, f_att, f_rep, f_total, vel, u_att, u_rep]
        debug = Float32MultiArray()
        debug.data = [
            float(elapsed),
            float(d),
            float(f_att),
            float(f_rep),
            float(f_total),
            float(new_speed),
            float(u_att),
            float(u_rep),
        ]
        self.debug_pub.publish(debug)

        # Publish status
        self._publish_status(
            'RUNNING', d, f_att, f_rep, f_total, new_speed, elapsed)

        # Terminal log
        self.get_logger().info(
            f't={elapsed:.1f}s | d={d:.3f}m | '
            f'F_att={f_att:.3f} | F_rep={f_rep:.3f} | '
            f'F_total={f_total:.3f} | v={new_speed:.3f}m/s')

    # ================================================================
    #   HELPERS
    # ================================================================
    def _publish_status(self, state, d, f_att, f_rep, f_total, v, t):
        status = {
            'state': state,
            'experiment': self.experiment_name,
            'time': round(t, 2),
            'distance': round(d, 3),
            'f_att': round(f_att, 3),
            'f_rep': round(f_rep, 3),
            'f_total': round(f_total, 3),
            'velocity': round(v, 3),
            'params': {
                'k_att': self.k_att,
                'k_rep': self.k_rep,
                'k_damp': self.k_damp,
                'd0': self.influence_radius,
                'd_goal': self.goal_distance,
            }
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

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
        node.get_logger().info('Keyboard interrupt — stopping robot.')
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
