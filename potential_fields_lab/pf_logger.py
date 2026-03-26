#!/usr/bin/env python3
"""
Potential Fields Lab — Data Logger
============================================================
Subscribes to /pf_debug and logs all field values to CSV.
Runs silently in background alongside the main node.

CSV columns:
  time, distance, f_att, f_rep, f_total, velocity, u_att, u_rep

Usage (automatic via launch file):
  ros2 run potential_fields_lab pf_logger --ros-args -p experiment_name:=exp1
============================================================
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import csv
import os
import json
from datetime import datetime


class PFLogger(Node):
    def __init__(self):
        super().__init__('pf_logger')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('experiment_name', 'baseline')
        self.declare_parameter('log_dir', os.path.expanduser('~/pf_logs'))

        self.experiment_name = self.get_parameter('experiment_name').value
        self.log_dir         = self.get_parameter('log_dir').value

        # ── Setup log directory and file ────────────────────────────
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename   = f'{self.experiment_name}_{timestamp}.csv'
        self.filepath = os.path.join(self.log_dir, filename)

        # ── Open CSV file ───────────────────────────────────────────
        self.csv_file = open(self.filepath, 'w', newline='')
        self.writer   = csv.writer(self.csv_file)

        # Write header
        self.writer.writerow([
            'time_s',
            'distance_m',
            'f_att',
            'f_rep',
            'f_total',
            'velocity_ms',
            'u_att',
            'u_rep',
        ])
        self.csv_file.flush()

        # ── Subscribers ─────────────────────────────────────────────
        self.debug_sub = self.create_subscription(
            Float32MultiArray, '/pf_debug', self.debug_callback, 10)

        self.status_sub = self.create_subscription(
            String, '/pf_status', self.status_callback, 10)

        # ── State ───────────────────────────────────────────────────
        self.row_count    = 0
        self.last_state   = None
        self.params_saved = False

        self.get_logger().info('=' * 55)
        self.get_logger().info('  PF LOGGER — Started')
        self.get_logger().info(f'  Experiment : {self.experiment_name}')
        self.get_logger().info(f'  Logging to : {self.filepath}')
        self.get_logger().info('=' * 55)

    def debug_callback(self, msg):
        """Write one row per control loop cycle."""
        if len(msg.data) < 8:
            return

        row = [round(v, 4) for v in msg.data]
        self.writer.writerow(row)
        self.row_count += 1

        # Flush every 10 rows so data is safe if node crashes
        if self.row_count % 10 == 0:
            self.csv_file.flush()

    def status_callback(self, msg):
        """Log parameter info on first message and state changes."""
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        # Save parameters to a separate metadata file on first message
        if not self.params_saved:
            self._save_metadata(status)
            self.params_saved = True

        # Log state transitions
        state = status.get('state', '')
        if state != self.last_state:
            self.get_logger().info(f'  State: {state}')
            if state == 'EMERGENCY_STOP':
                self.get_logger().warn(
                    f"  EMERGENCY STOP at d={status['distance']:.3f}m")
            self.last_state = state

    def _save_metadata(self, status):
        """Save experiment parameters to a JSON metadata file."""
        meta_path = self.filepath.replace('.csv', '_metadata.json')
        metadata  = {
            'experiment': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': status.get('params', {}),
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.get_logger().info(f'  Metadata saved: {meta_path}')

    def destroy_node(self):
        """Clean shutdown — flush and close CSV."""
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info('─' * 55)
        self.get_logger().info(f'  Logger stopped.')
        self.get_logger().info(f'  Rows logged  : {self.row_count}')
        self.get_logger().info(f'  File saved   : {self.filepath}')
        self.get_logger().info('─' * 55)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PFLogger()
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
