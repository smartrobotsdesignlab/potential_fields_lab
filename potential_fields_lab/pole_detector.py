#!/usr/bin/env python3
"""
pole_detector  —  Session 3: obstacles from the LiDAR
============================================================
Subscribes : /scan                      (sensor_msgs/LaserScan, LDS-01)
Publishes  : /pf2d_obstacles            (geometry_msgs/PoseArray, odom frame)
             /pole_markers              (visualization_msgs/MarkerArray)

Per scan:
  1. detect poles in the laser frame        (pole_detection.detect_poles)
  2. move each pole centre into the odom frame with TF, looked up at
     the time the laser actually swept that pole (beam_time_mode)
  3. track + smooth in odom                 (pole_detection.PoleTracker)
  4. publish the confirmed poles  (an EMPTY list is also published, so the
     field node can tell 'no poles' from 'detector not running')

beam_time_mode  — when did the beam that hit the pole fire?
  'stamp' : use header.stamp for every beam (simplest)
  'start' : stamp is the START of the sweep  -> t = stamp + i*time_increment
  'end'   : stamp is the END of the sweep    -> t = stamp - (N-1-i)*time_increment
The right choice for the real LDS-01 driver is found with the spin test.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

try:
    from potential_fields_lab.pole_detection import detect_poles, PoleTracker
except ImportError:                       # running the file directly
    from pole_detection import detect_poles, PoleTracker


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PoleDetector(Node):
    def __init__(self):
        super().__init__('pole_detector')
        P = self.declare_parameter
        P('scan_topic', '/scan'); P('fixed_frame', 'odom')
        P('pole_radius', 0.05); P('min_range', 0.12); P('max_range', 1.6)
        P('jump', 0.08); P('min_points', 3); P('min_width', 0.05); P('max_width', 0.16); P('bg_gap', 0.15)
        P('assoc_dist', 0.25); P('alpha', 0.3); P('confirm_hits', 3); P('drop_after', 1.5)
        P('beam_time_mode', 'stamp')
        P('tf_wait', 0.05)        # s to wait for the pose matching each scan
        # Test arena in the fixed frame: poles outside this box are ignored
        # (chairs and desks in the office). Disabled by default.
        P('roi_enable', False)
        P('roi_xmin', -10.0); P('roi_xmax', 10.0); P('roi_ymin', -10.0); P('roi_ymax', 10.0)
        g = lambda k: self.get_parameter(k).value
        self.fixed = g('fixed_frame'); self.mode = g('beam_time_mode')
        self.params = dict(pole_radius=g('pole_radius'), min_range=g('min_range'),
                           max_range=g('max_range'), jump=g('jump'), min_points=g('min_points'),
                           min_width=g('min_width'), max_width=g('max_width'), bg_gap=g('bg_gap'))
        self.R = g('pole_radius'); self.tf_wait = g('tf_wait')
        self.roi = (g('roi_xmin'), g('roi_xmax'), g('roi_ymin'), g('roi_ymax')) if g('roi_enable') else None
        self.tracker = PoleTracker(g('assoc_dist'), g('alpha'), g('confirm_hits'), g('drop_after'))

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        # own thread: TF keeps arriving while on_scan waits for the right moment
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(LaserScan, g('scan_topic'), self.on_scan, qos)
        self.pub = self.create_publisher(PoseArray, '/pf2d_obstacles', 10)
        self.mpub = self.create_publisher(MarkerArray, '/pole_markers', 10)
        self.n_scans = 0; self.n_fallback = 0; self.n_dets = 0; self.gaps = []; self.last_log = self.get_clock().now()
        self.get_logger().info(f'pole_detector up: r={self.R} m, range<{self.params["max_range"]} m, '
                               f'beam_time_mode={self.mode}, tf_wait={self.tf_wait} s, '
                               f'arena={"off" if self.roi is None else self.roi}')

    # -------- TF: laser frame -> fixed frame at a given time --------
    def lookup(self, laser_frame, t):
        """Pose of the laser in the fixed frame at time t. If the exact time is
        not available (usually: t is a few ms newer than the latest pose), use the
        newest pose and return the time gap, which is what decides the error."""
        try:
            return self.tf_buffer.lookup_transform(self.fixed, laser_frame, t,
                                                   timeout=Duration(seconds=self.tf_wait)), 0.0
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(self.fixed, laser_frame, Time())
                gap = abs((t - Time.from_msg(tf.header.stamp)).nanoseconds) / 1e6
                return tf, gap
            except Exception:
                return None, None

    def beam_time(self, msg, i):
        n = len(msg.ranges); stamp = Time.from_msg(msg.header.stamp)
        inc = msg.time_increment if msg.time_increment > 0 else (msg.scan_time / n if msg.scan_time > 0 else 0.2 / n)
        if self.mode == 'start':
            return stamp + Duration(seconds=i * inc)
        if self.mode == 'end':
            return stamp - Duration(seconds=(n - 1 - i) * inc)
        return stamp

    def on_scan(self, msg):
        self.n_scans += 1
        dets = detect_poles(msg.ranges, msg.angle_min, msg.angle_increment, self.params)
        pts = []
        for d in dets:
            i_mid = int(d['beams'][len(d['beams']) // 2])
            tf, gap = self.lookup(msg.header.frame_id, self.beam_time(msg, i_mid))
            if tf is None:
                continue
            self.n_dets += 1
            if gap > 0.0:
                self.n_fallback += 1; self.gaps.append(gap)
            tr = tf.transform.translation; yaw = quat_to_yaw(tf.transform.rotation)
            c, s = math.cos(yaw), math.sin(yaw)
            px, py = tr.x + c * d['x'] - s * d['y'], tr.y + s * d['x'] + c * d['y']
            if self.roi and not (self.roi[0] <= px <= self.roi[1] and self.roi[2] <= py <= self.roi[3]):
                continue
            pts.append((px, py))
        now_s = self.get_clock().now().nanoseconds / 1e9
        poles = self.tracker.update(pts, now_s)
        self.publish(poles, msg)
        if (self.get_clock().now() - self.last_log).nanoseconds > 1e9:
            self.last_log = self.get_clock().now()
            txt = '  '.join(f'({x:.2f},{y:.2f})' for x, y in poles) or 'none'
            g = self.gaps[-50:]
            gtxt = (f'pose gap mean {sum(g)/len(g):.0f} ms, max {max(g):.0f} ms' if g else 'pose gap 0 ms')
            self.get_logger().info(f'poles [{self.fixed}]: {txt}   raw this scan: {len(dets)}'
                                   f'   exact-time poses: {self.n_dets - self.n_fallback}/{self.n_dets}, {gtxt}')

    def publish(self, poles, scan):
        pa = PoseArray(); pa.header.stamp = scan.header.stamp; pa.header.frame_id = self.fixed
        for x, y in poles:
            p = Pose(); p.position.x = x; p.position.y = y; p.orientation.w = 1.0; pa.poses.append(p)
        self.pub.publish(pa)
        ma = MarkerArray()
        clr = Marker(); clr.action = Marker.DELETEALL; clr.ns = 'poles'; ma.markers.append(clr)
        for k, (x, y) in enumerate(poles):
            m = Marker(); m.header = pa.header; m.ns = 'poles'; m.id = k
            m.type = Marker.CYLINDER; m.action = Marker.ADD
            m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.25
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * self.R; m.scale.z = 0.5
            m.color.r, m.color.g, m.color.b, m.color.a = 0.95, 0.55, 0.1, 0.85
            ma.markers.append(m)
            t = Marker(); t.header = pa.header; t.ns = 'poles'; t.id = 100 + k
            t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = x; t.pose.position.y = y; t.pose.position.z = 0.6
            t.pose.orientation.w = 1.0; t.scale.z = 0.07
            t.color.r = t.color.g = t.color.b = 0.15; t.color.a = 1.0
            t.text = f'({x:.2f},{y:.2f})'
            ma.markers.append(t)
        self.mpub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PoleDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
