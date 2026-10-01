#!/usr/bin/env python3
"""
Pole detection from a single 2D laser scan  (no ROS imports here)
====================================================================
Pure functions, so the same code runs in the ROS node, in the
simulator, and in offline tests.

Pipeline for one scan
  1. range gate      keep min_range <= r <= max_range
  2. clustering      split where consecutive ranges jump by > jump
  3. width gate      keep clusters with >= min_points and a chord
                     between min_width and max_width
  3b. foreground     the beams just outside the cluster must be farther
                     away by >= bg_gap (or empty): a pole stands in front
                     of its background, a piece of wall does not
  4. centre          fit a circle of KNOWN radius to the cluster points
                     (the laser only sees the front surface, so the
                     centre lies one radius behind it)

All coordinates returned are in the LASER frame (x forward, y left).
"""
import math
import numpy as np

DEFAULTS = dict(
    pole_radius=0.05,   # 10 cm diameter thermocol cylinder
    min_range=0.12,     # LDS-01 lower limit
    max_range=1.6,      # beyond this a 10 cm pole has < ~3 beams
    jump=0.08,          # range jump that separates two objects [m]
    min_points=3,
    min_width=0.05,     # chord limits [m]; chair legs (~2-3 cm) fail
    max_width=0.16,
    bg_gap=0.15,        # a pole must stand this far IN FRONT of the returns
                        # on both sides of it (rejects pieces of wall)
)


def scan_to_points(ranges, angle_min, angle_inc):
    r = np.asarray(ranges, dtype=float)
    a = angle_min + angle_inc * np.arange(len(r))
    return r, a


def cluster_scan(r, a, p):
    """Return list of index arrays, one per cluster, handling the 360 wrap."""
    valid = np.isfinite(r) & (r >= p['min_range']) & (r <= p['max_range'])
    n = len(r)
    clusters, cur = [], []
    for i in range(n):
        if not valid[i]:
            if cur:
                clusters.append(cur); cur = []
            continue
        if cur and abs(r[i] - r[cur[-1]]) > p['jump']:
            clusters.append(cur); cur = []
        cur.append(i)
    if cur:
        clusters.append(cur)
    # merge first and last cluster if the object straddles angle 0
    if (len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == n - 1
            and abs(r[0] - r[n - 1]) <= p['jump']):
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    return [np.array(c) for c in clusters]


def fit_known_radius(px, py, R, iters=15):
    """Least-squares centre of a circle with known radius R.
    Initial guess: closest point pushed back by R along its ray."""
    k = int(np.argmin(np.hypot(px, py)))
    rng = math.hypot(px[k], py[k]); ang = math.atan2(py[k], px[k])
    cx, cy = (rng + R) * math.cos(ang), (rng + R) * math.sin(ang)
    for _ in range(iters):
        dx, dy = cx - px, cy - py
        d = np.hypot(dx, dy) + 1e-12
        res = d - R
        J = np.stack([dx / d, dy / d], axis=1)
        try:
            step = np.linalg.lstsq(J, -res, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        cx += step[0]; cy += step[1]
        if np.hypot(*step) < 1e-5:
            break
    rms = float(np.sqrt(np.mean((np.hypot(cx - px, cy - py) - R) ** 2)))
    return cx, cy, rms


def is_foreground(r, idx, gap):
    """True if the beams on BOTH sides of the cluster are empty or farther
    than the cluster's nearest point by at least gap (360-degree wrap)."""
    n = len(r); near = np.min(r[idx])
    for j in ((idx[0] - 1) % n, (idx[-1] + 1) % n):
        if np.isfinite(r[j]) and r[j] < near + gap:
            return False
    return True


def detect_poles(ranges, angle_min, angle_inc, params=None):
    p = dict(DEFAULTS); p.update(params or {})
    r, a = scan_to_points(ranges, angle_min, angle_inc)
    # The real LDS-01 reports "no return" as 0.0 (the simulator used inf).
    # Treat anything below min_range as no return, otherwise a 0.0 next to a
    # pole looks like a very close object and the foreground test rejects it.
    r = np.where(r >= p['min_range'], r, np.inf)
    out = []
    for idx in cluster_scan(r, a, p):
        if len(idx) < p['min_points']:
            continue
        x = r[idx] * np.cos(a[idx]); y = r[idx] * np.sin(a[idx])
        chord = math.hypot(x[-1] - x[0], y[-1] - y[0])
        if not (p['min_width'] <= chord <= p['max_width']):
            continue
        if not is_foreground(r, idx, p['bg_gap']):
            continue
        cx, cy, rms = fit_known_radius(x, y, p['pole_radius'])
        # reject fits that are clearly not a cylinder of this size
        if rms > 0.03:
            continue
        out.append(dict(x=cx, y=cy, n=int(len(idx)), width=chord,
                        range=math.hypot(cx, cy), rms=rms,
                        beams=idx))
    return out


class PoleTracker:
    """Keeps a short memory of poles in a FIXED frame (odom).
    - associate each detection with the nearest track within assoc_dist
    - smooth with an exponential moving average (alpha)
    - a track is reported once it has been seen confirm_hits times
    - a track is dropped if not seen for drop_after seconds
    """
    def __init__(self, assoc_dist=0.25, alpha=0.3, confirm_hits=3, drop_after=1.5):
        self.assoc_dist, self.alpha = assoc_dist, alpha
        self.confirm_hits, self.drop_after = confirm_hits, drop_after
        self.tracks = []          # dicts: x, y, hits, last

    def update(self, detections_xy, t):
        used = set()
        for (x, y) in detections_xy:
            best, bd = None, self.assoc_dist
            for k, tr in enumerate(self.tracks):
                if k in used:
                    continue
                d = math.hypot(tr['x'] - x, tr['y'] - y)
                if d < bd:
                    best, bd = k, d
            if best is None:
                self.tracks.append(dict(x=x, y=y, hits=1, last=t))
                used.add(len(self.tracks) - 1)
            else:
                tr = self.tracks[best]
                tr['x'] += self.alpha * (x - tr['x'])
                tr['y'] += self.alpha * (y - tr['y'])
                tr['hits'] += 1
                tr['last'] = t
                used.add(best)
        self.tracks = [tr for tr in self.tracks if t - tr['last'] <= self.drop_after]
        return self.confirmed()

    def confirmed(self):
        return [(tr['x'], tr['y']) for tr in self.tracks if tr['hits'] >= self.confirm_hits]
