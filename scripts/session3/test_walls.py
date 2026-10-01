import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'potential_fields_lab'))
import numpy as np
from pole_detection import detect_poles
rng = np.random.default_rng(5)
WALLS = (-2.6, 2.8, -2.4, 2.5)          # same as sim_poles_*.yaml

def world_scan(rx, ry, th, poles, walls, R=0.05):
    """Ray-cast in the WORLD frame from pose (rx,ry,th), like fake_diffdrive."""
    out = np.full(360, np.inf)
    for i in range(360):
        a = th + i * 2 * np.pi / 360; dx, dy = math.cos(a), math.sin(a); best = np.inf
        for cx, cy in poles:
            ox, oy = cx - rx, cy - ry; b = dx * ox + dy * oy
            disc = b * b - (ox * ox + oy * oy - R * R)
            if disc >= 0:
                t = b - math.sqrt(disc)
                if 0 < t < best: best = t
        xmin, xmax, ymin, ymax = walls
        for t in ((xmax - rx) / dx if dx > 1e-9 else None, (xmin - rx) / dx if dx < -1e-9 else None,
                  (ymax - ry) / dy if dy > 1e-9 else None, (ymin - ry) / dy if dy < -1e-9 else None):
            if t is not None and 0 < t < best: best = t
        if np.isfinite(best):
            best += rng.normal(0, 0.005 if best < 0.5 else 0.0175 * best)
        out[i] = best if 0.12 <= best <= 3.5 else np.inf
    return out

def evaluate(poles, n_poses, label, positions=None):
    for name, prm in [('old (no foreground test)', {'bg_gap': -1e9}), ('new (foreground test) ', {})]:
        fp = 0; hits = 0; vis = 0; errs = []
        r2 = np.random.default_rng(11)
        for k in range(n_poses):
            if positions is None:
                rx = r2.uniform(WALLS[0] + 0.3, WALLS[1] - 0.3); ry = r2.uniform(WALLS[2] + 0.3, WALLS[3] - 0.3)
            else:
                rx, ry = positions[k % len(positions)]
            th = r2.uniform(-np.pi, np.pi)
            if any(math.hypot(rx - px, ry - py) < 0.25 for px, py in poles): continue
            s = world_scan(rx, ry, th, poles, WALLS)
            for d in detect_poles(s, 0.0, 2 * np.pi / 360, prm):
                c, sn = math.cos(th), math.sin(th)
                wx, wy = rx + c * d['x'] - sn * d['y'], ry + sn * d['x'] + c * d['y']
                dist = [math.hypot(wx - px, wy - py) for px, py in poles]
                if dist and min(dist) < 0.25: hits += 1; errs.append(min(dist))
                else: fp += 1
            vis += sum(1 for px, py in poles if math.hypot(px - rx, py - ry) < 1.6)
        print(f"  {label:34s} {name}: false poles {fp:4d} in {n_poses} scans | "
              f"real poles found {hits}/{vis} in range" + (f" | err {np.mean(errs)*100:.1f} cm" if errs else ""))

print("1) your sim run: robot along the path to the goal, near the x = 2.8 wall")
path = [(x, -0.1) for x in np.linspace(0.0, 2.3, 24)]
evaluate([(1.0, 0.25)], 240, "single pole, along the path", path)
print("2) robot anywhere in the room (random poses), one pole in the open")
evaluate([(1.0, 0.25)], 400, "random poses")
print("3) gap poles, random poses")
evaluate([(1.0, 0.3), (1.0, -0.3)], 400, "random poses, two poles")
print("4) pole 0.30 m in front of a wall (limit case)")
evaluate([(2.5, 0.0)], 200, "pole 0.30 m from wall", [(1.6, 0.0), (1.8, 0.3), (1.7, -0.4)])
print("5) pole 0.10 m in front of a wall (expected to be rejected)")
evaluate([(2.7, 0.0)], 200, "pole 0.10 m from wall", [(1.8, 0.0), (2.0, 0.3), (1.9, -0.4)])
