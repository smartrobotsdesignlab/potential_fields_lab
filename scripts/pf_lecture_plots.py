#!/usr/bin/env python3
"""
Potential Fields 2D — Lecture Figures
============================================================
Clean, reproducible matplotlib figures for the 20-minute lecture.
Same field equations as the ROS node (potential_field_2d.py) and the
same colour language as RViz: green goal, red obstacles, orange
influence rings.

Draws, per scenario:
  - potential-energy contours (the "landscape")
  - the force vector field (quiver)
  - the d0 influence ring around each obstacle
  - a point-robot trajectory that follows the field
  - goal, obstacles, and the start point

Usage
-----
  python3 pf_lecture_plots.py --scenario 1        # one scenario
  python3 pf_lecture_plots.py --all               # all five, one figure
  python3 pf_lecture_plots.py --scenario 2 --field-only
  python3 pf_lecture_plots.py --all --save ./figs # write PNG + PDF

No ROS needed. Requires only numpy and matplotlib.
============================================================
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
#   HOUSE STYLE  (clean white, matching Session 1)
# ------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'font.size':        11,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
})

C_GOAL   = '#3cb44b'   # green  (matches RViz goal)
C_OBS    = '#e6194b'   # red    (matches RViz obstacle)
C_RING   = '#f58231'   # orange (matches RViz d0 ring)
C_PATH   = '#1f4e79'   # dark blue trajectory
C_QUIV   = '#9aa7b4'   # muted grey-blue arrows
C_CONT   = '#d9d9d9'   # faint contour lines
C_START  = '#222222'

# ------------------------------------------------------------------
#   SCENARIOS  (identical to the YAML configs)
# ------------------------------------------------------------------
SCENARIOS = {
    1: dict(name='1  Single obstacle (avoids)',
            goal=(2.0, 0.0), obstacles=[(1.0, 0.25)],
            d0=0.7, k_att=1.0, k_rep=0.5, rot=False, k_rot=0.5),
    2: dict(name='2  Gap (spurious minimum)',
            goal=(2.0, 0.0), obstacles=[(1.0, 0.3), (1.0, -0.3)],
            d0=0.7, k_att=1.0, k_rep=0.5, rot=False, k_rot=0.5),
    3: dict(name='3  Box canyon (true trap)',
            goal=(2.5, 0.0),
            obstacles=[(1.5, -0.4), (1.5, -0.2), (1.5, 0.0), (1.5, 0.2),
                       (1.5, 0.4), (1.3, 0.5), (1.1, 0.5), (0.9, 0.5),
                       (1.3, -0.5), (1.1, -0.5), (0.9, -0.5)],
            d0=0.55, k_att=1.0, k_rep=0.5, rot=False, k_rot=0.5),
    4: dict(name='4  Collinear (saddle)',
            goal=(2.0, 0.0), obstacles=[(1.0, 0.0)],
            d0=0.7, k_att=1.0, k_rep=0.5, rot=False, k_rot=0.5),
    5: dict(name='5  Rotational escape',
            goal=(2.0, 0.0), obstacles=[(1.0, 0.0)],
            d0=0.7, k_att=1.0, k_rep=0.5, rot=True, k_rot=0.5),
}

# ------------------------------------------------------------------
#   FIELD EQUATIONS  (match potential_field_2d.py exactly)
# ------------------------------------------------------------------
def attractive_force(pos, goal, k_att):
    diff = np.asarray(goal) - pos
    d = np.linalg.norm(diff)
    if d < 1e-6:
        return np.zeros(2)
    return k_att * diff / d if d > 1.0 else k_att * diff

def repulsive_force(pos, obstacles, d0, k_rep):
    total = np.zeros(2)
    for obs in obstacles:
        diff = pos - np.asarray(obs)
        d = np.linalg.norm(diff)
        if d < 1e-6 or d >= d0:
            continue
        total += k_rep * (1.0/d - 1.0/d0) * (1.0/(d*d)) * (diff / d)
    return total

def rotational_force(pos, obstacles, d0, k_rot, on):
    if not on:
        return np.zeros(2)
    total = np.zeros(2)
    for obs in obstacles:
        diff = pos - np.asarray(obs)
        d = np.linalg.norm(diff)
        if d < 1e-6 or d >= d0:
            continue
        dirn = diff / d
        tangent = np.array([-dirn[1], dirn[0]])
        total += k_rot * (1.0/d - 1.0/d0) * tangent
    return total

def total_force(pos, s):
    return (attractive_force(pos, s['goal'], s['k_att'])
            + repulsive_force(pos, s['obstacles'], s['d0'], s['k_rep'])
            + rotational_force(pos, s['obstacles'], s['d0'], s['k_rot'], s['rot']))

# ------------------------------------------------------------------
#   POTENTIAL ENERGY  (the field's forces are -grad of this)
# ------------------------------------------------------------------
def potential(pos, s):
    diff = np.asarray(s['goal']) - pos
    d = np.linalg.norm(diff)
    # attractive: parabolic within 1 m, conical beyond (C1-continuous)
    if d <= 1.0:
        U = 0.5 * s['k_att'] * d * d
    else:
        U = s['k_att'] * d - 0.5 * s['k_att']
    # repulsive: Khatib
    for obs in s['obstacles']:
        do = np.linalg.norm(pos - np.asarray(obs))
        if 1e-6 < do < s['d0']:
            U += 0.5 * s['k_rep'] * (1.0/do - 1.0/s['d0'])**2
    return U

# ------------------------------------------------------------------
#   POINT-ROBOT TRAJECTORY  (follows the field direction)
# ------------------------------------------------------------------
def integrate_path(s, start=(0.0, 0.0), step=0.005, max_steps=8000,
                   goal_tol=0.1, stall_tol=0.03):
    pos = np.array(start, dtype=float)
    path = [pos.copy()]
    outcome = 'trapped'
    for _ in range(max_steps):
        if np.linalg.norm(np.asarray(s['goal']) - pos) < goal_tol:
            outcome = 'reached'
            break
        F = total_force(pos, s)
        fmag = np.linalg.norm(F)
        if fmag < stall_tol:
            outcome = 'stuck'
            break
        pos = pos + (F / fmag) * step        # constant-speed along the field
        path.append(pos.copy())
    return np.array(path), outcome

# ------------------------------------------------------------------
#   DRAW ONE SCENARIO INTO AN AXIS
# ------------------------------------------------------------------
def draw_scenario(ax, s, field_only=False):
    gx, gy = s['goal']
    xs = [0.0, gx] + [o[0] for o in s['obstacles']]
    ys = [0.0, gy] + [o[1] for o in s['obstacles']]
    pad = 0.8
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    # potential contours
    gx_n, gy_n = 220, 220
    X, Y = np.meshgrid(np.linspace(xmin, xmax, gx_n),
                       np.linspace(ymin, ymax, gy_n))
    U = np.zeros_like(X)
    for i in range(gx_n):
        for j in range(gy_n):
            U[j, i] = potential(np.array([X[j, i], Y[j, i]]), s)
    U = np.clip(U, None, np.percentile(U, 97))   # tame the repulsive spikes
    ax.contour(X, Y, U, levels=14, colors=C_CONT, linewidths=0.6, zorder=1)

    # force field (coarse quiver, direction only)
    qn = 22
    qx, qy = np.meshgrid(np.linspace(xmin, xmax, qn),
                         np.linspace(ymin, ymax, qn))
    U2 = np.zeros_like(qx); V2 = np.zeros_like(qy)
    for i in range(qn):
        for j in range(qn):
            F = total_force(np.array([qx[j, i], qy[j, i]]), s)
            n = np.linalg.norm(F)
            if n > 1e-6:
                U2[j, i], V2[j, i] = F / n
    ax.quiver(qx, qy, U2, V2, color=C_QUIV, alpha=0.8,
              scale=42, width=0.0032, zorder=2)

    # influence rings
    for obs in s['obstacles']:
        ax.add_patch(plt.Circle(obs, s['d0'], fill=False, ec=C_RING,
                                ls='--', lw=1.2, alpha=0.9, zorder=3))
    # obstacles
    ax.scatter([o[0] for o in s['obstacles']], [o[1] for o in s['obstacles']],
               s=140, c=C_OBS, edgecolors='white', linewidths=1.0,
               zorder=5, label='obstacle')
    # goal + start
    ax.scatter([gx], [gy], marker='*', s=420, c=C_GOAL, edgecolors='white',
               linewidths=1.0, zorder=6, label='goal')
    ax.scatter([0.0], [0.0], s=70, c=C_START, zorder=6, label='start')

    # trajectory
    if not field_only:
        path, outcome = integrate_path(s)
        ax.plot(path[:, 0], path[:, 1], color=C_PATH, lw=2.4, zorder=4)
        end = path[-1]
        if outcome == 'reached':
            tag, tc = 'reaches goal', C_GOAL
        elif outcome == 'stuck':
            tag, tc = 'stuck (local min)', C_OBS
            ax.scatter([end[0]], [end[1]], s=110, facecolors='none',
                       edgecolors=C_OBS, linewidths=1.8, zorder=7)
        else:
            tag, tc = 'trapped', C_OBS
        ax.text(0.02, 0.02, tag, transform=ax.transAxes, fontsize=9,
                color=tc, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=tc, lw=0.8))

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_title(s['name'], fontsize=11)
    ax.grid(True, color='#f0f0f0', lw=0.6)
    ax.set_axisbelow(True)

# ------------------------------------------------------------------
#   FIGURE BUILDERS
# ------------------------------------------------------------------
def figure_single(sid, field_only, save):
    s = SCENARIOS[sid]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    draw_scenario(ax, s, field_only=field_only)
    ax.set_xlabel('x  [m]'); ax.set_ylabel('y  [m]')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    fig.tight_layout()
    _finish(fig, f'scenario{sid}', save)

def figure_all(field_only, save):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
    axes = axes.ravel()
    for k, sid in enumerate(sorted(SCENARIOS)):
        draw_scenario(axes[k], SCENARIOS[sid], field_only=field_only)
    axes[-1].axis('off')   # sixth panel unused (five scenarios)
    fig.suptitle('Potential Fields in 2D — the five scenarios',
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _finish(fig, 'all_scenarios', save)

def _finish(fig, stem, save):
    if save:
        os.makedirs(save, exist_ok=True)
        for ext in ('png', 'pdf'):
            path = os.path.join(save, f'{stem}.{ext}')
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f'wrote {path}')
        plt.close(fig)
    else:
        plt.show()

# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Potential-field lecture figures')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--scenario', type=int, choices=[1, 2, 3, 4, 5],
                   help='draw a single scenario')
    g.add_argument('--all', action='store_true',
                   help='draw all five in one figure')
    ap.add_argument('--field-only', action='store_true',
                    help='omit the robot trajectory (field + landscape only)')
    ap.add_argument('--save', metavar='DIR', default=None,
                    help='save PNG + PDF into DIR instead of showing')
    args = ap.parse_args()

    if args.all:
        figure_all(args.field_only, args.save)
    else:
        figure_single(args.scenario or 1, args.field_only, args.save)

if __name__ == '__main__':
    main()
