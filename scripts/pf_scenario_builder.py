#!/usr/bin/env python3
"""
Potential Fields 2D — Scenario Builder
============================================================
An interactive tool to design your own scenarios and export a
YAML config that the ROS node (potential_field_2d.py) reads
directly. Same field equations as the node, the simulator, and
the lecture figures, so a scenario you build here behaves the
same everywhere.

Controls
--------
  left click            : add an obstacle
  right click           : remove the nearest obstacle
  shift + left click    : move the goal
  sliders               : k_att, k_rep, influence radius d0, k_rot
  "rotational" checkbox : turn the rotational escape field on/off
  name box + Save YAML  : write <name>.yaml into the output folder
  Clear                 : remove all obstacles

The panel redraws the force field and a point-robot path after
every change, with the outcome (reaches goal / stuck / trapped).

Run
---
  python3 pf_scenario_builder.py                 # interactive
  python3 pf_scenario_builder.py --out ./config  # save YAMLs there
  python3 pf_scenario_builder.py --selftest      # headless check, no window
============================================================
"""

import argparse
import os
import sys

# ---- parse args before importing pyplot so --selftest can pick a backend ----
_ap = argparse.ArgumentParser(description='Potential-field scenario builder')
_ap.add_argument('--out', default='.', help='folder to save YAML configs into')
_ap.add_argument('--selftest', action='store_true',
                 help='run a headless build+export check and exit')
ARGS = _ap.parse_args()

import matplotlib
if ARGS.selftest:
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox, RadioButtons

# ------------------------------------------------------------------
#   COLOURS (match RViz / lecture figures)
# ------------------------------------------------------------------
C_GOAL, C_OBS, C_RING = '#3cb44b', '#e6194b', '#f58231'
C_PATH, C_QUIV, C_START = '#1f4e79', '#9aa7b4', '#222222'

# ------------------------------------------------------------------
#   FIELD EQUATIONS (identical to potential_field_2d.py)
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
        total += k_rot * (1.0/d - 1.0/d0) * np.array([-dirn[1], dirn[0]])
    return total

def total_force(pos, goal, obstacles, d0, k_att, k_rep, k_rot, rot):
    return (attractive_force(pos, goal, k_att)
            + repulsive_force(pos, obstacles, d0, k_rep)
            + rotational_force(pos, obstacles, d0, k_rot, rot))

def integrate_path(goal, obstacles, d0, k_att, k_rep, k_rot, rot,
                   start=(0.0, 0.0), step=0.005, max_steps=8000,
                   goal_tol=0.1, stall_tol=0.03):
    pos = np.array(start, dtype=float)
    path = [pos.copy()]
    outcome = 'trapped'
    for _ in range(max_steps):
        if np.linalg.norm(np.asarray(goal) - pos) < goal_tol:
            outcome = 'reached'
            break
        F = total_force(pos, goal, obstacles, d0, k_att, k_rep, k_rot, rot)
        n = np.linalg.norm(F)
        if n < stall_tol:
            outcome = 'stuck'
            break
        pos = pos + (F / n) * step
        path.append(pos.copy())
    return np.array(path), outcome

# ------------------------------------------------------------------
#   YAML EXPORT (matches the node's parameter block exactly)
# ------------------------------------------------------------------
def _f(v):
    """Format a float so it ALWAYS keeps a decimal point. ROS2 declares
    these parameters as doubles; an integer-looking value like '1' would
    trigger a parameter-type mismatch when the node loads the file."""
    s = f'{float(v):.6g}'
    if '.' not in s and 'e' not in s and 'E' not in s:
        s += '.0'
    return s

def scenario_yaml(name, goal, obstacles, d0, k_att, k_rep, k_rot, rot,
                  goal_tol=0.15, v_max=0.15, w_max=1.0, h_gain=1.5):
    flat = []
    for o in obstacles:
        flat += [round(float(o[0]), 3), round(float(o[1]), 3)]
    obs_str = '[' + ', '.join(_f(v) for v in flat) + ']' if flat else '[]'
    return (
        f'potential_field_2d:\n'
        f'  ros__parameters:\n'
        f'    scenario_name: "{name}"\n'
        f'    k_att: {_f(k_att)}\n'
        f'    k_rep: {_f(k_rep)}\n'
        f'    influence_radius: {_f(d0)}\n'
        f'    goal_x: {_f(goal[0])}\n'
        f'    goal_y: {_f(goal[1])}\n'
        f'    goal_tolerance: {_f(goal_tol)}\n'
        f'    obstacles: {obs_str}\n'
        f'    rotational_field: {"true" if rot else "false"}\n'
        f'    k_rot: {_f(k_rot)}\n'
        f'    max_lin_speed: {_f(v_max)}\n'
        f'    max_ang_speed: {_f(w_max)}\n'
        f'    heading_gain: {_f(h_gain)}\n'
    )

# ------------------------------------------------------------------
#   INTERACTIVE BUILDER
# ------------------------------------------------------------------
class Builder:
    XLIM = (-0.5, 3.0)
    YLIM = (-1.5, 1.5)
    START = (0.0, 0.0)          # robot start is fixed at the origin

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.goal = None                 # blank map: no goal yet
        self.obstacles = []              # blank map: no obstacles yet
        self.k_att, self.k_rep, self.d0, self.k_rot = 1.0, 0.5, 0.7, 0.5
        self.rot = False
        self.mode = 'Set goal'           # place the goal first

        self.fig = plt.figure(figsize=(11, 7.5))
        self.ax = self.fig.add_axes([0.06, 0.30, 0.62, 0.64])
        self._build_widgets()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.redraw()

    # ---- widgets ----
    def _build_widgets(self):
        axc = '#fafafa'
        self.s_katt = Slider(self.fig.add_axes([0.10, 0.20, 0.55, 0.025], facecolor=axc),
                             'k_att', 0.1, 3.0, valinit=self.k_att)
        self.s_krep = Slider(self.fig.add_axes([0.10, 0.16, 0.55, 0.025], facecolor=axc),
                             'k_rep', 0.0, 2.0, valinit=self.k_rep)
        self.s_d0   = Slider(self.fig.add_axes([0.10, 0.12, 0.55, 0.025], facecolor=axc),
                             'd0', 0.2, 1.5, valinit=self.d0)
        self.s_krot = Slider(self.fig.add_axes([0.10, 0.08, 0.55, 0.025], facecolor=axc),
                             'k_rot', 0.0, 2.0, valinit=self.k_rot)
        for s in (self.s_katt, self.s_krep, self.s_d0, self.s_krot):
            s.on_changed(self.on_slider)

        # click-mode selector: place the goal, or drop obstacles
        self.radio = RadioButtons(self.fig.add_axes([0.72, 0.79, 0.24, 0.13]),
                                  ('Set goal', 'Add obstacle'), active=0)
        self.radio.on_clicked(self.on_mode)

        self.chk = CheckButtons(self.fig.add_axes([0.72, 0.70, 0.24, 0.07]),
                                ['rotational field'], [self.rot])
        self.chk.on_clicked(self.on_check)

        self.tb = TextBox(self.fig.add_axes([0.80, 0.62, 0.15, 0.05]),
                          'name  ', initial='my_scenario')

        # NOTE: keep references to the buttons. matplotlib garbage-collects
        # widget objects that are not stored, and collected widgets silently
        # stop responding to clicks.
        self.btn_save = Button(self.fig.add_axes([0.72, 0.52, 0.11, 0.06]), 'Save YAML')
        self.btn_save.on_clicked(self.on_save)
        self.btn_clear = Button(self.fig.add_axes([0.85, 0.52, 0.11, 0.06]), 'Clear all')
        self.btn_clear.on_clicked(self.on_clear)

        self.msg = self.fig.text(0.72, 0.45, '', fontsize=9, color=C_PATH,
                                 va='top', wrap=True)

    # ---- events ----
    def on_slider(self, _):
        self.k_att, self.k_rep = self.s_katt.val, self.s_krep.val
        self.d0, self.k_rot = self.s_d0.val, self.s_krot.val
        self.redraw()

    def on_check(self, _):
        self.rot = self.chk.get_status()[0]
        self.redraw()

    def on_mode(self, label):
        self.mode = label

    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        p = np.array([event.xdata, event.ydata])
        if event.button == 3:                      # right click: remove nearest obstacle
            if self.obstacles:
                d = [np.linalg.norm(p - o) for o in self.obstacles]
                if min(d) < 0.4:
                    self.obstacles.pop(int(np.argmin(d)))
        elif event.button == 1:                    # left click: depends on mode
            if self.mode == 'Set goal':
                self.goal = p
            else:
                self.obstacles.append(p)
        self.redraw()

    def on_clear(self, _):
        self.obstacles = []
        self.goal = None
        self.msg.set_text('')
        self.redraw()

    def on_save(self, _):
        if self.goal is None:
            self.msg.set_text('set a goal before saving (mode: Set goal)')
            self.fig.canvas.draw_idle()
            return
        name = (self.tb.text or 'my_scenario').strip().replace(' ', '_')
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f'{name}.yaml')
        with open(path, 'w') as f:
            f.write(scenario_yaml(name, self.goal, self.obstacles, self.d0,
                                  self.k_att, self.k_rep, self.k_rot, self.rot))
        self.msg.set_text(f'saved: {path}')
        self.fig.canvas.draw_idle()

    # ---- field with an optional goal ----
    def _force(self, pos):
        f = (repulsive_force(pos, self.obstacles, self.d0, self.k_rep)
             + rotational_force(pos, self.obstacles, self.d0, self.k_rot, self.rot))
        if self.goal is not None:
            f = f + attractive_force(pos, self.goal, self.k_att)
        return f

    def _path(self):
        pos = np.array(self.START, dtype=float)
        path = [pos.copy()]
        outcome = 'trapped'
        for _ in range(8000):
            if np.linalg.norm(np.asarray(self.goal) - pos) < 0.1:
                outcome = 'reached'; break
            F = self._force(pos); n = np.linalg.norm(F)
            if n < 0.03:
                outcome = 'stuck'; break
            pos = pos + (F / n) * 0.005
            path.append(pos.copy())
        return np.array(path), outcome

    # ---- draw ----
    def redraw(self):
        ax = self.ax
        ax.clear()
        # force field (direction only), drawn from whatever is on the map
        qn = 20
        qx, qy = np.meshgrid(np.linspace(*self.XLIM, qn),
                             np.linspace(*self.YLIM, qn))
        U = np.zeros_like(qx); V = np.zeros_like(qy)
        for i in range(qn):
            for j in range(qn):
                F = self._force(np.array([qx[j, i], qy[j, i]]))
                n = np.linalg.norm(F)
                if n > 1e-6:
                    U[j, i], V[j, i] = F / n
        ax.quiver(qx, qy, U, V, color=C_QUIV, alpha=0.8, scale=40, width=0.003)

        # obstacles + influence rings
        for o in self.obstacles:
            ax.add_patch(plt.Circle(o, self.d0, fill=False, ec=C_RING,
                                    ls='--', lw=1.1, alpha=0.9))
        if self.obstacles:
            ax.scatter([o[0] for o in self.obstacles],
                       [o[1] for o in self.obstacles],
                       s=130, c=C_OBS, edgecolors='white', zorder=5)

        # fixed start
        ax.scatter([self.START[0]], [self.START[1]], s=70, c=C_START, zorder=6)
        ax.annotate('start', self.START, textcoords='offset points',
                    xytext=(6, 6), fontsize=8, color=C_START)

        # goal + path (only once a goal exists)
        if self.goal is not None:
            ax.scatter([self.goal[0]], [self.goal[1]], marker='*', s=380,
                       c=C_GOAL, edgecolors='white', zorder=6)
            path, outcome = self._path()
            ax.plot(path[:, 0], path[:, 1], color=C_PATH, lw=2.2, zorder=4)
            tag = {'reached': 'reaches goal', 'stuck': 'stuck (local min)',
                   'trapped': 'trapped'}[outcome]
            tc = C_GOAL if outcome == 'reached' else C_OBS
            ax.text(0.02, 0.02, tag, transform=ax.transAxes, fontsize=10, color=tc,
                    va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                           ec=tc, lw=0.8))
        else:
            ax.text(0.02, 0.02, 'click to place the goal', transform=ax.transAxes,
                    fontsize=10, color=C_GOAL, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GOAL, lw=0.8))

        ax.set_xlim(*self.XLIM); ax.set_ylim(*self.YLIM)
        ax.set_aspect('equal')
        ax.grid(True, color='#f0f0f0', lw=0.6); ax.set_axisbelow(True)
        ax.set_title(f'mode: {self.mode}    (left click places it, '
                     f'right click removes an obstacle)', fontsize=10)
        self.fig.canvas.draw_idle()


def run_selftest(out_dir):
    goal = (2.0, 0.0)
    obstacles = [(1.0, 0.3), (1.0, -0.3)]
    y = scenario_yaml('selftest_gap', goal, obstacles, 0.7, 1.0, 0.5, 0.5, False)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'selftest_gap.yaml')
    with open(path, 'w') as f:
        f.write(y)
    _, outcome = integrate_path(goal, obstacles, 0.7, 1.0, 0.5, 0.5, False)
    print('--- exported YAML ---')
    print(y)
    print(f'integrated path outcome: {outcome}')
    # verify it parses if PyYAML is present
    try:
        import yaml
        d = yaml.safe_load(open(path))
        p = d['potential_field_2d']['ros__parameters']
        assert p['obstacles'] == [1.0, 0.3, 1.0, -0.3]
        assert p['rotational_field'] is False
        print('YAML re-parsed OK, obstacle list and flags match.')
    except ImportError:
        print('(PyYAML not installed here; skipped re-parse check)')
    print(f'wrote {path}')


def main():
    if ARGS.selftest:
        run_selftest(ARGS.out)
        return
    builder = Builder(ARGS.out)   # keep a reference so widgets stay alive
    plt.show()


if __name__ == '__main__':
    main()
