#!/usr/bin/env python3
"""
Potential Fields Lab — Results Plotter
============================================================
Reads CSV log files and generates a comprehensive plot
showing distance, forces, velocity, phase portrait,
and potential energy landscape.

Usage:
  # Plot most recent log file
  python3 scripts/plot_results.py

  # Plot specific experiment
  python3 scripts/plot_results.py --exp exp2_no_damping

  # Compare two experiments
  python3 scripts/plot_results.py --compare exp1_baseline exp2_no_damping

  # Plot potential field landscape (no robot data needed)
  python3 scripts/plot_results.py --landscape
============================================================
"""

import argparse
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import csv

# ── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#0A0E1A',
    'axes.facecolor': '#0F1525',
    'axes.labelcolor': '#E2E8F0',
    'xtick.color': '#64748B',
    'ytick.color': '#64748B',
    'grid.color': '#1E2A42',
    'text.color': '#E2E8F0',
    'axes.titlecolor': '#3B82F6',
})

COLORS = {
    'distance': '#3B82F6',
    'f_att':    '#10B981',
    'f_rep':    '#EF4444',
    'f_total':  '#F59E0B',
    'velocity': '#8B5CF6',
    'u_att':    '#10B981',
    'u_rep':    '#EF4444',
    'u_total':  '#3B82F6',
    'phase':    '#06B6D4',
    'goal':     '#22C55E',
    'min_dist': '#EF4444',
}

LOG_DIR = os.path.expanduser('~/pf_logs')


# ── Data Loading ─────────────────────────────────────────────────────
def find_latest_log(exp_name=None):
    pattern = os.path.join(
        LOG_DIR,
        f'{exp_name}_*.csv' if exp_name else '*.csv'
    )
    files = glob.glob(pattern)
    files = [f for f in files if 'metadata' not in f]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_csv(filepath):
    data = {
        'time': [], 'distance': [], 'f_att': [],
        'f_rep': [], 'f_total': [], 'velocity': [],
        'u_att': [], 'u_rep': []
    }
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data['time'].append(float(row['time_s']))
                data['distance'].append(float(row['distance_m']))
                data['f_att'].append(float(row['f_att']))
                data['f_rep'].append(float(row['f_rep']))
                data['f_total'].append(float(row['f_total']))
                data['velocity'].append(float(row['velocity_ms']))
                data['u_att'].append(float(row['u_att']))
                data['u_rep'].append(float(row['u_rep']))
            except (ValueError, KeyError):
                continue
    return {k: np.array(v) for k, v in data.items()}


def load_metadata(filepath):
    meta_path = filepath.replace('.csv', '_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


# ── Potential Field Landscape ─────────────────────────────────────────
def compute_landscape(params, d_range=(0.15, 3.0)):
    k_att   = params.get('k_att', 2.0)
    k_rep   = params.get('k_rep', 0.8)
    d0      = params.get('d0', 1.5)
    d_goal  = params.get('d_goal', 0.5)

    d = np.linspace(d_range[0], d_range[1], 500)

    u_att = 0.5 * k_att * (d - d_goal)**2

    u_rep = np.zeros_like(d)
    mask  = d < d0
    d_s   = np.maximum(d[mask], 0.01)
    u_rep[mask] = 0.5 * k_rep * (1.0/d_s - 1.0/d0)**2

    u_total = u_att + u_rep

    f_att = k_att * (d - d_goal)
    f_rep = np.zeros_like(d)
    f_rep[mask] = -k_rep * (1.0/d_s - 1.0/d0) * (1.0/d_s**2)
    f_total = f_att + f_rep

    return d, u_att, u_rep, u_total, f_att, f_rep, f_total


# ── Plot Functions ────────────────────────────────────────────────────
def plot_full_report(data, meta, exp_name):
    """6-panel full report plot."""
    params = meta.get('parameters', {})
    d_goal = params.get('d_goal', 0.5)
    d0     = params.get('d0', 1.5)
    k_damp = params.get('k_damp', 1.5)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Potential Fields Lab — {exp_name}\n'
        f'k_att={params.get("k_att","?")}  '
        f'k_rep={params.get("k_rep","?")}  '
        f'k_damp={k_damp}  '
        f'd0={d0}m  '
        f'd_goal={d_goal}m',
        fontsize=12, fontweight='bold', color='#3B82F6'
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    t = data['time']
    d = data['distance']

    # ── Panel 1: Distance vs Time ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, d, color=COLORS['distance'], lw=2, label='Distance')
    ax1.axhline(d_goal, color=COLORS['goal'], ls='--', lw=1.5,
                label=f'Goal ({d_goal}m)')
    ax1.axhline(0.25, color=COLORS['min_dist'], ls=':', lw=1,
                label='Min distance (0.25m)')
    ax1.axhline(d0, color='#F59E0B', ls='-.', lw=1,
                label=f'Influence radius ({d0}m)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Distance to Obstacle vs Time')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylim(0, max(d) * 1.2)

    # Annotate equilibrium
    if len(d) > 20:
        eq_d = np.mean(d[-20:])
        ax1.annotate(
            f'Equilibrium\n≈ {eq_d:.3f}m',
            xy=(t[-1], eq_d),
            xytext=(t[-1] * 0.7, eq_d + 0.2),
            arrowprops=dict(arrowstyle='->', color='#E2E8F0'),
            color='#E2E8F0', fontsize=8
        )

    # ── Panel 2: Velocity vs Time ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(t, data['velocity'], color=COLORS['velocity'],
             lw=2, label='Velocity')
    ax2.axhline(0, color='#64748B', ls='-', lw=0.8)
    ax2.fill_between(t, data['velocity'], 0,
                     where=[v > 0 for v in data['velocity']],
                     alpha=0.2, color=COLORS['f_att'],
                     label='Forward')
    ax2.fill_between(t, data['velocity'], 0,
                     where=[v < 0 for v in data['velocity']],
                     alpha=0.2, color=COLORS['f_rep'],
                     label='Backward')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time  — Spring-Damper Response')
    ax2.legend(fontsize=8)

    # ── Panel 3: Forces vs Time ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(t, data['f_att'],   color=COLORS['f_att'],
             lw=1.5, label='F_att (attractive)')
    ax3.plot(t, data['f_rep'],   color=COLORS['f_rep'],
             lw=1.5, label='F_rep (repulsive)')
    ax3.plot(t, data['f_total'], color=COLORS['f_total'],
             lw=2, label='F_total', zorder=5)
    ax3.axhline(0, color='#64748B', ls='-', lw=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force')
    ax3.set_title('Forces vs Time')
    ax3.legend(fontsize=8)

    # ── Panel 4: Phase Portrait (v vs d) ──────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    sc = ax4.scatter(d, data['velocity'],
                     c=t, cmap='plasma', s=8, alpha=0.8)
    ax4.axvline(d_goal, color=COLORS['goal'], ls='--', lw=1)
    ax4.axhline(0, color='#64748B', ls='-', lw=0.8)
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Phase Portrait\n(v vs d)')
    plt.colorbar(sc, ax=ax4, label='Time (s)', pad=0.02)

    # Mark start and end
    ax4.plot(d[0], data['velocity'][0], 'o',
             color='#22C55E', ms=8, label='Start', zorder=10)
    ax4.plot(d[-1], data['velocity'][-1], 's',
             color='#EF4444', ms=8, label='End', zorder=10)
    ax4.legend(fontsize=7)

    # ── Panel 5: Potential Energy Landscape ───────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    d_range, u_att, u_rep, u_total, _, _, _ = compute_landscape(params)
    ax5.plot(d_range, np.minimum(u_att, 8),
             color=COLORS['u_att'], lw=1.5, label='U_att')
    ax5.plot(d_range, np.minimum(u_rep, 8),
             color=COLORS['u_rep'], lw=1.5, label='U_rep')
    ax5.plot(d_range, np.minimum(u_total, 8),
             color=COLORS['u_total'], lw=2, label='U_total')
    ax5.axvline(d_goal, color=COLORS['goal'], ls='--', lw=1)
    ax5.axvline(d0, color='#F59E0B', ls='-.', lw=1)
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Potential Energy')
    ax5.set_title('Energy Landscape\nU(d)')
    ax5.set_ylim(0, 6)
    ax5.legend(fontsize=7)

    # ── Panel 6: Force vs Distance ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    _, _, _, _, f_att_c, f_rep_c, f_total_c = compute_landscape(params)
    ax6.plot(d_range, np.clip(f_att_c, -3, 3),
             color=COLORS['f_att'], lw=1.5, label='F_att')
    ax6.plot(d_range, np.clip(f_rep_c, -3, 3),
             color=COLORS['f_rep'], lw=1.5, label='F_rep')
    ax6.plot(d_range, np.clip(f_total_c, -3, 3),
             color=COLORS['f_total'], lw=2, label='F_total')
    ax6.axhline(0, color='#64748B', ls='-', lw=0.8)
    ax6.axvline(d_goal, color=COLORS['goal'], ls='--', lw=1,
                label=f'Goal ({d_goal}m)')
    ax6.axvline(d0, color='#F59E0B', ls='-.', lw=1,
                label=f'd0 ({d0}m)')

    # Mark equilibrium on force plot
    zero_crossings = np.where(np.diff(np.sign(f_total_c)))[0]
    for zc in zero_crossings:
        if d_range[zc] > 0.3:
            ax6.axvline(d_range[zc], color='#EC4899',
                        ls=':', lw=1.5, label=f'Eq ≈{d_range[zc]:.2f}m')
            break

    ax6.set_xlabel('Distance (m)')
    ax6.set_ylabel('Force')
    ax6.set_title('Force Landscape\nF(d)')
    ax6.set_ylim(-2.5, 2.5)
    ax6.legend(fontsize=7)

    # ── Save ──────────────────────────────────────────────────────
    out_path = os.path.join(LOG_DIR, f'{exp_name}_report.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nPlot saved: {out_path}')
    plt.show()


def plot_comparison(exp_names):
    """Overlay distance and velocity for multiple experiments."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Potential Fields Lab — Experiment Comparison',
                 fontsize=13, fontweight='bold', color='#3B82F6')

    palette = ['#3B82F6', '#EF4444', '#10B981',
               '#F59E0B', '#8B5CF6', '#06B6D4']

    for i, exp_name in enumerate(exp_names):
        filepath = find_latest_log(exp_name)
        if not filepath:
            print(f'No log found for {exp_name}')
            continue
        data = load_csv(filepath)
        meta = load_metadata(filepath)
        params = meta.get('parameters', {})
        color  = palette[i % len(palette)]
        label  = (f"{exp_name} "
                  f"(k_damp={params.get('k_damp','?')})")

        ax1.plot(data['time'], data['distance'],
                 color=color, lw=2, label=label)
        ax2.plot(data['time'], data['velocity'],
                 color=color, lw=2, label=label)

    # Get goal from first experiment
    fp = find_latest_log(exp_names[0])
    if fp:
        meta   = load_metadata(fp)
        params = meta.get('parameters', {})
        d_goal = params.get('d_goal', 0.5)
        ax1.axhline(d_goal, color=COLORS['goal'],
                    ls='--', lw=1.5, label=f'Goal ({d_goal}m)')

    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Distance vs Time')
    ax1.legend(fontsize=8)

    ax2.axhline(0, color='#64748B', ls='-', lw=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time — Spring-Damper Comparison')
    ax2.legend(fontsize=8)

    out_path = os.path.join(
        LOG_DIR,
        'comparison_' + '_vs_'.join(exp_names) + '.png'
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nComparison plot saved: {out_path}')
    plt.show()


def plot_landscape_only(k_att=2.0, k_rep=0.8, d0=1.5, d_goal=0.5):
    """Plot potential field landscape without any robot data."""
    params = {'k_att': k_att, 'k_rep': k_rep,
              'd0': d0, 'd_goal': d_goal}

    d, u_att, u_rep, u_total, f_att, f_rep, f_total = \
        compute_landscape(params)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f'Potential Field Landscape\n'
        f'k_att={k_att}  k_rep={k_rep}  '
        f'd0={d0}m  d_goal={d_goal}m',
        fontsize=12, fontweight='bold', color='#3B82F6'
    )

    # Energy landscape
    ax1.plot(d, np.minimum(u_att, 8),
             color=COLORS['u_att'], lw=2, label='U_att (attractive)')
    ax1.plot(d, np.minimum(u_rep, 8),
             color=COLORS['u_rep'], lw=2, label='U_rep (repulsive)')
    ax1.plot(d, np.minimum(u_total, 8),
             color=COLORS['u_total'], lw=2.5,
             label='U_total (robot follows this)')
    ax1.axvline(d_goal, color=COLORS['goal'],
                ls='--', lw=1.5, label=f'Goal distance ({d_goal}m)')
    ax1.axvline(d0, color='#F59E0B',
                ls='-.', lw=1.5, label=f'Influence radius ({d0}m)')
    ax1.set_ylabel('Potential Energy U(d)')
    ax1.set_title('Energy Landscape — Robot rolls downhill')
    ax1.set_ylim(0, 6)
    ax1.legend()
    ax1.annotate('Robot rolls\ndownhill →',
                 xy=(2.0, 1.5), fontsize=9, color='#E2E8F0')

    # Force landscape
    ax2.plot(d, np.clip(f_att, -3, 3),
             color=COLORS['f_att'], lw=2, label='F_att (pulls forward)')
    ax2.plot(d, np.clip(f_rep, -3, 3),
             color=COLORS['f_rep'], lw=2, label='F_rep (pushes back)')
    ax2.plot(d, np.clip(f_total, -3, 3),
             color=COLORS['f_total'], lw=2.5, label='F_total')
    ax2.axhline(0, color='#64748B', ls='-', lw=0.8)
    ax2.axvline(d_goal, color=COLORS['goal'],
                ls='--', lw=1.5, label=f'Goal ({d_goal}m)')
    ax2.axvline(d0, color='#F59E0B',
                ls='-.', lw=1.5, label=f'd0 ({d0}m)')

    # Mark equilibrium
    zero_crossings = np.where(np.diff(np.sign(f_total)))[0]
    for zc in zero_crossings:
        if d[zc] > 0.3:
            ax2.axvline(d[zc], color='#EC4899', ls=':', lw=2,
                        label=f'Equilibrium ≈ {d[zc]:.3f}m')
            ax2.annotate(
                f'Equilibrium\n≈ {d[zc]:.3f}m\n(F_total = 0)',
                xy=(d[zc], 0),
                xytext=(d[zc] + 0.3, 1.0),
                arrowprops=dict(arrowstyle='->', color='#EC4899'),
                color='#EC4899', fontsize=9
            )
            break

    ax2.set_xlabel('Distance to Obstacle (m)')
    ax2.set_ylabel('Force')
    ax2.set_title('Force Landscape — Equilibrium where F_total = 0')
    ax2.set_ylim(-2.5, 2.5)
    ax2.legend()

    out_path = os.path.join(LOG_DIR, 'landscape.png')
    os.makedirs(LOG_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nLandscape plot saved: {out_path}')
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Potential Fields Lab — Results Plotter')
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment name to plot')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='Compare multiple experiments')
    parser.add_argument('--landscape', action='store_true',
                        help='Plot potential field landscape only')
    parser.add_argument('--k_att', type=float, default=2.0)
    parser.add_argument('--k_rep', type=float, default=0.8)
    parser.add_argument('--d0', type=float, default=1.5)
    parser.add_argument('--d_goal', type=float, default=0.5)
    args = parser.parse_args()

    if args.landscape:
        plot_landscape_only(args.k_att, args.k_rep, args.d0, args.d_goal)
        return

    if args.compare:
        plot_comparison(args.compare)
        return

    # Single experiment
    exp_name = args.exp
    filepath = find_latest_log(exp_name)

    if not filepath:
        print(f'No log files found in {LOG_DIR}')
        if exp_name:
            print(f'Looking for: {exp_name}_*.csv')
        print('Run the lab first to generate data.')
        return

    print(f'Loading: {filepath}')
    data = load_csv(filepath)
    meta = load_metadata(filepath)

    if len(data['time']) == 0:
        print('CSV file is empty — did the robot run?')
        return

    name = exp_name or os.path.basename(filepath).split('_2')[0]
    plot_full_report(data, meta, name)


if __name__ == '__main__':
    main()
