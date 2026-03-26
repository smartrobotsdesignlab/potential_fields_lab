#!/usr/bin/env python3
"""
Potential Fields Lab — Result Plotter (clean white style)
=========================================================
Usage:
  Single experiment:
    python3 plot_results.py --exp exp1_baseline

  Compare multiple:
    python3 plot_results.py --compare exp1_baseline exp2_no_damping exp3_weak_repulsion exp4_strong_repulsion

  Energy landscape only:
    python3 plot_results.py --landscape --k_att 2.0 --k_rep 0.8 --d0 1.5 --d_goal 0.5

CSV columns expected: time_s, distance_m, f_att, f_rep, f_total, velocity_ms, u_att, u_rep
Logs directory: ~/pf_logs/
"""

import os
import sys
import glob
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Style constants ────────────────────────────────────────────────────
COLORS = {
    "distance":  "#1565C0",   # blue
    "velocity":  "#6A1B9A",   # purple
    "f_att":     "#2E7D32",   # green
    "f_rep":     "#C62828",   # red
    "f_total":   "#E65100",   # orange
    "u_att":     "#2E7D32",
    "u_rep":     "#C62828",
    "u_total":   "#1565C0",
    "eq":        "#6A1B9A",   # purple dashed — equilibrium
    "goal":      "#2E7D32",   # green dashed — d_goal
    "d0":        "#9E9E9E",   # grey dashed — influence radius
    "zero":      "#9E9E9E",   # zero reference line
}

# Experiment display names
EXP_LABELS = {
    "exp1_baseline":         "Exp 1 — Baseline",
    "exp2_no_damping":       "Exp 2 — No Damping",
    "exp3_weak_repulsion":   "Exp 3 — Weak Repulsion",
    "exp4_strong_repulsion": "Exp 4 — Strong Repulsion",
    "exp5_challenge":        "Exp 5 — Challenge",
}

# Colors for comparison plot (one per experiment)
COMPARE_COLORS = ["#1565C0", "#C62828", "#2E7D32", "#E65100", "#6A1B9A"]

LOGS_DIR = os.path.expanduser("~/pf_logs")


# ── Helpers ────────────────────────────────────────────────────────────
def apply_style():
    """Global matplotlib style — clean white, minimal."""
    plt.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       "#CCCCCC",
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "grid.color":           "#EEEEEE",
        "grid.linewidth":       0.6,
        "grid.linestyle":       "-",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "xtick.color":          "#555555",
        "ytick.color":          "#555555",
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "axes.labelsize":       10,
        "axes.labelcolor":      "#333333",
        "axes.titlesize":       11,
        "axes.titleweight":     "bold",
        "axes.titlecolor":      "#222222",
        "legend.fontsize":      9,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
        "legend.frameon":       True,
        "figure.dpi":           150,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "font.family":          "sans-serif",
        "font.size":            9,
        "lines.linewidth":      1.8,
    })


def find_latest_csv(exp_name):
    """Find the most recent CSV for a given experiment name."""
    pattern = os.path.join(LOGS_DIR, f"{exp_name}_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No CSV found for '{exp_name}' in {LOGS_DIR}\n"
            f"  Looked for: {pattern}\n"
            f"  Run the experiment first."
        )
    return max(files, key=os.path.getmtime)


def load_csv(csv_path):
    """Load experiment CSV into a dict of numpy arrays."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {
        "time":     data["time_s"],
        "distance": data["distance_m"],
        "f_att":    data["f_att"],
        "f_rep":    data["f_rep"],
        "f_total":  data["f_total"],
        "velocity": data["velocity_ms"],
        "u_att":    data["u_att"],
        "u_rep":    data["u_rep"],
    }


def load_metadata(csv_path):
    """Load matching metadata JSON if it exists."""
    meta_path = csv_path.replace(".csv", "_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def find_equilibrium(d, f_total):
    """Find equilibrium distance where F_total crosses zero from positive."""
    for i in range(1, len(f_total)):
        if f_total[i - 1] > 0 and f_total[i] <= 0:
            return d[i]
    # fallback — return distance at minimum |F_total| in second half
    half = len(d) // 2
    idx = np.argmin(np.abs(f_total[half:])) + half
    return d[idx]


def param_string(meta):
    """Build a compact parameter string from metadata."""
    keys = [
        ("k_att", "k_att"), ("k_rep", "k_rep"),
        ("k_damp", "k_damp"), ("influence_radius", "d₀"),
        ("goal_distance", "d_goal"),
    ]
    parts = []
    for key, label in keys:
        if key in meta:
            parts.append(f"{label}={meta[key]}")
    return "  |  ".join(parts) if parts else ""


# ── Single experiment plot ─────────────────────────────────────────────
def plot_single(exp_name, save=True):
    csv_path = find_latest_csv(exp_name)
    d = load_csv(csv_path)
    meta = load_metadata(csv_path)

    d_goal = meta.get("goal_distance", 0.5)
    d0     = meta.get("influence_radius", 1.5)
    t      = d["time"]
    dist   = d["distance"]

    # Find equilibrium
    eq_dist = find_equilibrium(dist, d["f_total"])

    label = EXP_LABELS.get(exp_name, exp_name)
    params = param_string(meta)

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f"{label}\n{params}", fontsize=11, fontweight="bold", y=0.98)

    # ── Panel 1: Distance ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, dist, color=COLORS["distance"], lw=2, label="Distance to obstacle")
    ax.axhline(d_goal, color=COLORS["goal"], lw=1.2, ls="--", label=f"d_goal = {d_goal} m")
    ax.axhline(d0,     color=COLORS["d0"],   lw=1.0, ls=":",  label=f"d₀ = {d0} m")
    ax.axhline(eq_dist, color=COLORS["eq"],  lw=1.2, ls="--", label=f"Equilibrium ≈ {eq_dist:.2f} m")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance to Obstacle")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)

    # ── Panel 2: Velocity ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, d["velocity"], color=COLORS["velocity"], lw=2)
    ax.axhline(0, color=COLORS["zero"], lw=0.8, ls="-")
    ax.fill_between(t, d["velocity"], 0,
                    where=np.array(d["velocity"]) >= 0,
                    alpha=0.12, color=COLORS["distance"], label="Forward")
    ax.fill_between(t, d["velocity"], 0,
                    where=np.array(d["velocity"]) < 0,
                    alpha=0.12, color=COLORS["f_rep"], label="Reverse")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity")
    ax.legend(loc="upper right")

    # ── Panel 3: Forces ─────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, d["f_att"],   color=COLORS["f_att"],   lw=1.8, label="F_att (attractive)")
    ax.plot(t, d["f_rep"],   color=COLORS["f_rep"],   lw=1.8, label="F_rep (repulsive)")
    ax.plot(t, d["f_total"], color=COLORS["f_total"], lw=2.2, label="F_total", zorder=5)
    ax.axhline(0, color=COLORS["zero"], lw=0.8, ls="-")
    ax.set_ylabel("Force")
    ax.set_xlabel("Time (s)")
    ax.set_title("Forces")
    ax.legend(loc="upper right", ncol=3)

    for ax in axes:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out = os.path.join(LOGS_DIR, f"{exp_name}_plot.pdf")
        fig.savefig(out)
        print(f"Saved: {out}")

        out_png = os.path.join(LOGS_DIR, f"{exp_name}_plot.png")
        fig.savefig(out_png)
        print(f"Saved: {out_png}")

    plt.show()
    return fig


# ── Comparison plot ────────────────────────────────────────────────────
def plot_compare(exp_names, save=True):
    datasets = {}
    for name in exp_names:
        try:
            csv_path = find_latest_csv(name)
            datasets[name] = {
                "data": load_csv(csv_path),
                "meta": load_metadata(csv_path),
            }
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not datasets:
        print("No data found. Run experiments first.")
        return

    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("Experiment Comparison — Potential Fields Lab",
                 fontsize=12, fontweight="bold")

    # ── Panel 1: Distance ───────────────────────────────────────────
    ax = axes[0]
    for i, (name, ds) in enumerate(datasets.items()):
        t    = ds["data"]["time"]
        dist = ds["data"]["distance"]
        col  = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        lbl  = EXP_LABELS.get(name, name)
        # Trim to same length
        ax.plot(t, dist, color=col, lw=2.0, label=lbl)
        # Mark equilibrium with a small dot at the end
        eq = find_equilibrium(dist, ds["data"]["f_total"])
        ax.axhline(eq, color=col, lw=0.6, ls="--", alpha=0.5)

    # Goal line from first dataset
    first_meta = list(datasets.values())[0]["meta"]
    d_goal = first_meta.get("goal_distance", 0.5)
    ax.axhline(d_goal, color=COLORS["goal"], lw=1.2, ls="--",
               label=f"d_goal = {d_goal} m", zorder=10)

    ax.set_ylabel("Distance to obstacle (m)")
    ax.set_title("Distance vs Time")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)

    # ── Panel 2: Velocity ───────────────────────────────────────────
    ax = axes[1]
    for i, (name, ds) in enumerate(datasets.items()):
        t   = ds["data"]["time"]
        vel = ds["data"]["velocity"]
        col = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        lbl = EXP_LABELS.get(name, name)
        ax.plot(t, vel, color=col, lw=2.0, label=lbl)

    ax.axhline(0, color=COLORS["zero"], lw=0.8)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Velocity vs Time")
    ax.legend(loc="upper right", ncol=2)

    for ax in axes:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        out = os.path.join(LOGS_DIR, "comparison_plot.pdf")
        fig.savefig(out)
        print(f"Saved: {out}")

        out_png = os.path.join(LOGS_DIR, "comparison_plot.png")
        fig.savefig(out_png)
        print(f"Saved: {out_png}")

    plt.show()
    return fig


# ── Energy landscape plot ──────────────────────────────────────────────
def plot_landscape(k_att=2.0, k_rep=0.8, d0=1.5, d_goal=0.5, save=True):
    d = np.linspace(0.15, 3.0, 2000)
    d_safe = np.maximum(d, 1e-6)

    u_att = 0.5 * k_att * (d - d_goal) ** 2
    u_rep = np.where(d < d0, 0.5 * k_rep * (1.0 / d_safe - 1.0 / d0) ** 2, 0)
    u_total = u_att + u_rep

    f_att = k_att * (d - d_goal)
    f_rep = np.where(d < d0, -k_rep * (1.0 / d_safe - 1.0 / d0) * (1.0 / d_safe ** 2), 0)
    f_total = f_att + f_rep

    # Find equilibrium
    eq_idx = np.argmin(np.abs(f_total[d > 0.3]))
    d_eq = d[d > 0.3][eq_idx]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    params = f"k_att={k_att}  |  k_rep={k_rep}  |  d₀={d0} m  |  d_goal={d_goal} m"
    fig.suptitle(f"Potential Field Landscape\n{params}",
                 fontsize=11, fontweight="bold")

    # ── Left: Potential energy ──────────────────────────────────────
    ax = axes[0]
    ax.plot(d, np.minimum(u_att, 6),   color=COLORS["u_att"],   lw=1.8, label="U_att")
    ax.plot(d, np.minimum(u_rep, 6),   color=COLORS["u_rep"],   lw=1.8, label="U_rep")
    ax.plot(d, np.minimum(u_total, 6), color=COLORS["u_total"], lw=2.2, label="U_total")
    ax.axvline(d_goal, color=COLORS["goal"], lw=1.2, ls="--", label=f"d_goal={d_goal}")
    ax.axvline(d0,     color=COLORS["d0"],   lw=1.0, ls=":",  label=f"d₀={d0}")
    ax.axvline(d_eq,   color=COLORS["eq"],   lw=1.2, ls="--", label=f"d_eq≈{d_eq:.2f}")
    ax.scatter([d_eq], [np.minimum(u_total[d > 0.3][eq_idx], 6)],
               color=COLORS["eq"], s=50, zorder=10)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Distance to obstacle (m)")
    ax.set_ylabel("Potential energy U")
    ax.set_title("Energy Landscape U(d)")
    ax.legend(fontsize=8, ncol=2)

    # ── Right: Forces ───────────────────────────────────────────────
    ax = axes[1]
    ax.plot(d, np.clip(f_att,   -3, 3), color=COLORS["f_att"],   lw=1.8, label="F_att")
    ax.plot(d, np.clip(f_rep,   -3, 3), color=COLORS["f_rep"],   lw=1.8, label="F_rep")
    ax.plot(d, np.clip(f_total, -3, 3), color=COLORS["f_total"], lw=2.2, label="F_total")
    ax.axhline(0,       color=COLORS["zero"], lw=0.8)
    ax.axvline(d_goal,  color=COLORS["goal"], lw=1.2, ls="--", label=f"d_goal={d_goal}")
    ax.axvline(d0,      color=COLORS["d0"],   lw=1.0, ls=":",  label=f"d₀={d0}")
    ax.axvline(d_eq,    color=COLORS["eq"],   lw=1.2, ls="--", label=f"d_eq≈{d_eq:.2f}")
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Distance to obstacle (m)")
    ax.set_ylabel("Force")
    ax.set_title("Force Landscape F(d)")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        tag = f"landscape_katt{k_att}_krep{k_rep}"
        out = os.path.join(LOGS_DIR, f"{tag}.pdf")
        fig.savefig(out)
        print(f"Saved: {out}")
        out_png = os.path.join(LOGS_DIR, f"{tag}.png")
        fig.savefig(out_png)
        print(f"Saved: {out_png}")

    plt.show()
    return fig


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Potential Fields Lab — Plot Results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp",       type=str, help="Single experiment name")
    group.add_argument("--compare",   nargs="+", help="List of experiments to compare")
    group.add_argument("--landscape", action="store_true", help="Plot potential field landscape")

    parser.add_argument("--k_att",  type=float, default=2.0)
    parser.add_argument("--k_rep",  type=float, default=0.8)
    parser.add_argument("--d0",     type=float, default=1.5)
    parser.add_argument("--d_goal", type=float, default=0.5)
    parser.add_argument("--no-save", action="store_true", help="Do not save to file")

    args = parser.parse_args()
    save = not args.no_save

    if args.exp:
        plot_single(args.exp, save=save)
    elif args.compare:
        plot_compare(args.compare, save=save)
    elif args.landscape:
        plot_landscape(args.k_att, args.k_rep, args.d0, args.d_goal, save=save)


if __name__ == "__main__":
    main()
