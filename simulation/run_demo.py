"""Demo runner & figure generator for the frog-chorus simulator.

Usage:
    python run_demo.py            # run demos, write PNGs to out/
    python run_demo.py --show     # also show windows (GUI env)

Outputs (out/):
    two_frogs.png     two oscillators converging to anti-phase (pi)
    chain_frogs.png   1-D chain (frogs along a paddy edge): nearest-neighbor alternation
    field_frogs.png   2-D random field: frustrated -> dynamic local clusters
    coupling_off.png  no coupling baseline for comparison

Labels are kept in English on purpose so the PNGs render without a CJK font.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

from frog_chorus import (
    Config,
    nearest_neighbor_phase_diffs,
    phase_difference_series,
    simulate,
)

OUT = os.path.join(os.path.dirname(__file__), "out")


def _raster(ax, res, title, tail=None):
    if res.call_times:
        ts, ids = zip(*res.call_times)
        ax.scatter(ts, ids, s=10, marker="|", linewidths=1.3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frog id")
    ax.set_title(title)
    ax.set_ylim(-0.5, res.config.n - 0.5)
    if tail:
        ax.set_xlim(res.config.duration - tail, res.config.duration)


def demo_two_frogs(show):
    import matplotlib.pyplot as plt

    cfg = Config(n=2, duration=120.0, coupling=0.15, prc_sign=+1.0,
                 period_spread=0.0, loudness_gate=0.05, seed=1)
    res = simulate(cfg)

    diff = phase_difference_series(res, 0, 1)
    diff_folded = np.minimum(diff, 1.0 - diff)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[1, 1.4])
    _raster(ax1, res, "2 frogs: call raster (alternating = anti-phase)")

    ax2.plot(res.hist_t, diff_folded, lw=1.5)
    ax2.axhline(0.5, color="crimson", ls="--", lw=1, label="anti-phase (pi)")
    ax2.axhline(0.0, color="gray", ls=":", lw=1, label="in-phase")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("|phase difference| (0..0.5)")
    ax2.set_ylim(-0.02, 0.52)
    ax2.set_title("phase difference converges to 0.5 = anti-phase sync")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, "two_frogs.png", show)
    final = float(np.mean(diff_folded[-len(diff_folded) // 5:]))
    print(f"[two_frogs] mean phase diff (tail) = {final:.3f}  (1/2 = anti-phase)")


def demo_chain(show):
    import matplotlib.pyplot as plt

    # Frogs aggregate along the edge of a paddy field (Aihara's field observation).
    # 1-D chain -> nearest-neighbor coupling is (near-)bipartite -> clear alternation.
    n = 10
    pos = np.column_stack([np.arange(n) * 1.0, np.zeros(n)])
    cfg = Config(n=n, duration=160.0, coupling=0.20, prc_sign=+1.0,
                 period_mean=2.0, period_spread=0.04,
                 ref_distance=1.0, loudness_gate=0.3,   # gate ~ only immediate neighbors
                 coupling_mode="all", refractory=0.35,
                 detection_latency=0.1, positions=pos, seed=5)
    res = simulate(cfg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _raster(ax1, res, "10 frogs in a line: raster (last 40 s)", tail=40)

    nn = nearest_neighbor_phase_diffs(res, tail_fraction=0.5)
    ax2.hist(nn, bins=25, range=(0, 0.5), color="teal", alpha=0.85)
    ax2.axvline(0.5, color="crimson", ls="--", lw=1, label="anti-phase (pi)")
    ax2.axvline(0.25, color="gray", ls=":", lw=1, label="random baseline")
    ax2.set_xlabel("|phase difference| of nearest-neighbor pairs")
    ax2.set_ylabel("count")
    ax2.set_title("nearest-neighbor phase diff (peak near 0.5 = alternation)")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "chain_frogs.png", show)
    print(f"[chain] nearest-neighbor phase-diff median = {np.median(nn):.3f}")


def demo_field(show):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 4, size=(14, 2))
    cfg = Config(n=14, duration=160.0, coupling=0.20, prc_sign=+1.0,
                 period_mean=2.0, period_spread=0.06,
                 ref_distance=1.0, loudness_gate=0.3,
                 coupling_mode="all", refractory=0.35,
                 detection_latency=0.1, positions=pos, seed=3)
    res = simulate(cfg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _raster(ax1, res, "14 frogs in 2-D field: raster (last 40 s)", tail=40)

    nn = nearest_neighbor_phase_diffs(res, tail_fraction=0.5)
    ax2.hist(nn, bins=25, range=(0, 0.5), color="indigo", alpha=0.8)
    ax2.axvline(0.5, color="crimson", ls="--", lw=1, label="anti-phase (pi)")
    ax2.axvline(0.25, color="gray", ls=":", lw=1, label="random baseline")
    ax2.set_xlabel("|phase difference| of nearest-neighbor pairs")
    ax2.set_ylabel("count")
    ax2.set_title("frustrated field: biased to 0.5 but not perfect")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "field_frogs.png", show)
    print(f"[field] nearest-neighbor phase-diff median = {np.median(nn):.3f}")


def demo_coupling_off(show):
    import matplotlib.pyplot as plt

    n = 10
    pos = np.column_stack([np.arange(n) * 1.0, np.zeros(n)])
    cfg = Config(n=n, duration=160.0, coupling=0.0,
                 period_mean=2.0, period_spread=0.04, positions=pos, seed=5)
    res = simulate(cfg)

    fig, ax = plt.subplots(figsize=(9, 5))
    _raster(ax, res, "coupling OFF: each frog calls on its own (baseline)", tail=40)
    fig.tight_layout()
    _save(fig, "coupling_off.png", show)


def _save(fig, name, show):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110)
    print(f"saved: {path}")
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    demo_two_frogs(args.show)
    demo_chain(args.show)
    demo_field(args.show)
    demo_coupling_off(args.show)


if __name__ == "__main__":
    main()
