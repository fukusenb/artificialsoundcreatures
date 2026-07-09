"""T = 1s (固有鳴き周期 1 秒) でのシミュレーション.

周期を短くすると 1 サイクル内の「聞く時間」が減るため、逆相がまだ成立するか、
収束が遅く/不安定にならないかを確認する。
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from frog_chorus import (  # noqa: E402
    Config,
    nearest_neighbor_phase_diffs,
    phase_difference_series,
    simulate,
)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

T = 1.0  # ★ 固有鳴き周期 1 秒


def raster(ax, res, title, tail=None):
    if res.call_times:
        ts, ids = zip(*res.call_times)
        ax.scatter(ts, ids, s=10, marker="|", linewidths=1.3)
    ax.set_xlabel("time [s]"); ax.set_ylabel("frog id"); ax.set_title(title)
    ax.set_ylim(-0.5, res.config.n - 0.5)
    if tail:
        ax.set_xlim(res.config.duration - tail, res.config.duration)


def two_frogs():
    cfg = Config(n=2, duration=60.0, period_mean=T, period_spread=0.0,
                 coupling=0.15, prc_sign=+1.0, loudness_gate=0.05,
                 refractory=0.2, detection_latency=0.1, seed=1)
    res = simulate(cfg)
    diff = phase_difference_series(res, 0, 1)
    folded = np.minimum(diff, 1.0 - diff)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[1, 1.4])
    raster(a1, res, "T=1s, 2 frogs: call raster")
    a1.set_xlim(0, 30)
    a2.plot(res.hist_t, folded, lw=1.3)
    a2.axhline(0.5, color="crimson", ls="--", lw=1, label="anti-phase (pi)")
    a2.axhline(0.0, color="gray", ls=":", lw=1, label="in-phase")
    a2.set_xlabel("time [s]"); a2.set_ylabel("|phase diff| (0..0.5)")
    a2.set_ylim(-0.02, 0.52); a2.set_title("T=1s: phase difference over time")
    a2.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "t1s_two_frogs.png"), dpi=110)
    plt.close(fig)
    tail = folded[-len(folded) // 5:]
    print(f"[T=1s two] mean |phase diff| (tail) = {np.mean(tail):.3f} "
          f"(0.5=anti-phase); std={np.std(tail):.3f}")


def chain():
    n = 10
    pos = np.column_stack([np.arange(n) * 1.0, np.zeros(n)])
    cfg = Config(n=n, duration=120.0, period_mean=T, period_spread=0.04,
                 coupling=0.2, prc_sign=+1.0, ref_distance=1.0, loudness_gate=0.3,
                 coupling_mode="all", refractory=0.2, detection_latency=0.1,
                 positions=pos, seed=5)
    res = simulate(cfg)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    raster(a1, res, "T=1s, 10 frogs in a line (last 20 s)", tail=20)
    nn = nearest_neighbor_phase_diffs(res, tail_fraction=0.5)
    a2.hist(nn, bins=25, range=(0, 0.5), color="teal", alpha=0.85)
    a2.axvline(0.5, color="crimson", ls="--", lw=1, label="anti-phase (pi)")
    a2.axvline(0.25, color="gray", ls=":", lw=1, label="random baseline")
    a2.set_xlabel("|phase diff| nearest-neighbor"); a2.set_ylabel("count")
    a2.set_title("T=1s chain: nearest-neighbor phase diff"); a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "t1s_chain.png"), dpi=110)
    plt.close(fig)
    print(f"[T=1s chain] nearest-neighbor phase-diff median = {np.median(nn):.3f}")


if __name__ == "__main__":
    two_frogs()
    chain()
    print("done")
