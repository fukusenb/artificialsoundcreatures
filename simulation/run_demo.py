"""2 層モデル(Felix Hess の内部状態 + 合原の逆相)のデモ & 図の生成.

出力 (out/):
    two_frogs.png       種火 -> もう 1 体が +T/2 逆相で参加、CALLING 中の位相差
    ignition.png        1 匹の種火から合唱が点火して広がる(active 割合の時間変化)
    human_approach.png  人(環境音)が近づくと静まり、去るとバラバラに鳴き戻る

ラベルは CJK フォント無しでも描画できるよう英語にしている。
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from frog_chorus import (  # noqa: E402
    Config,
    active_fraction_series,
    nearest_neighbor_phase_diffs,
    simulate,
)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def raster(ax, res, title, tail=None):
    if res.call_times:
        ts, ids = zip(*res.call_times)
        ax.scatter(ts, ids, s=10, marker="|", linewidths=1.3)
    ax.set_xlabel("time [s]"); ax.set_ylabel("frog id"); ax.set_title(title)
    ax.set_ylim(-0.5, res.config.n - 0.5)
    if tail:
        ax.set_xlim(res.config.duration - tail, res.config.duration)


def demo_two_frogs():
    cfg = Config(n=2, duration=60.0, period_mean=1.0, period_spread=0.0,
                 rise_rate=0.35, rise_spread=0.0, seed=2,
                 positions=np.array([[0.0, 0.0], [0.8, 0.0]]))
    res = simulate(cfg)

    both = res.hist_active[:, 0] & res.hist_active[:, 1]
    d = (res.hist_phase[:, 0] - res.hist_phase[:, 1]) % 1.0
    d = np.minimum(d, 1.0 - d)
    d_plot = np.where(both, d, np.nan)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[1, 1.3])
    raster(a1, res, "2 frogs: seed ignites, other joins in anti-phase")
    a2.plot(res.hist_t, d_plot, lw=1.4)
    a2.axhline(0.5, color="crimson", ls="--", lw=1, label="anti-phase (pi)")
    a2.set_ylim(-0.02, 0.52); a2.set_xlabel("time [s]")
    a2.set_ylabel("|phase diff| while both calling")
    a2.set_title("one-shot +T/2 set at onset, then free-run (drifts slightly)")
    a2.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "two_frogs.png"), dpi=110)
    plt.close(fig)
    print("[two_frogs] saved")


def demo_ignition():
    rng = np.random.default_rng(7)
    pos = rng.uniform(0, 4, size=(10, 2))
    cfg = Config(n=10, duration=90.0, period_mean=1.0, period_spread=0.05,
                 rise_rate=0.3, rise_spread=0.25,
                 ref_distance=1.2, loudness_gate=0.2, positions=pos, seed=7)
    res = simulate(cfg)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1.3, 1])
    raster(a1, res, "10 frogs: chorus ignites from a spontaneous seed")
    a2.plot(res.hist_t, active_fraction_series(res), lw=1.6, color="teal")
    a2.set_ylim(-0.02, 1.02); a2.set_xlabel("time [s]")
    a2.set_ylabel("fraction calling"); a2.set_title("chorus builds up over time")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ignition.png"), dpi=110)
    plt.close(fig)
    print("[ignition] saved")


def demo_human_approach():
    rng = np.random.default_rng(4)
    pos = rng.uniform(0, 4, size=(12, 2))

    # 人: 40-70 秒の間だけ中央に居て大きな非カエル音を出す
    def human_level(t):
        return 1.2 if 40.0 <= t <= 70.0 else 0.0

    cfg = Config(n=12, duration=120.0, period_mean=1.0, period_spread=0.05,
                 rise_rate=0.3, rise_spread=0.25,
                 ref_distance=1.2, loudness_gate=0.2, inh_gain=2.0,
                 positions=pos,
                 noises=[{"x": 2.0, "y": 2.0, "level": human_level, "radius": 2.5}],
                 seed=4)
    res = simulate(cfg)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1.3, 1])
    raster(a1, res, "human approaches (40-70 s): chorus falls silent, then rebuilds")
    a1.axvspan(40, 70, color="crimson", alpha=0.08)
    a2.plot(res.hist_t, active_fraction_series(res), lw=1.6, color="teal")
    a2.axvspan(40, 70, color="crimson", alpha=0.12, label="human present")
    a2.set_ylim(-0.02, 1.02); a2.set_xlabel("time [s]")
    a2.set_ylabel("fraction calling"); a2.set_title("quiet when human near, staggered recovery")
    a2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "human_approach.png"), dpi=110)
    plt.close(fig)
    nn = nearest_neighbor_phase_diffs(res, 0.3)
    print(f"[human_approach] saved; nearest-neighbor phase-diff median = "
          f"{np.median(nn) if nn.size else float('nan'):.3f}")


if __name__ == "__main__":
    demo_two_frogs()
    demo_ignition()
    demo_human_approach()
    print("done")
