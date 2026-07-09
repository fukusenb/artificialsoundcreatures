"""Artificial Sound Creatures - カエル合唱の自己組織化シミュレーター.

合原モデル(ニホンアマガエルの位相振動子モデル)を、実機ハードウェアの制約
(マイク1個で混合音を聞く=個々の位相は直接取れない)に合わせて
「パルス結合振動子 + 音量ゲート + 位相応答曲線(PRC)」として実装したもの。

各個体は内部位相 phi in [0,1) を固有周期 T で進め、phi>=1 で発声(位相リセット)。
他個体の発声は距離で減衰した音量として届き、閾値以上のイベントにだけ PRC で応答する。
反発性(抑制的)の PRC により、2 体では位相差 0.5(=pi, 逆相)が安定化する。

このファイルはエンジン本体。図の生成は run_demo.py 側で行う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    n: int = 2                      # 個体数
    dt: float = 0.01                # 時間刻み [s]
    duration: float = 120.0         # シミュレーション時間 [s]

    period_mean: float = 2.0        # 固有鳴き周期の平均 [s]
    period_spread: float = 0.05     # 固有周期の個体差(相対標準偏差)

    coupling: float = 0.15          # 結合強度 K(PRC の振幅)
    prc_sign: float = +1.0          # +1: 反発(逆相/pi 安定), -1: 引力(同相)

    # マイク=音量による近傍結合。heard = A0 / (1 + (d/d0)^2)
    source_level: float = 1.0       # 発声音量 A0
    ref_distance: float = 1.0       # 減衰の基準距離 d0 [m]
    loudness_gate: float = 0.15     # これ未満の音量イベントは無視(遠い個体を捨てる)

    # 結合相手の選び方: "all"=閾値以上すべてに音量重み付き, "nearest"=最近傍1体のみ
    coupling_mode: str = "all"

    refractory: float = 0.3         # 発声直後にマイク判定を無効化する時間 [s](自己ゲート)
    detection_latency: float = 0.0  # 発声から他個体が反応するまでの遅延 [s](AI 検出遅延)

    seed: int = 0

    # 位置。None なら 1 次元等間隔に自動配置。(n,2) 配列を渡してもよい。
    positions: np.ndarray | None = None

    history_dt: float = 0.02        # 位相履歴を記録する間隔 [s]


@dataclass
class Result:
    config: Config
    positions: np.ndarray                       # (n,2)
    call_times: list = field(default_factory=list)   # 発声イベント (time, frog_id)
    hist_t: np.ndarray = None                   # 位相履歴の時刻 (H,)
    hist_phase: np.ndarray = None               # 位相履歴 (H, n)
    distances: np.ndarray = None                # 個体間距離 (n,n)


def _default_positions(n: int) -> np.ndarray:
    """1 次元に等間隔配置(隣接個体の距離 = ref_distance 相当)."""
    xs = np.arange(n, dtype=float)
    return np.column_stack([xs, np.zeros(n)])


def simulate(cfg: Config) -> Result:
    rng = np.random.default_rng(cfg.seed)

    pos = cfg.positions if cfg.positions is not None else _default_positions(cfg.n)
    pos = np.asarray(pos, dtype=float)

    # 個体間距離と、各個体が各個体の声を聞く音量
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    loud = cfg.source_level / (1.0 + (dist / cfg.ref_distance) ** 2)
    np.fill_diagonal(loud, 0.0)  # 自分の声は結合に使わない(自己ゲート)

    nearest = np.full(cfg.n, -1, dtype=int)
    if cfg.n > 1:
        masked = dist + np.eye(cfg.n) * 1e9
        nearest = masked.argmin(axis=1)

    # 固有周期の個体差
    periods = cfg.period_mean * (1.0 + cfg.period_spread * rng.standard_normal(cfg.n))
    periods = np.clip(periods, 0.3 * cfg.period_mean, 3.0 * cfg.period_mean)

    phase = rng.random(cfg.n)          # 初期位相はランダム
    last_call = np.full(cfg.n, -1e9)   # 最終発声時刻(不応期判定用)

    n_steps = int(round(cfg.duration / cfg.dt))
    hist_every = max(1, int(round(cfg.history_dt / cfg.dt)))

    call_times: list[tuple[float, int]] = []
    hist_t: list[float] = []
    hist_phase: list[np.ndarray] = []

    # AI 検出遅延を表す発声イベントの遅延キュー: (適用時刻, 発声した個体)
    pending: list[tuple[float, int]] = []

    for step in range(n_steps):
        t = step * cfg.dt

        # 1) 自然進行
        phase += cfg.dt / periods

        # 2) 発声(位相が 1 を超えた個体)
        fired = np.where(phase >= 1.0)[0]
        for f in fired:
            phase[f] -= 1.0
            last_call[f] = t
            call_times.append((t, int(f)))
            pending.append((t + cfg.detection_latency, int(f)))

        # 3) 遅延キューから、今 tick で「聞こえる」発声を取り出して結合を適用
        ready = [ev for ev in pending if ev[0] <= t + 1e-9]
        pending = [ev for ev in pending if ev[0] > t + 1e-9]
        for _, f in ready:
            _apply_coupling(f, phase, loud, last_call, nearest, t, cfg)

        # 4) 位相履歴の記録
        if step % hist_every == 0:
            hist_t.append(t)
            hist_phase.append(phase.copy())

    return Result(
        config=cfg,
        positions=pos,
        call_times=call_times,
        hist_t=np.asarray(hist_t),
        hist_phase=np.asarray(hist_phase),
        distances=dist,
    )


def _apply_coupling(f, phase, loud, last_call, nearest, t, cfg: Config) -> None:
    """発声個体 f の声を聞いた各個体に位相応答(PRC)を適用する."""
    if cfg.coupling_mode == "nearest":
        # 「f を最近傍とする個体」だけが反応(近くの 1 体との結合を明示的に再現)
        listeners = np.where(nearest == f)[0]
    else:
        listeners = np.arange(len(phase))

    for l in listeners:
        if l == f:
            continue
        if t - last_call[l] < cfg.refractory:   # 不応期(自己ゲート)
            continue
        heard = loud[l, f]
        if heard < cfg.loudness_gate:            # 遠い個体は無視
            continue
        weight = 1.0 if cfg.coupling_mode == "nearest" else heard / cfg.source_level
        # 反発性 PRC: Delta phi = sign * K * weight * sin(2 pi phi)
        # sign=+1 のとき位相差 0.5(=pi, 逆相)が安定点になる
        dphi = cfg.prc_sign * cfg.coupling * weight * np.sin(2.0 * np.pi * phase[l])
        phase[l] = (phase[l] + dphi) % 1.0


def phase_difference_series(res: Result, i: int, j: int) -> np.ndarray:
    """個体 i, j の位相差(0..1 に折り返し)の時系列. 0.5 が逆相(pi)."""
    d = (res.hist_phase[:, i] - res.hist_phase[:, j]) % 1.0
    return d


def nearest_neighbor_phase_diffs(res: Result, tail_fraction: float = 0.5) -> np.ndarray:
    """後半区間での、最近傍ペアの位相差(0..0.5 に折り返し)の分布用データ.

    0.5 付近にピークが立てば「最近傍どうしが交互(逆相)に鳴いている」ことを示す。
    """
    n = res.config.n
    if n < 2:
        return np.array([])
    dist = res.distances + np.eye(n) * 1e9
    nearest = dist.argmin(axis=1)
    start = int(len(res.hist_t) * (1.0 - tail_fraction))
    diffs = []
    seen = set()
    for i in range(n):
        j = int(nearest[i])
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        d = (res.hist_phase[start:, i] - res.hist_phase[start:, j]) % 1.0
        d = np.minimum(d, 1.0 - d)   # 0..0.5 に折り返し(位相差の絶対量)
        diffs.append(d)
    return np.concatenate(diffs) if diffs else np.array([])
