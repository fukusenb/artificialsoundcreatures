"""Artificial Sound Creatures - カエル合唱の自己組織化シミュレーター (2 層モデル).

設計は Felix Hess《Electronic Sound Creatures》の内部状態モデルと、合原一究の
ニホンアマガエル逆相同期モデルを組み合わせたもの。実機(マイク 1 個で混合音を
聞く)の制約に合わせて 2 層で構成する。

遅い層(Felix Hess): 「そもそも鳴くかどうか」= 単一の鳴きたさ E ∈ [0, 1] (=0〜100%)
  - E は自発的にじわじわ上昇し、カエルの声を聞くとジャンプ(興奮)。
  - 環境音・人がいると下がり(抑制)、鳴いている間は疲れて下がる。
  - E が 1.0(100%)に達したら CALLING 開始。鳴いている間に E が off_level を
    下回ったら SILENT に戻る(ヒステリシス)。
  => 人が近づくと静まり、去るとじわじわ鳴き戻る。1 匹の種火から合唱が点火する。

速い層(合原/逆相): 「鳴くとして、いつ鳴くか」
  - SILENT の間だけ、他個体の鳴きとタイミングを聞く。
  - SILENT -> CALLING の瞬間に、最も大きく聞いた鳴きへ +T/2(逆相)で位相を一度セット。
  - 鳴き始めたらタイミングは聞かず自走。以降は音がカエルか環境音かで E を更新するだけ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    n: int = 6
    dt: float = 0.01
    duration: float = 120.0

    # --- 速い層(発声リズム) ---
    period_mean: float = 1.0
    period_spread: float = 0.05
    latency: float = 0.1            # AI 検出遅延 [s]
    lat_comp: bool = True           # 逆相セット時に遅延を補正するか

    # --- 音の伝播(近い個体ほど大きく聞こえる) ---
    source_level: float = 1.0
    ref_distance: float = 1.0
    loudness_gate: float = 0.12

    # --- 遅い層(鳴きたさ E ∈ [0,1] のダイナミクス) ---
    rise_rate: float = 0.35         # 自発上昇の速さ [1/s] (E は 3s ほどで満タン)
    rise_spread: float = 0.15       # 自発上昇の個体差(相対) -> 点火・復帰がばらつく
    exc_gain: float = 0.40          # カエル 1 声を聞いたときの E のジャンプ量
    inh_gain: float = 1.5           # 環境音の単位音量あたり E を下げる速さ [1/s]
    fatigue: float = 0.50           # 鳴いている間 E が下がる速さ [1/s]
    off_level: float = 0.35         # 鳴いている E がこれを下回ると停止 (ON は 1.0)

    seed: int = 0
    positions: np.ndarray | None = None
    noises: list | None = None      # [{'x','y','level','radius'}] 環境音/人
    history_dt: float = 0.02


@dataclass
class Result:
    config: Config
    positions: np.ndarray
    call_times: list = field(default_factory=list)   # (time, frog_id)
    hist_t: np.ndarray = None
    hist_phase: np.ndarray = None
    hist_active: np.ndarray = None    # (H, n) bool
    hist_E: np.ndarray = None         # (H, n) 鳴きたさ E ∈ [0,1]
    distances: np.ndarray = None


def _default_positions(n: int) -> np.ndarray:
    xs = np.arange(n, dtype=float)
    return np.column_stack([xs, np.zeros(n)])


def simulate(cfg: Config) -> Result:
    rng = np.random.default_rng(cfg.seed)

    pos = cfg.positions if cfg.positions is not None else _default_positions(cfg.n)
    pos = np.asarray(pos, dtype=float)
    n = len(pos)

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    loud = cfg.source_level / (1.0 + (dist / cfg.ref_distance) ** 2)
    np.fill_diagonal(loud, 0.0)

    periods = cfg.period_mean * (1.0 + cfg.period_spread * rng.standard_normal(n))
    periods = np.clip(periods, 0.3 * cfg.period_mean, 3.0 * cfg.period_mean)
    rise = cfg.rise_rate * (1.0 + cfg.rise_spread * rng.standard_normal(n))
    rise = np.clip(rise, 0.05 * cfg.rise_rate, 3.0 * cfg.rise_rate)

    phase = rng.random(n)
    active = np.zeros(n, dtype=bool)
    E = rng.random(n) * 0.5           # 鳴きたさ 0..1
    last_heard_time = np.full(n, -1e9)
    last_heard_loud = np.zeros(n)

    n_steps = int(round(cfg.duration / cfg.dt))
    hist_every = max(1, int(round(cfg.history_dt / cfg.dt)))

    call_times: list[tuple[float, int]] = []
    hist_t: list = []
    hist_phase: list = []
    hist_active: list = []
    hist_E: list = []

    pending: list[tuple[float, int]] = []

    def noise_at(x, y, t):
        s = 0.0
        for nz in (cfg.noises or []):
            lvl = nz["level"](t) if callable(nz["level"]) else nz["level"]
            d = np.hypot(x - nz["x"], y - nz["y"])
            s += lvl / (1.0 + (d / nz["radius"]) ** 2)
        return s

    for step in range(n_steps):
        t = step * cfg.dt

        # --- 遅い層: 鳴きたさ E の連続更新 ---
        for i in range(n):
            E[i] += rise[i] * cfg.dt                                   # 自発上昇
            E[i] -= cfg.inh_gain * noise_at(pos[i, 0], pos[i, 1], t) * cfg.dt  # 抑制
            if active[i]:
                E[i] -= cfg.fatigue * cfg.dt                           # 発声疲労
        np.clip(E, 0.0, 1.0, out=E)

        # --- 状態遷移(ヒステリシス: ON=1.0, OFF=off_level) ---
        for i in range(n):
            if not active[i] and E[i] >= 1.0:
                if last_heard_time[i] > -1e8:
                    dt_since = t - last_heard_time[i]
                    comp = (cfg.latency / periods[i]) if cfg.lat_comp else 0.0
                    phase[i] = (0.5 + comp - (dt_since / periods[i])) % 1.0
                active[i] = True
            elif active[i] and E[i] <= cfg.off_level:
                active[i] = False

        # --- 速い層: CALLING の個体だけ発声 ---
        for i in range(n):
            if not active[i]:
                continue
            phase[i] += cfg.dt / periods[i]
            if phase[i] >= 1.0:
                phase[i] -= 1.0
                call_times.append((t, i))
                pending.append((t + cfg.latency, i))

        # --- 聞こえた鳴きの処理(検出遅延キュー) ---
        due = [ev for ev in pending if ev[0] <= t + 1e-9]
        pending = [ev for ev in pending if ev[0] > t + 1e-9]
        for _, src in due:
            for l in range(n):
                if l == src:
                    continue
                heard = loud[l, src]
                if heard < cfg.loudness_gate:
                    continue
                E[l] = min(1.0, E[l] + cfg.exc_gain * (heard / cfg.source_level))  # 興奮
                if not active[l] and heard >= last_heard_loud[l] * 0.9:
                    last_heard_time[l] = t
                    last_heard_loud[l] = max(heard, last_heard_loud[l] * 0.5)

        last_heard_loud *= (1.0 - cfg.dt / 1.5)

        if step % hist_every == 0:
            hist_t.append(t)
            hist_phase.append(phase.copy())
            hist_active.append(active.copy())
            hist_E.append(E.copy())

    return Result(
        config=cfg,
        positions=pos,
        call_times=call_times,
        hist_t=np.asarray(hist_t),
        hist_phase=np.asarray(hist_phase),
        hist_active=np.asarray(hist_active),
        hist_E=np.asarray(hist_E),
        distances=dist,
    )


def active_fraction_series(res: Result) -> np.ndarray:
    if res.hist_active is None or len(res.hist_active) == 0:
        return np.array([])
    return res.hist_active.mean(axis=1)


def nearest_neighbor_phase_diffs(res: Result, tail_fraction: float = 0.5) -> np.ndarray:
    n = res.config.n
    if n < 2:
        return np.array([])
    dmat = res.distances + np.eye(n) * 1e9
    nearest = dmat.argmin(axis=1)
    start = int(len(res.hist_t) * (1.0 - tail_fraction))
    diffs = []
    seen = set()
    for i in range(n):
        j = int(nearest[i])
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        both = res.hist_active[start:, i] & res.hist_active[start:, j]
        if both.sum() == 0:
            continue
        d = (res.hist_phase[start:, i] - res.hist_phase[start:, j]) % 1.0
        d = np.minimum(d, 1.0 - d)
        diffs.append(d[both])
    return np.concatenate(diffs) if diffs else np.array([])
