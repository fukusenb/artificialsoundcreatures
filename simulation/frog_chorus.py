"""Artificial Sound Creatures - カエル合唱の自己組織化シミュレーター (2 層モデル).

設計は Felix Hess の《Electronic Sound Creatures》と、合原一究のニホンアマガエル
逆相同期モデルを組み合わせたもの。実機ハードウェア(マイク 1 個で混合音を聞く)の
制約に合わせて、次の 2 層で構成する。

遅い層(Felix Hess): 「そもそも鳴くかどうか」の内部状態
  - 興奮 A: カエルの音を聞くと増加、時間減衰。鳴きが続くと高く維持される。
  - 抑制 I: カエル以外の音(環境音・人)で増加、時間減衰。
  - 駆動 D = baseline + A - I。D > 閾値 のとき CALLING、下回ると SILENT。
  => 人が近づくと静まり、去るとバラバラに鳴き戻る、を生む。

速い層(合原/逆相): 「鳴くとして、いつ鳴くか」
  - SILENT の間だけ、他個体の鳴きとそのタイミングを聞く。
  - SILENT -> CALLING へ移る瞬間に、最も大きく聞こえた鳴きを基準に位相を
    「逆相(+T/2)」で一度だけセットする。
  - 鳴き始めたら他個体のタイミングは聞かない(自分の固有周期で自走)。
    以降は音がカエルか環境音かで内部状態を更新するだけ。

状態遷移(有限状態):
  SILENT  --(D>thr; 最寄りの鳴きへ +T/2 一発セット or 種火なら固有位相)-->  CALLING
  CALLING --(D<thr)-->  SILENT
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
    loudness_gate: float = 0.12     # これ未満は「聞こえない」

    # --- 遅い層(内部状態: 興奮/抑制) ---
    baseline: float = 0.15          # 常時のベース駆動
    threshold: float = 1.0          # D>threshold で CALLING
    spont_rate: float = 0.20        # 沈黙中に自発発声へ向かって溜まる速度 [1/s]
    spont_max: float = 1.2          # 自発蓄積の上限
    fatigue: float = 0.06           # 発声中に自発蓄積が減る速度(自然な小休止) [1/s]
    exc_gain: float = 0.9           # カエル 1 声で興奮に加わる量
    exc_tau: float = 2.5            # 興奮の減衰時定数 [s]
    inh_gain: float = 2.5           # 環境音の単位音量あたり抑制の増加率 [1/s]
    inh_tau: float = 3.0            # 抑制の減衰時定数 [s]
    thr_spread: float = 0.15        # 閾値の個体差(相対) -> 点火・復帰がばらつく

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
    hist_active: np.ndarray = None    # (H, n) bool: CALLING かどうか
    hist_D: np.ndarray = None         # (H, n) 駆動 D
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
    thresholds = cfg.threshold * (1.0 + cfg.thr_spread * rng.standard_normal(n))

    phase = rng.random(n)
    active = np.zeros(n, dtype=bool)
    spont = rng.random(n) * 0.3       # 自発蓄積(沈黙中に溜まり、種火になる)
    exc = np.zeros(n)                 # 興奮 A(カエル音で増える)
    inh = np.zeros(n)                 # 抑制 I(環境音で増える)
    # SILENT 中に聞いた「最も大きい鳴き」の (時刻, 音量)
    last_heard_time = np.full(n, -1e9)
    last_heard_loud = np.zeros(n)

    n_steps = int(round(cfg.duration / cfg.dt))
    hist_every = max(1, int(round(cfg.history_dt / cfg.dt)))

    call_times: list[tuple[float, int]] = []
    hist_t: list[float] = []
    hist_phase: list = []
    hist_active: list = []
    hist_D: list = []

    pending: list[tuple[float, int]] = []   # (聞こえる時刻, 発声個体) 検出遅延キュー

    def noise_at(x, y, t):
        s = 0.0
        for nz in (cfg.noises or []):
            lvl = nz["level"](t) if callable(nz["level"]) else nz["level"]
            d = np.hypot(x - nz["x"], y - nz["y"])
            s += lvl / (1.0 + (d / nz["radius"]) ** 2)
        return s

    for step in range(n_steps):
        t = step * cfg.dt

        # --- 遅い層: 内部状態の更新 ---
        exc -= exc * (cfg.dt / cfg.exc_tau)
        inh -= inh * (cfg.dt / cfg.inh_tau)
        for i in range(n):
            inh[i] += cfg.inh_gain * noise_at(pos[i, 0], pos[i, 1], t) * cfg.dt
            if active[i]:
                spont[i] = max(0.0, spont[i] - cfg.fatigue * cfg.dt)   # 鳴くと疲れる
            else:
                spont[i] = min(cfg.spont_max, spont[i] + cfg.spont_rate * cfg.dt)  # 溜まる
        D = cfg.baseline + spont + exc - inh

        # --- 状態遷移 ---
        for i in range(n):
            if not active[i] and D[i] > thresholds[i]:
                # SILENT -> CALLING: 位相を一度だけセット
                if last_heard_time[i] > -1e8:
                    # 最寄りの鳴きに対して逆相(+T/2)。遅延補正で 0.5 に寄せる。
                    dt_since = t - last_heard_time[i]
                    comp = (cfg.latency / periods[i]) if cfg.lat_comp else 0.0
                    # 鳴きを聞いた時点の相手位相を 0 とみなし、そこから半周期後に発声
                    phase[i] = (0.5 + comp - (dt_since / periods[i])) % 1.0
                # 種火(誰も聞いていない)なら現在位相のまま自走を開始
                active[i] = True
            elif active[i] and D[i] < thresholds[i]:
                active[i] = False

        # --- 速い層: CALLING の個体だけ位相を進めて発声 ---
        for i in range(n):
            if not active[i]:
                continue
            phase[i] += cfg.dt / periods[i]
            if phase[i] >= 1.0:
                phase[i] -= 1.0
                call_times.append((t, i))
                pending.append((t + cfg.latency, i))
                exc[i] += 0.15 * cfg.exc_gain   # 自分の発声でも少し高揚(合唱の持続)

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
                # どの個体も、聞いたカエル声で興奮(プラス)
                exc[l] += cfg.exc_gain * (heard / cfg.source_level)
                # SILENT の個体だけ、鳴き始め用にタイミングを覚える(最大音量を採用)
                if not active[l] and heard >= last_heard_loud[l] * 0.9:
                    last_heard_time[l] = t
                    last_heard_loud[l] = max(heard, last_heard_loud[l] * 0.5)

        # 聞いた記憶は緩やかに忘れる(古いタイミングに固執しない)
        last_heard_loud *= (1.0 - cfg.dt / 1.5)

        if step % hist_every == 0:
            hist_t.append(t)
            hist_phase.append(phase.copy())
            hist_active.append(active.copy())
            hist_D.append(D.copy())

    return Result(
        config=cfg,
        positions=pos,
        call_times=call_times,
        hist_t=np.asarray(hist_t),
        hist_phase=np.asarray(hist_phase),
        hist_active=np.asarray(hist_active),
        hist_D=np.asarray(hist_D),
        distances=dist,
    )


def active_fraction_series(res: Result) -> np.ndarray:
    """各時刻で鳴いている個体の割合(合唱の盛り上がり)。"""
    if res.hist_active is None or len(res.hist_active) == 0:
        return np.array([])
    return res.hist_active.mean(axis=1)


def nearest_neighbor_phase_diffs(res: Result, tail_fraction: float = 0.5) -> np.ndarray:
    """後半区間で、両方が CALLING の最近傍ペアの位相差(0..0.5 折り返し)。"""
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
