"""ログメルスペクトログラム特徴抽出 (学習・推論で共有する唯一の実装).

このプロジェクトで最大の技術リスクは「学習時と推論時で特徴抽出がズレて、
実機だけ精度が壊れる」こと。それを避けるため、特徴抽出はこの 1 ファイルに集約し、
学習 (train_cnn.py) もリアルタイム推論 (frog_node.py) も必ずここを呼ぶ。

ESP32 / TFLite Micro へ移植する際は、この関数と同じパラメータ
(サンプルレート・窓長・ホップ・メル数・対数の取り方) を C++ 側でも再現すること。
依存を軽くするため numpy だけで実装している (librosa 非依存)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureParams:
    sample_rate: int = 16000     # ESP32 の I2S と揃える
    window_ms: float = 25.0      # FFT 窓長
    hop_ms: float = 10.0         # フレーム間隔
    n_fft: int = 512
    n_mels: int = 40             # メルフィルタ数 (KWS の定番)
    fmin: float = 100.0          # カエルの帯域下限に寄せる
    fmax: float = 8000.0         # ナイキスト以下
    clip_seconds: float = 1.0    # 1 判定あたりの音声窓長
    log_offset: float = 1e-6     # log(0) 回避

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.window_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def n_frames(self) -> int:
        n = int(round(self.sample_rate * self.clip_seconds))
        return 1 + max(0, (n - self.win_length) // self.hop_length)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(p: FeatureParams) -> np.ndarray:
    """三角メルフィルタバンク (n_mels, n_fft//2+1) を返す."""
    n_bins = p.n_fft // 2 + 1
    fft_freqs = np.linspace(0, p.sample_rate / 2, n_bins)
    mel_pts = np.linspace(_hz_to_mel(p.fmin), _hz_to_mel(p.fmax), p.n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.searchsorted(fft_freqs, hz_pts)
    fb = np.zeros((p.n_mels, n_bins), dtype=np.float32)
    for m in range(1, p.n_mels + 1):
        lo, ce, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        for k in range(n_bins):
            f = fft_freqs[k]
            if lo <= f <= ce and ce > lo:
                fb[m - 1, k] = (f - lo) / (ce - lo)
            elif ce <= f <= hi and hi > ce:
                fb[m - 1, k] = (hi - f) / (hi - ce)
    return fb


def logmel(signal: np.ndarray, p: FeatureParams, fb: np.ndarray | None = None) -> np.ndarray:
    """1 次元音声 (float, -1..1 目安) -> ログメル (n_frames, n_mels).

    長さが clip_seconds に足りなければゼロ詰め、超えれば先頭を切り出す。
    """
    if fb is None:
        fb = mel_filterbank(p)

    target = int(round(p.sample_rate * p.clip_seconds))
    x = np.asarray(signal, dtype=np.float32).flatten()
    if len(x) < target:
        x = np.pad(x, (0, target - len(x)))
    else:
        x = x[:target]

    window = np.hanning(p.win_length).astype(np.float32)
    frames = []
    for start in range(0, target - p.win_length + 1, p.hop_length):
        frame = x[start:start + p.win_length] * window
        spec = np.fft.rfft(frame, n=p.n_fft)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel = fb @ power
        frames.append(np.log(mel + p.log_offset))
    if not frames:
        return np.zeros((0, p.n_mels), dtype=np.float32)
    return np.stack(frames).astype(np.float32)
