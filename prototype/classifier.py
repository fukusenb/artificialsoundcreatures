"""差し替え可能な frog / non_frog 判定器.

- EnergyClassifier: 学習不要・その場で動く簡易版。CNN が無くても frog_node を
  今すぐ 2 台で試すためのプレースホルダ (帯域エネルギ + 簡易な周期性)。
  ※ これは「AI 音声識別」の代替ではなく、配線・音響結合・振動子を先に検証するための踏み台。
- KerasClassifier: 学習済み Keras (.h5) もしくは TFLite (.tflite) を読み込む本命。
  features.logmel と同じ特徴を入力する。

いずれも predict(signal) -> (prob_frog: float, loudness: float) を返す共通 IF。
loudness は近傍ゲート (近い個体=大きい) に使う RMS ベースの音量。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from features import FeatureParams, logmel, mel_filterbank


def rms_loudness(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=np.float32).flatten()
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2)))


class Classifier(Protocol):
    def predict(self, signal: np.ndarray) -> tuple[float, float]:
        ...


class EnergyClassifier:
    """帯域エネルギ比 + ピーク周期性による簡易判定 (踏み台).

    アマガエルの主要帯域 (~2-4kHz) のエネルギ比が高く、かつ振幅包絡に
    数十 ms 周期のパルス性があるときに frog 寄りの確率を返す。
    """

    def __init__(self, params: FeatureParams | None = None,
                 band=(2000.0, 4000.0), threshold=0.35):
        self.p = params or FeatureParams()
        self.band = band
        self.threshold = threshold

    def predict(self, signal: np.ndarray) -> tuple[float, float]:
        x = np.asarray(signal, dtype=np.float32).flatten()
        loud = rms_loudness(x)
        if x.size < self.p.win_length:
            return 0.0, loud
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.p.sample_rate)
        total = float(np.sum(spec) + 1e-9)
        band_mask = (freqs >= self.band[0]) & (freqs <= self.band[1])
        band_ratio = float(np.sum(spec[band_mask]) / total)
        # 帯域比を確率っぽく写像 (踏み台なので粗くてよい)
        prob = 1.0 / (1.0 + np.exp(-12.0 * (band_ratio - self.threshold)))
        return float(prob), loud


class KerasClassifier:
    """学習済みモデル (.h5 / .tflite) を読み込む本命判定器."""

    def __init__(self, model_path: str, params: FeatureParams | None = None):
        self.p = params or FeatureParams()
        self.fb = mel_filterbank(self.p)
        self.model_path = model_path
        self._mode = None
        self._load(model_path)

    def _load(self, path: str):
        if path.endswith(".tflite"):
            try:
                import tflite_runtime.interpreter as tfl  # type: ignore
            except ImportError:
                import tensorflow as tf  # type: ignore
                tfl = tf.lite
            self._interp = tfl.Interpreter(model_path=path)
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]
            self._out = self._interp.get_output_details()[0]
            self._mode = "tflite"
        else:
            import tensorflow as tf  # type: ignore
            self._model = tf.keras.models.load_model(path)
            self._mode = "keras"

    def _features(self, signal: np.ndarray) -> np.ndarray:
        feat = logmel(signal, self.p, self.fb)              # (n_frames, n_mels)
        return feat[np.newaxis, ..., np.newaxis].astype(np.float32)  # (1,T,M,1)

    def predict(self, signal: np.ndarray) -> tuple[float, float]:
        loud = rms_loudness(signal)
        x = self._features(signal)
        if self._mode == "keras":
            y = self._model.predict(x, verbose=0)[0]
        else:
            inp = x
            if self._in["dtype"] == np.int8:
                scale, zp = self._in["quantization"]
                inp = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
            self._interp.set_tensor(self._in["index"], inp)
            self._interp.invoke()
            y = self._interp.get_tensor(self._out["index"])[0]
            if self._out["dtype"] == np.int8:
                scale, zp = self._out["quantization"]
                y = (y.astype(np.float32) - zp) * scale
        y = np.asarray(y, dtype=np.float32).flatten()
        # クラス順は [non_frog, frog] を想定 (train_cnn.py と合わせる)
        prob_frog = float(y[-1]) if y.size >= 2 else float(y[0])
        return prob_frog, loud
