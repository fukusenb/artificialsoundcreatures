"""frog / non_frog を分類する軽量 DS-CNN の学習スクリプト (Keras).

方針:
- 特徴抽出は features.logmel を使う (推論と完全に同一 = 実機精度崩壊の回避)。
- 軽量 Depthwise-Separable CNN。int8 量子化した TFLite も書き出し、後で
  TFLite Micro / Edge Impulse 相当のエッジ推論へ繋げる土台にする。

データ配置:
    data/
      frog/       *.wav   (アマガエル。実機スピーカーから再生してマイクで録り直した音を主に)
      non_frog/   *.wav   (会場の暗騒音・人の声・足音・音楽など)

使い方:
    pip install tensorflow soundfile
    python train_cnn.py --data ./data --epochs 30
出力:
    frog_cnn.h5           Keras モデル
    frog_cnn_int8.tflite  int8 量子化モデル (エッジ推論用)
"""

from __future__ import annotations

import argparse
import glob
import os
import wave

import numpy as np

from features import FeatureParams, logmel, mel_filterbank

CLASSES = ["non_frog", "frog"]  # index 0,1。classifier.py の解釈と合わせる。


def _read_wav(path: str, sample_rate: int) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n, sw, ch = w.getnframes(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    peak = float(np.max(np.abs(x))) or 1.0
    return x / peak


def load_dataset(data_dir: str, p: FeatureParams, stride: float = 0.5):
    """各 WAV をスライディングウィンドウで 1 秒窓に分割してサンプル化する。

    Args:
        stride: ウィンドウをずらすステップ [秒]。
                0.5 なら 50% オーバーラップ(デフォルト)。
                1.0 なら重複なし。
                clip_seconds 未満のファイルはゼロ詰めで 1 サンプルとして使う。
    """
    fb = mel_filterbank(p)
    clip_len = int(round(p.sample_rate * p.clip_seconds))
    step = max(1, int(round(p.sample_rate * stride)))
    X, y = [], []
    counts = {}

    for label, cls in enumerate(CLASSES):
        paths = sorted(glob.glob(os.path.join(data_dir, cls, "*.wav")))
        n_files, n_samples = 0, 0
        for path in paths:
            sig = _read_wav(path, p.sample_rate)
            n_files += 1
            if len(sig) < clip_len:
                # 短いファイルはゼロ詰めして 1 サンプルとして使う
                X.append(logmel(sig, p, fb))
                y.append(label)
                n_samples += 1
            else:
                # clip_len ずつ step ステップでスライス
                for start in range(0, len(sig) - clip_len + 1, step):
                    X.append(logmel(sig[start:start + clip_len], p, fb))
                    y.append(label)
                    n_samples += 1
        counts[cls] = (n_files, n_samples)

    if not X:
        raise SystemExit(f"no wav found under {data_dir}/{{{','.join(CLASSES)}}}/")

    for cls, (nf, ns) in counts.items():
        print(f"  {cls:12s}: {nf:4d} files -> {ns:6d} samples "
              f"(stride={stride}s, window={p.clip_seconds}s)")

    X = np.stack(X)[..., np.newaxis].astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y


def build_model(input_shape, n_classes=2):
    import tensorflow as tf
    from tensorflow.keras import layers as L

    return tf.keras.Sequential([
        L.Input(shape=input_shape),
        L.Conv2D(16, 3, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.MaxPooling2D(2),
        L.SeparableConv2D(32, 3, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.MaxPooling2D(2),
        L.SeparableConv2D(64, 3, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.GlobalAveragePooling2D(),
        L.Dropout(0.3),
        L.Dense(n_classes, activation="softmax"),
    ])


def export_int8(model, X_ref, out_path):
    import tensorflow as tf

    def rep():
        for i in range(min(200, len(X_ref))):
            yield [X_ref[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    with open(out_path, "wb") as f:
        f.write(conv.convert())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="判定窓長 [秒] (学習・推論で一致させること)")
    ap.add_argument("--stride", type=float, default=0.5,
                    help="スライスのステップ [秒]。1.0=重複なし、0.5=50%%オーバーラップ")
    args = ap.parse_args()

    import tensorflow as tf

    p = FeatureParams(clip_seconds=args.clip)
    print(f"loading dataset (window={args.clip}s, stride={args.stride}s) ...")
    X, y = load_dataset(args.data, p, stride=args.stride)
    print(f"dataset: X={X.shape}  class counts={np.bincount(y)}")

    idx = np.random.default_rng(0).permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    model = build_model(X.shape[1:], len(CLASSES))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xte, yte),
              epochs=args.epochs, batch_size=args.batch)

    model.save("frog_cnn.h5")
    export_int8(model, Xtr, "frog_cnn_int8.tflite")
    print("saved: frog_cnn.h5, frog_cnn_int8.tflite")


if __name__ == "__main__":
    main()
