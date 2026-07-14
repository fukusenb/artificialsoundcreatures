"""学習済みモデルでローカル推論を実行する (frog / non_frog).

学習(train_cnn.ipynb / train_cnn.py)と同じ features.logmel を使うので、
実機やノートブックと特徴抽出が一致する。

使い方:
    # 1) WAV 1 ファイルを判定
    python infer.py --model frog_cnn.h5 --wav sample.wav

    # 2) フォルダを一括判定(frog/ と non_frog/ サブフォルダがあれば精度も出す)
    python infer.py --model frog_cnn_int8.tflite --dir ./data

    # 3) マイクでリアルタイム判定 (要 sounddevice)
    python infer.py --model frog_cnn.h5 --mic

--model を省くと踏み台の EnergyClassifier で動く(モデルの配線確認用)。
モデルは Keras (.h5) か TFLite (.tflite)。Edge Impulse は TFLite でエクスポートしたものを渡す。
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import wave

import numpy as np

from classifier import Classifier, EnergyClassifier, KerasClassifier
from features import FeatureParams

CLASSES = ["non_frog", "frog"]


def read_wav(path: str, target_sr: int) -> np.ndarray:
    """WAV を読み、モノラル float(-1..1)・target_sr にそろえて返す。"""
    with wave.open(path, "rb") as w:
        n, sw, ch, sr = w.getnframes(), w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak
    if sr != target_sr and len(x) > 1:          # 16kHz に線形リサンプル
        tgt_len = int(round(len(x) * target_sr / sr))
        x = np.interp(np.linspace(0, len(x) - 1, tgt_len),
                      np.arange(len(x)), x).astype(np.float32)
    return x


def build_classifier(model_path: str | None, p: FeatureParams) -> Classifier:
    if model_path:
        print(f"[infer] model: {model_path}")
        return KerasClassifier(model_path, p)
    print("[infer] --model 未指定 -> EnergyClassifier(踏み台)を使用")
    return EnergyClassifier(p)


def verdict(prob: float, thresh: float) -> str:
    return "FROG " if prob >= thresh else "non  "


def run_wav(clf, p, path, thresh):
    prob, loud = clf.predict(read_wav(path, p.sample_rate))
    print(f"{verdict(prob, thresh)} prob_frog={prob:5.2f}  loud={loud:5.3f}  {path}")


def run_dir(clf, p, d, thresh):
    labeled = all(os.path.isdir(os.path.join(d, c)) for c in CLASSES)
    if labeled:
        cm = np.zeros((2, 2), dtype=int)
        for label, cls in enumerate(CLASSES):
            for path in sorted(glob.glob(os.path.join(d, cls, "*.wav"))):
                prob, _ = clf.predict(read_wav(path, p.sample_rate))
                pred = 1 if prob >= thresh else 0
                cm[label, pred] += 1
        total = cm.sum()
        acc = (cm[0, 0] + cm[1, 1]) / total if total else float("nan")
        print("confusion (rows=true [non_frog, frog], cols=pred):")
        print(cm)
        print(f"accuracy = {acc:.3f}  (n={total}, threshold={thresh})")
    else:
        for path in sorted(glob.glob(os.path.join(d, "*.wav"))):
            run_wav(clf, p, path, thresh)


def run_mic(clf, p, thresh, infer_hz):
    import sounddevice as sd

    win = int(p.sample_rate * p.clip_seconds)
    ring = np.zeros(win, dtype=np.float32)
    block = int(p.sample_rate * 0.05)
    stream = sd.InputStream(samplerate=p.sample_rate, channels=1, blocksize=block)
    stream.start()
    print("[infer] mic live. Ctrl-C で停止。")
    last = 0.0
    try:
        while True:
            data, _ = stream.read(block)
            ring = np.roll(ring, -len(data)); ring[-len(data):] = data[:, 0]
            now = time.time()
            if now - last >= 1.0 / infer_hz:
                last = now
                prob, loud = clf.predict(ring)
                bar = "#" * int(prob * 30)
                print(f"\r{verdict(prob, thresh)} {prob:4.2f} |{bar:<30}| loud={loud:5.3f}", end="")
    except KeyboardInterrupt:
        print("\n[infer] stopped.")
    finally:
        stream.stop(); stream.close()


def main():
    ap = argparse.ArgumentParser(description="local frog/non_frog inference")
    ap.add_argument("--model", default=None, help=".h5 / .tflite (省略で踏み台)")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--clip", type=float, default=1.0, help="判定窓 [s]")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--infer-hz", type=float, default=4.0)
    args = ap.parse_args()

    p = FeatureParams(clip_seconds=args.clip)
    clf = build_classifier(args.model, p)

    if args.wav:
        run_wav(clf, p, args.wav, args.thresh)
    elif args.dir:
        run_dir(clf, p, args.dir, args.thresh)
    elif args.mic:
        run_mic(clf, p, args.thresh, args.infer_hz)
    else:
        ap.error("--wav / --dir / --mic のいずれかを指定してください")


if __name__ == "__main__":
    main()
