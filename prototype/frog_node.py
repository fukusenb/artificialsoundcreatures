"""1 台の Mac を「1 匹のカエル」にするリアルタイムノード.

マイク入力 -> (スライディング窓で) frog/non_frog 判定 -> 判定を振動子へ入力
-> 位相が閾値を超えたらスピーカーからカエル音を再生 -> 位相リセット.

2 台の Mac をそれぞれこのノードで動かして向かい合わせると、互いの再生音を
マイクで拾って結合し、位相差 pi (逆相 = 交互鳴き) に落ちることを検証できる。

音響 I/O には sounddevice が必要 (Mac: `pip install sounddevice`, PortAudio 同梱)。
このファイルの音声 I/O 部分はこのクラウド環境では実行できないが、Mac ではそのまま動く。
振動子ロジックは simulation/frog_chorus.py の PRC と同じ規則。
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np

from classifier import Classifier, EnergyClassifier, KerasClassifier
from features import FeatureParams


class PhaseOscillator:
    """パルス結合位相振動子 (simulation/frog_chorus.py と同じ規則).

    phase を固有周期 T で進め、>=1 で発声要求。近傍イベント受信で反発 PRC を適用。
    """

    def __init__(self, period=2.0, coupling=0.15, prc_sign=+1.0,
                 loudness_gate=0.05, refractory=0.3):
        self.phase = float(np.random.random())
        self.period = period
        self.coupling = coupling
        self.prc_sign = prc_sign
        self.loudness_gate = loudness_gate
        self.refractory = refractory
        self.last_call = -1e9

    def advance(self, dt: float, now: float) -> bool:
        """時間 dt 進める。発声すべきなら True。"""
        self.phase += dt / self.period
        if self.phase >= 1.0:
            self.phase -= 1.0
            self.last_call = now
            return True
        return False

    def hear_call(self, loudness: float, now: float) -> None:
        """近傍の発声イベントを受けて位相応答 (自己ゲート・音量ゲート込み)。"""
        if now - self.last_call < self.refractory:
            return
        if loudness < self.loudness_gate:
            return
        weight = min(1.0, loudness)
        dphi = self.prc_sign * self.coupling * weight * np.sin(2.0 * np.pi * self.phase)
        self.phase = (self.phase + dphi) % 1.0


def load_wav(path: str, sample_rate: int) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        sw = w.getsampwidth()
        ch = w.getnchannels()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    peak = float(np.max(np.abs(data))) or 1.0
    return data / peak  # -1..1 に正規化 (再生時にゲインを掛ける)


def build_classifier(model_path: str | None, params: FeatureParams) -> Classifier:
    if model_path:
        print(f"[node] using KerasClassifier: {model_path}")
        return KerasClassifier(model_path, params)
    print("[node] using EnergyClassifier (踏み台: CNN 未指定)")
    return EnergyClassifier(params)


def run(args):
    import sounddevice as sd  # Mac でのみ必要。ここで遅延 import。

    p = FeatureParams(clip_seconds=args.window)
    clf = build_classifier(args.model, p)
    call_wav = load_wav(args.call, p.sample_rate) if args.call else _synth_call(p.sample_rate)

    osc = PhaseOscillator(period=args.period, coupling=args.coupling,
                          prc_sign=+1.0, loudness_gate=args.gate,
                          refractory=args.refractory)

    ring = np.zeros(int(p.sample_rate * args.window), dtype=np.float32)
    block = int(p.sample_rate * 0.05)  # 50 ms ブロック
    muted_until = 0.0                  # 自己再生中はマイク判定を止める

    stream_in = sd.InputStream(samplerate=p.sample_rate, channels=1, blocksize=block)
    stream_in.start()
    print("[node] started. Ctrl-C で停止。")

    last = time.time()
    last_infer = 0.0
    try:
        while True:
            data, _ = stream_in.read(block)
            frame = data[:, 0]
            ring = np.roll(ring, -len(frame))
            ring[-len(frame):] = frame

            now = time.time()
            dt = now - last
            last = now

            # 振動子を進める → 発声
            if osc.advance(dt, now):
                gain = args.volume
                sd.play(np.clip(call_wav * gain, -1, 1), p.sample_rate)
                muted_until = now + len(call_wav) / p.sample_rate + args.guard
                print(f"[{now:8.2f}] CALL   phase reset")

            # マイク判定 (自己再生中は skip)。args.infer_hz の周期で。
            if now > muted_until and (now - last_infer) >= 1.0 / args.infer_hz:
                last_infer = now
                prob, loud = clf.predict(ring)
                if prob >= args.thresh:
                    osc.hear_call(loud, now)
                    print(f"[{now:8.2f}] heard  prob={prob:.2f} loud={loud:.3f} "
                          f"phase={osc.phase:.2f}")
    except KeyboardInterrupt:
        print("\n[node] stopped.")
    finally:
        stream_in.stop()
        stream_in.close()


def _synth_call(sample_rate: int, seconds: float = 0.25) -> np.ndarray:
    """カエル音 wav が無い場合の仮の合成音 (2-4kHz のパルス列)。"""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    tone = np.sin(2 * np.pi * 3000 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 40 * t))
    env = np.exp(-3 * t)
    return (tone * env).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Real-time frog node (1 Mac = 1 frog)")
    ap.add_argument("--model", default=None, help="学習済み .h5 / .tflite (無ければ踏み台判定)")
    ap.add_argument("--call", default=None, help="再生するカエル音 wav (無ければ合成音)")
    ap.add_argument("--period", type=float, default=2.0, help="固有鳴き周期 [s]")
    ap.add_argument("--coupling", type=float, default=0.15, help="結合強度 K")
    ap.add_argument("--gate", type=float, default=0.03, help="近傍音量ゲート")
    ap.add_argument("--thresh", type=float, default=0.6, help="frog 判定確率の閾値")
    ap.add_argument("--refractory", type=float, default=0.3, help="発声後の不応期 [s]")
    ap.add_argument("--guard", type=float, default=0.15, help="再生後の追加ミュート [s]")
    ap.add_argument("--window", type=float, default=1.0, help="判定窓 [s]")
    ap.add_argument("--infer-hz", type=float, default=10.0, help="判定周期 [Hz]")
    ap.add_argument("--volume", type=float, default=0.9, help="再生ゲイン 0..1")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
