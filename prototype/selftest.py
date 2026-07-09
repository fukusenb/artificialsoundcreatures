"""音声デバイス無しで動く自己テスト (ロジック検証用).

- features.logmel の出力形状が FeatureParams と整合するか
- EnergyClassifier がカエル帯域(3kHz)とホワイトノイズを区別できるか
- PhaseOscillator が発声・PRC で位相を動かすか
"""

import numpy as np

from classifier import EnergyClassifier
from features import FeatureParams, logmel
from frog_node import PhaseOscillator, _synth_call


def test_features():
    p = FeatureParams()
    sig = np.random.randn(p.sample_rate).astype(np.float32) * 0.1
    feat = logmel(sig, p)
    assert feat.shape == (p.n_frames, p.n_mels), (feat.shape, (p.n_frames, p.n_mels))
    assert np.isfinite(feat).all()
    print(f"[features] OK shape={feat.shape}")


def test_classifier():
    p = FeatureParams()
    clf = EnergyClassifier(p)
    frog = _synth_call(p.sample_rate, seconds=1.0)  # 3kHz パルス列
    noise = np.random.randn(p.sample_rate).astype(np.float32) * 0.3
    pf, lf = clf.predict(frog)
    pn, ln = clf.predict(noise)
    print(f"[classifier] frog-like prob={pf:.2f} loud={lf:.3f} | noise prob={pn:.2f} loud={ln:.3f}")
    assert pf > pn, "frog 帯域の方が高い確率になるべき"


def test_oscillator():
    osc = PhaseOscillator(period=2.0, coupling=0.15, refractory=0.0, loudness_gate=0.0)
    osc.phase = 0.0
    fired = [osc.advance(0.01, t * 0.01) for t in range(400)]  # 4 s
    assert any(fired), "2s 周期なら 4s で発声するはず"
    osc.phase = 0.25
    before = osc.phase
    osc.hear_call(loudness=1.0, now=100.0)
    print(f"[oscillator] fires={sum(fired)} phase {before:.3f} -> {osc.phase:.3f} on hear")
    assert osc.phase != before, "近傍イベントで位相が動くべき"


if __name__ == "__main__":
    test_features()
    test_classifier()
    test_oscillator()
    print("ALL OK")
