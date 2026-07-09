# Mac 2 台プロトタイプ (CNN + 振動子 + 音響結合)

ESP32 の C++ に降りる前に、**フルパイプライン（マイク → frog/non_frog 判定 →
振動子 → スピーカー再生）を Python の楽な世界で通し切る**ための足場です。
Mac 2 台を向かい合わせ、互いの再生音を拾って **逆相（位相差 π）に落ちる**ことを検証します。

## 重要な前提

- ここで検証するのは **システムの挙動**（判定が振動子を駆動するか、自己発火しないか、
  逆相に落ちるか）。ファームウェアそのものではない。
- Mac 版コードは ESP32 にそのまま載らない。移植して価値があるのは
  **(1) 振動子ロジック** と **(2) 特徴抽出の仕様**（`features.py`）の 2 つ。
- 特徴抽出は学習 (`train_cnn.py`) と推論 (`classifier.py` / `frog_node.py`) で
  **同一の `features.logmel` を共有**している。これが「実機だけ精度が壊れる」事故の予防線。

## ファイル

| ファイル | 役割 |
|---|---|
| `features.py` | ログメル特徴抽出（学習・推論で共有する唯一の実装） |
| `classifier.py` | `EnergyClassifier`（踏み台・学習不要）/ `KerasClassifier`（本命） |
| `frog_node.py` | 1 台の Mac を 1 匹にするリアルタイムノード |
| `train_cnn.py` | 軽量 DS-CNN の学習 → `.h5` と int8 `.tflite` を出力 |
| `selftest.py` | 音声デバイス無しで動くロジック検証 |

## 進め方（3 フェーズ）

### フェーズ 1: CNN を作る
1. データ収集: `data/frog/*.wav` と `data/non_frog/*.wav` を用意。
   frog は **実機スピーカーから再生してマイクで録り直した音**を主軸に、
   non_frog は会場想定の暗騒音・人声・足音・音楽を実録で。各クラス 100 個以上推奨。
2. 学習: `pip install tensorflow soundfile && python train_cnn.py --data ./data`
3. 完了条件: テストセットで実用精度が出る。

### フェーズ 2: 1 台に統合
```bash
pip install sounddevice numpy
python frog_node.py --model frog_cnn.h5 --call frog_call.wav
```
- `--model` を省くと `EnergyClassifier`（踏み台）で今すぐ動く。CNN 完成前の配線確認に。
- `--call` を省くと仮の合成音を鳴らす。
- 完了条件: 自分の再生音で自己発火しない（不応期 `--refractory` と再生後ミュート `--guard` が効く）。

### フェーズ 3: 2 台で結合
- 2 台の Mac で同じコマンドを実行し、向かい合わせる。
- 完了条件: 片方が鳴くともう片方が反応し、交互鳴き（逆相）に落ち着く。
- うまく揃わない/暴走するときの調整: `--coupling`（結合強度）、`--gate`（近傍音量閾値）、
  `--thresh`（判定確率閾値）、`--guard`（自己ミュート）。
  シミュレーター (`../simulation`) で先に効く範囲を掴んでから実音響で追い込むと速い。

## 自己テスト（このリポジトリの CI 環境でも動く）
```bash
python selftest.py
```
