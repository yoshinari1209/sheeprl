# SheepRL (Python 3.12 / NumPy 2 fork)

This repository is a fork of [Eclectic-Sheep/sheeprl](https://github.com/Eclectic-Sheep/sheeprl) adjusted for Python 3.12 and NumPy 2.x. It retains the original Apache License 2.0; see `LICENSE` and `NOTICE` for copyright and modification details.  
Python 3.12 / NumPy 2.x 対応のために調整した sheeprl フォークです。ライセンスは Apache License 2.0 を継承しています。

## Overview / 概要

- Core algorithms and training pipeline from the original SheepRL. / 元の SheepRL のアルゴリズムと学習パイプラインを継承。
- Updated dependencies for Python 3.12 and NumPy 2.x. / Python 3.12 と NumPy 2.x に合わせて依存関係を更新。
- Limited extras: MineRL / MineDojo remain Python <3.12 only. / MineRL・MineDojo は Python 3.12 では非対応。

## Requirements / 前提

- Python 3.8–3.12 (tested mainly on 3.12). / 主に Python 3.12 で検証。
- CUDA/cuDNN optional for GPU training (match your PyTorch build). / GPU 利用時は PyTorch ビルドに合わせて CUDA/cuDNN を用意。
- Recommended: virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`). / 仮想環境の利用を推奨。

## Installation / インストール

```bash
pip install -e .[dev,test]           # full development install / 開発用一式
# or minimal runtime
pip install -e .                     # 最小構成
```

Optional extras (examples) / 任意のエクストラ:

- `pip install -e .[atari,box2d,mujoco,diambra,dmc,supermario,mlflow]`
- MineRL/MineDojo extras are not provided for Python 3.12 due to upstream gym limitations. / MineRL・MineDojo は上流の gym 制約により 3.12 では未提供。

## Quick start / すぐ試す

Train a PPO agent on CartPole:

```bash
python sheeprl.py exp=ppo env=gym env.id=CartPole-v1
```

Evaluate a saved checkpoint:

```bash
python sheeprl_eval.py checkpoint.resume_from=/path/to/ckpt.ckpt
```

## Notes on this fork / フォークの変更点

- Dependency bumps for Python 3.12 / NumPy 2.x (e.g., gymnasium >=1.2, torch >=2.3, lightning >=2.3). / gymnasium>=1.2, torch>=2.3, lightning>=2.3 などへ更新。
- Compatibility tweaks (e.g., RecordVideo wrapper) and minor test fixes for updated gym environments. / RecordVideo 周辺の互換対応や gym 更新に伴う軽微なテスト修正。
- Optional Weights & Biases logging supported; supply your own API key. / Weights & Biases ロギング対応（API キーは各自用意）。
- A summary of modifications is in `NOTICE`. / 変更の概要は `NOTICE` に記載。

## License / ライセンス

- Licensed under the Apache License 2.0 (see `LICENSE`). / Apache License 2.0 準拠。
- Original work © 2023 Eclectic Sheep Team; this fork’s modifications © 2024 current maintainers. / 原著作権: Eclectic Sheep Team (2023)、本フォークの変更: 現メンテナ (2024)。
- Third-party components retain their respective licenses; see `NOTICE` for details. / サードパーティのライセンスは各コンポーネントに従い、詳細は `NOTICE` を参照。
