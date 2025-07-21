# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

OCEANは、マイクロサービス環境における根本原因分析のためのマルチモーダル深層学習フレームワークです。時系列メトリクス、ログ、サービス依存グラフを統合して、異常の根本原因を特定します。

## 主要コマンド

### 基本テスト・実行
```bash
# 最も簡単な動作確認
python simple_test.py

# Dockerでの動作確認
make docker-test
docker-compose run --rm ocean-dev python simple_test.py

# 因果関係抽出のデモ（核心機能）
python causal_extraction_example.py
docker-compose run --rm ocean-dev python causal_extraction_example.py
```

### テスト実行
```bash
# 全テストの実行
make test
python -m pytest tests/ -v

# ユニットテストのみ
make test-unit
python -m pytest tests/unit/ -v

# 統合テストのみ
make test-integration
python -m pytest tests/integration/ -v

# カバレッジ付きテスト
make test-coverage
python -m pytest tests/ --cov=ocean --cov-report=html --cov-report=term
```

### 開発コマンド
```bash
# コードフォーマット
make format
black ocean tests
isort ocean tests

# リント実行
make lint
flake8 ocean tests
mypy ocean

# 開発環境のセットアップ
make dev-setup
pip install -r requirements.txt
pip install -e .
```

### Docker環境
```bash
# Dockerビルド
make docker-build
docker-compose build

# 開発用コンテナ起動
make docker-dev
docker-compose run --rm ocean-dev

# テスト用コンテナ
make docker-test
docker-compose run --rm ocean-test

# 統合テスト
make docker-integration
docker-compose run --rm ocean-integration-test
```

## アーキテクチャ概要

### 主要コンポーネント

1. **OCEAN Model (`ocean/models/ocean_model.py`)**
   - 統合モデルクラス：DilatedCNN、GNN、MultiFactorAttention、GraphFusionを統合
   - エントリーポイント：`OCEANModel`クラス

2. **モデルコンポーネント (`ocean/models/components/`)**
   - `DilatedCNN`: 時系列メトリクス処理（1次元拡張畳み込み）
   - `GraphNeuralNetwork`: サービス依存グラフ処理（GAT使用）
   - `MultiFactorAttention`: マルチモーダル注意機構
   - `GraphFusionModule`: 対比学習（InfoNCE）による特徴融合

3. **データ処理 (`ocean/data/`)**
   - `data_types.py`: 型定義（ServiceGraph、DatasetSample等）
   - `processing/`: データ前処理パイプライン
   - `loaders/`: データローダー

4. **設定管理 (`ocean/configs/`)**
   - `default_config.py`: 階層的設定システム（ModelConfig、DataConfig等）

### データフロー

```
Raw Data → DataProcessor → DatasetSample → OCEANModel → Root Cause Prediction
    ↓           ↓              ↓             ↓              ↓
メトリクス → 正規化 → バッチ化 → DCNN+GNN → 注意機構 → 因果推定
ログ → 埋め込み → テンソル化 → 融合モジュール → 分類ヘッド
グラフ → 隣接行列 → ServiceGraph → 対比学習 → 確率出力
```

### 重要な設計パターン

- **モジュラー設計**: 各コンポーネントが独立して動作・テスト可能
- **型安全性**: dataclassとtype hintsを活用した型定義
- **設定管理**: dataclassベースの階層的設定システム
- **PyTorch Geometric**: グラフ処理にPyG形式（edge_index）を使用

## 開発時の注意点

### モデル使用例
```python
from ocean.configs.default_config import default_config
from ocean.models.ocean_model import OCEANModel
from ocean.data.data_types import ServiceGraph

# 基本的な使用パターン
config = default_config()
model = OCEANModel(config)

# 推論時は必ずモデルを評価モードに
model.eval()
with torch.no_grad():
    outputs = model(metrics, service_graph, logs)
    root_cause_probs = outputs['root_cause_probs']
```

### 主要な依存関係
- PyTorch 2.0+ (基盤フレームワーク)
- torch-geometric (グラフ処理)
- transformers (ログ埋め込み)
- pandas (データ処理)
- wandb (実験管理)

### デバッグのコツ
- 環境変数 `OCEAN_LOG_LEVEL=DEBUG` でデバッグログ有効化
- `return_intermediate=True` で中間表現を確認
- tensor.shape で次元チェック

### よくある問題
- **次元エラー**: データ前処理での形状不一致 → DataProcessorの確認
- **メモリエラー**: バッチサイズ過大 → `config.data.batch_size`を削減
- **学習停滞**: 学習率不適切 → `config.training.learning_rate`を調整

### テスト戦略
- ユニットテスト: 各コンポーネントの独立テスト
- 統合テスト: エンドツーエンドのパイプラインテスト
- simple_test.py: 最低限の動作確認
- causal_extraction_example.py: 核心機能のデモ

### エラー対応時の確認順序
1. simple_test.py で基本動作確認
2. requirements.txt の依存関係確認
3. ログレベルをDEBUGに設定して詳細確認
4. テンソル次元の妥当性確認