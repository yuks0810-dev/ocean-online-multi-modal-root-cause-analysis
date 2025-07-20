# OCEAN: Online Multi-modal Causal structure lEArNiNG

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

マイクロサービスシステムにおけるリアルタイム根本原因分析のためのマルチモーダル深層学習フレームワーク

## 📖 概要

OCEANは、複雑なマイクロサービス環境における異常の根本原因を特定するための革新的な深層学習フレームワークです。以下の特徴を持ちます：

- **🔄 オンライン学習**: ストリーミングデータに対応したリアルタイム学習
- **🌐 マルチモーダル**: メトリクス、ログ、サービス依存グラフの統合分析
- **🧠 注意機構**: Multi-factor Attentionによる重要な特徴の自動抽出
- **📊 対比学習**: InfoNCE損失による効果的な特徴表現学習
- **⚡ 高性能**: 大規模マイクロサービス環境での実用的な性能

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      OCEAN Framework                        │
├─────────────────────────────────────────────────────────────┤
│  Input Data Sources                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  Metrics    │ │    Logs     │ │ Service     │          │
│  │  (時系列)    │ │  (テキスト)  │ │ Graph       │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Feature Extractors                                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Dilated CNN │ │    BERT     │ │    GNN      │          │
│  │             │ │ Embeddings  │ │   (GAT)     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Multi-factor Attention & Graph Fusion                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        Contrastive Learning (InfoNCE)                  │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Root Cause Prediction                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         Service-level Anomaly Prediction               │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📁 プロジェクト構造

```
ocean-online-multi-modal-root-cause-analysis/
├── 📁 ocean/                          # メインパッケージ
│   ├── 📁 configs/                    # 設定管理
│   │   ├── __init__.py
│   │   └── default_config.py          # デフォルト設定
│   ├── 📁 data/                       # データ管理
│   │   ├── __init__.py
│   │   ├── data_types.py              # データ型定義
│   │   ├── 📁 datasets/               # データセット読み込み
│   │   ├── 📁 loaders/                # データローダ
│   │   └── 📁 processing/             # データ前処理
│   ├── 📁 models/                     # モデル実装
│   │   ├── __init__.py
│   │   ├── ocean_model.py             # メインモデル
│   │   ├── 📁 components/             # モデルコンポーネント
│   │   │   ├── dilated_cnn.py         # 時系列CNN
│   │   │   ├── graph_neural_network.py # グラフNN
│   │   │   ├── multi_factor_attention.py # 注意機構
│   │   │   └── graph_fusion.py        # グラフ融合
│   │   └── 📁 training/               # 訓練フレームワーク
│   │       ├── trainer.py             # バッチ訓練
│   │       ├── online_learner.py      # オンライン学習
│   │       └── streaming_handler.py   # ストリーミング処理
│   ├── 📁 evaluation/                 # 評価フレームワーク
│   │   ├── __init__.py
│   │   ├── metrics.py                 # 性能指標
│   │   ├── evaluator.py               # 評価器
│   │   └── profiler.py                # 性能プロファイラ
│   └── 📁 utils/                      # ユーティリティ
├── 📁 tests/                          # テストスイート
│   ├── conftest.py                    # テスト設定
│   ├── 📁 unit/                       # ユニットテスト
│   └── 📁 integration/                # 統合テスト
├── 📁 docs/                           # ドキュメント
├── 🐳 Dockerfile                      # Docker設定
├── 🐳 docker-compose.yml              # Docker Compose設定
├── 📋 requirements.txt                # Python依存関係
├── ⚙️ setup.py                        # パッケージ設定
├── 📊 TEST_REPORT.md                  # テスト結果報告書
└── 📖 README.md                       # このファイル
```

## 🚀 クイックスタート

### 前提条件

- Docker & Docker Compose
- Git

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd ocean-online-multi-modal-root-cause-analysis
```

### 2. Docker環境でのセットアップ

```bash
# Dockerイメージをビルド
make docker-build
# または
docker-compose build
```

### 3. 基本機能のテスト

```bash
# 基本機能テスト（推奨）
make docker-test
# または
docker-compose run --rm ocean-dev python simple_test.py
```

### 4. 開発環境の起動

```bash
# 開発コンテナに入る
make docker-dev
# または
docker-compose run --rm ocean-dev /bin/bash
```

## 💡 使用方法

### 基本的な使用例

```python
import torch
from ocean.configs.default_config import default_config
from ocean.models.ocean_model import OCEANModel
from ocean.data.data_types import ServiceGraph, DatasetSample

# 1. 設定の読み込み
config = default_config()

# 2. モデルの初期化
model = OCEANModel(config)

# 3. データの準備
# サービスグラフの作成
adjacency_matrix = torch.eye(5)  # 5つのサービス
node_features = torch.randn(5, 12)  # サービス特徴量
service_graph = ServiceGraph(
    adjacency_matrix=adjacency_matrix,
    node_features=node_features,
    service_names=['web', 'api', 'db', 'cache', 'queue']
)

# 4. 推論の実行
metrics = torch.randn(1, 10, 12)  # (batch, seq_len, features)
logs = torch.randn(1, 10, 768)    # (batch, seq_len, log_embedding_dim)

model.eval()
with torch.no_grad():
    outputs = model(metrics, service_graph, logs)
    root_cause_probs = outputs['root_cause_probs']
    print(f"Root cause probabilities: {root_cause_probs}")
```

### オンライン学習の例

```python
from ocean.models.training.online_learner import OnlineLearner
from ocean.models.training.streaming_handler import StreamingHandler

# オンライン学習器の初期化
online_learner = OnlineLearner(model, config)

# ストリーミングハンドラーの初期化
streaming_handler = StreamingHandler(online_learner)

# ストリーミング開始
streaming_handler.start()

# データストリームの作成
stream_id = "production_stream"
streaming_handler.create_stream(stream_id, buffer_size=1000)

# サンプルデータの追加
for sample in data_stream:
    streaming_handler.add_sample_to_stream(stream_id, sample)

# 統計情報の取得
stats = streaming_handler.get_comprehensive_stats()
print(f"Processing stats: {stats}")
```

### 訓練の例

```python
from ocean.models.training.trainer import Trainer
from ocean.data.loaders.multimodal_dataset import MultiModalDataLoader

# データローダーの準備
data_loader = MultiModalDataLoader(config)

# 訓練器の初期化
trainer = Trainer(
    model=model,
    config=config,
    data_loader=data_loader,
    use_wandb=True  # Weights & Biasesログ
)

# 訓練の実行
history = trainer.train(num_epochs=100)

# 訓練履歴の可視化
trainer.plot_training_history("training_plots.png")
```

### 評価の例

```python
from ocean.evaluation.evaluator import Evaluator
from ocean.evaluation.metrics import PerformanceMetrics

# 評価器の初期化
evaluator = Evaluator(model, config, data_loader)

# モデル評価の実行
results = evaluator.evaluate_model(save_results=True)

# アブレーション研究
ablation_configs = [
    {'disable_attention': True},
    {'disable_graph_fusion': True},
    {'disable_temporal_cnn': True}
]
ablation_results = evaluator.evaluate_ablation_study(ablation_configs)

# 包括的なレポート生成
report = evaluator.generate_evaluation_report("evaluation_report.md")
```

## 🔧 設定

### 設定ファイルの構造

```python
@dataclass
class OCEANConfig:
    model: ModelConfig      # モデル構造の設定
    data: DataConfig        # データ処理の設定  
    training: TrainingConfig # 訓練パラメータの設定
    system: SystemConfig    # システム設定
```

### 主要な設定パラメータ

```python
# モデル設定例
config = default_config()
config.model.temporal_dim = 128        # 時系列特徴次元
config.model.spatial_dim = 128         # 空間特徴次元
config.model.attention_dim = 128       # 注意機構次元
config.model.fusion_dim = 256          # 融合次元

# 訓練設定例
config.training.learning_rate = 0.001  # 学習率
config.training.batch_size = 32        # バッチサイズ
config.training.num_epochs = 100       # エポック数
```

## 🧪 テスト

### 利用可能なテストコマンド

```bash
# 基本機能テスト（最も簡単）
make docker-test
docker-compose run --rm ocean-dev python simple_test.py

# ユニットテスト
make test-unit
docker-compose run --rm ocean-test pytest tests/unit/ -v

# 統合テスト  
make test-integration
docker-compose run --rm ocean-integration-test

# 全テストスイート
make test
docker-compose run --rm ocean-test
```

### テスト結果の確認

```bash
# テストカバレッジレポート
make test-coverage

# テスト結果の詳細
cat TEST_REPORT.md
```

## 📊 評価指標

OCEANは以下の評価指標をサポートしています：

### 分類指標
- **Accuracy**: 全体的な分類精度
- **Precision/Recall/F1**: クラス別性能
- **ROC-AUC**: 分類性能の総合評価

### ランキング指標
- **Precision@k**: 上位k件の精度
- **NDCG@k**: 正規化割引累積利得
- **MAP**: 平均適合率

### 統計的検定
- **paired t-test**: 対応のあるt検定
- **Wilcoxon test**: ノンパラメトリック検定
- **Bootstrap test**: ブートストラップ検定

### プロファイリング
- **推論時間**: バッチ別・レイヤー別測定
- **メモリ使用量**: ピーク・平均使用量
- **スループット**: サンプル/秒

## 🔍 理解を深めるために

### 1. 論文の背景理解

OCEAN論文の核心概念：
- **マルチモーダル学習**: 異なる種類のデータ（メトリクス、ログ、グラフ）を統合
- **対比学習**: InfoNCE損失による表現学習の向上
- **オンライン学習**: ストリーミングデータに対応した継続学習
- **因果構造学習**: サービス間の因果関係の推定

### 2. コード構造の理解

#### 重要なクラスと関係性

```
OCEANModel (メインモデル)
├── DilatedCNN (時系列処理)
├── GraphNeuralNetwork (グラフ処理)  
├── MultiFactorAttention (注意機構)
└── GraphFusionModule (特徴融合)
    └── ContrastiveLearningModule (対比学習)
```

#### データフロー

```
Raw Data → DataProcessor → DatasetSample → Model → Predictions
     ↓           ↓              ↓           ↓         ↓
  メトリクス    正規化        バッチ化      推論     根本原因
  ログ        ベクトル化     グラフ化
  トレース    グラフ構築
```

### 3. 実装のポイント

#### 重要な設計決定
- **モジュラー設計**: 各コンポーネントが独立して動作
- **型安全性**: dataclassとtype hintsの活用
- **設定管理**: 階層的な設定システム
- **ログ記録**: 詳細なログによるデバッグ支援

#### パフォーマンス最適化
- **バッチ処理**: 効率的なテンソル操作
- **GPU対応**: CUDA/MPS自動対応
- **メモリ管理**: ストリーミング処理での最適化

### 4. 拡張のガイド

#### 新しいコンポーネントの追加

```python
# 新しい特徴抽出器の例
class CustomFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # コンポーネントの実装
    
    def forward(self, x):
        # フォワードパスの実装
        return processed_features
```

#### カスタム評価指標の追加

```python
# カスタム指標の例
class CustomMetrics(PerformanceMetrics):
    def compute_custom_metric(self):
        # カスタム指標の計算
        return custom_score
```

## 🛠️ 開発・デバッグガイド

### デバッグのコツ

1. **ログレベルの設定**
```bash
export OCEAN_LOG_LEVEL=DEBUG
```

2. **中間出力の確認**
```python
outputs = model(data, return_intermediate=True)
print(outputs['intermediate'].keys())
```

3. **次元チェック**
```python
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
```

### よくある問題と解決方法

| 問題 | 原因 | 解決方法 |
|-----|------|---------|
| 次元エラー | 入力データの形状不一致 | データ前処理の確認 |
| メモリエラー | バッチサイズが大きすぎる | `config.data.batch_size`を削減 |
| 学習が進まない | 学習率が不適切 | `config.training.learning_rate`を調整 |

## 📚 参考資料

### 関連論文
- OCEAN論文（オリジナル研究）
- InfoNCE: Contrastive Learning論文
- Graph Attention Networks論文

### 技術文書
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 コントリビューション

1. Issueの作成またはコメント
2. フォークとブランチ作成
3. 変更の実装とテスト
4. プルリクエストの作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 📞 サポート

- **Issues**: GitHubのIssueページ
- **Documentation**: `docs/`フォルダ
- **Test Report**: `TEST_REPORT.md`

---

**🎯 このREADMEで理解できること:**
- OCEANの概要と特徴
- プロジェクト構造と各ファイルの役割
- 実際の使用方法とコード例
- テストの実行方法
- 開発・拡張のガイドライン
- トラブルシューティング

**📈 次のステップ:**
1. `simple_test.py`を実行して動作確認
2. 基本的な使用例を試す
3. 自分のデータでの実験
4. カスタマイズと拡張