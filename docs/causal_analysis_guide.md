# OCEAN による因果関係抽出ガイド

## 概要

OCEANモデルを使用してマイクロサービス間の因果関係を抽出し、根本原因分析を行う方法を詳しく説明します。

## 因果関係抽出の仕組み

### 1. Multi-factor Attention による因果関係推定

OCEANモデルは以下の方法で因果関係を抽出します：

```python
# 注意機構による重要度計算
attention_weights = multi_factor_attention(temporal_features, spatial_features, log_features)

# 因果関係スコア = 注意重み × 時系列相関 × グラフ構造
causality_score = attention_weights * temporal_correlation * graph_adjacency
```

### 2. 対比学習による特徴表現

```python
# InfoNCE損失による表現学習
contrastive_loss = info_nce_loss(positive_pairs, negative_pairs)

# 学習された表現から因果関係を推定
causal_embeddings = contrastive_learning_module(multimodal_features)
```

## 実践的な使用方法

### 基本的な因果関係抽出

```python
from causal_extraction_example import CausalRelationshipExtractor
from ocean.models.ocean_model import OCEANModel
from ocean.configs.default_config import default_config

# モデルの初期化
config = default_config()
model = OCEANModel(config)
model.eval()

# 因果関係抽出器の作成
service_names = ['web', 'api', 'db', 'cache', 'queue']
extractor = CausalRelationshipExtractor(model, service_names)

# 因果関係の抽出
results = extractor.extract_causal_relationships(
    metrics=time_series_data,      # (batch, seq_len, features)
    service_graph=service_graph,   # ServiceGraph object
    logs=log_embeddings,           # (batch, seq_len, log_dim)
    threshold=0.5                  # 根本原因の確信度閾値
)
```

### 結果の解釈

抽出された因果関係は以下の形式で返されます：

```python
{
    'root_cause_probabilities': np.ndarray,  # 各サービスの根本原因確率
    'causal_relationships': {
        'cross_modal_attention': {},      # クロスモーダル注意パターン
        'temporal_dependencies': {},      # 時系列依存関係
        'spatial_relationships': {}       # 空間的関係
    },
    'service_influences': {},             # サービス間影響度
    'temporal_causality': {},             # 時系列因果関係
    'summary': {}                         # 分析要約
}
```

## 因果関係の種類

### 1. 直接的因果関係

サービスAの異常が直接的にサービスBに影響を与える関係：

```
Database Error → User Service Timeout → API Gateway Slow Response
```

### 2. 間接的因果関係

複数のサービスを経由して伝播する影響：

```
Cache Miss → Database Overload → Multiple Service Degradation
```

### 3. 共通原因

同一の根本原因が複数のサービスに同時に影響：

```
Network Issue → Web Frontend + API Gateway + Database (All affected)
```

## 実際のシナリオ例

### シナリオ1: データベース過負荷

```python
# データベースで異常が発生するシナリオ
def database_overload_scenario():
    # 時系列データでデータベースの異常を模擬
    metrics[anomaly_start:, db_idx] += 3.0  # CPU使用率上昇
    metrics[anomaly_start+1:, user_service_idx] += 2.0  # 依存サービスへの影響
    metrics[anomaly_start+2:, api_idx] += 1.0  # 上流サービスへの波及
    
    # 因果関係抽出
    results = extractor.extract_causal_relationships(metrics, service_graph)
    
    # 期待される結果:
    # - データベースが最高確率の根本原因
    # - user-service → api の順で影響が伝播
    # - 時系列的に遅延を持った因果関係が検出
```

### シナリオ2: ネットワーク分断

```python
def network_partition_scenario():
    # 複数サービスが同時に影響を受けるシナリオ
    affected_services = ['web', 'api', 'cache']
    
    for service in affected_services:
        service_idx = service_names.index(service)
        metrics[anomaly_start:, service_idx] += 2.5
    
    # 因果関係抽出
    results = extractor.extract_causal_relationships(metrics, service_graph)
    
    # 期待される結果:
    # - 複数の根本原因候補
    # - 共通の時刻での異常開始
    # - ネットワーク関連の共通因子が検出
```

## 高度な分析手法

### 1. 時系列ラグ分析

```python
def analyze_causal_lags(temporal_features, window_size=5):
    """因果関係の時間的遅延を分析"""
    lags = {}
    
    for i, source_service in enumerate(service_names):
        for j, target_service in enumerate(service_names):
            if i != j:
                # 相互相関による遅延検出
                correlation = cross_correlation(
                    temporal_features[:, i], 
                    temporal_features[:, j],
                    max_lag=window_size
                )
                lags[f"{source_service}->{target_service}"] = {
                    'optimal_lag': np.argmax(correlation),
                    'correlation_strength': np.max(correlation)
                }
    
    return lags
```

### 2. グラフベース因果推定

```python
def graph_based_causality(service_graph, causal_scores):
    """グラフ構造を考慮した因果関係推定"""
    
    # PageRankアルゴリズムによる影響度計算
    adjacency = service_graph.adjacency_matrix
    influence_scores = pagerank(adjacency, causal_scores)
    
    # 最短パスによる因果チェーン検出
    causal_chains = []
    for root_cause in high_probability_causes:
        paths = shortest_paths(adjacency, root_cause)
        causal_chains.extend(paths)
    
    return influence_scores, causal_chains
```

### 3. マルチモーダル一貫性チェック

```python
def multimodal_consistency_check(metrics_causality, log_causality, graph_causality):
    """複数のモダリティ間での因果関係の一貫性をチェック"""
    
    consistency_scores = {}
    
    for service in service_names:
        metric_score = metrics_causality.get(service, 0)
        log_score = log_causality.get(service, 0)
        graph_score = graph_causality.get(service, 0)
        
        # 一貫性スコア = モダリティ間の合意度
        consistency = 1.0 - np.std([metric_score, log_score, graph_score])
        consistency_scores[service] = consistency
    
    return consistency_scores
```

## 実行例

### サンプルの実行

```bash
# 因果関係抽出デモの実行
python causal_extraction_example.py
```

期待される出力：
```
🔍 OCEAN因果関係抽出のデモンストレーション
==================================================
📊 サンプルシナリオを作成中...
🤖 OCEANモデルを初期化中...
🔬 因果関係抽出器を初期化中...
🎯 因果関係を抽出中...

📋 分析結果:
------------------------------
最有力根本原因: database
確信度: 0.847

根本原因候補数: 3

上位根本原因候補:
  1. database: 0.847 (high)
  2. user-service: 0.623 (medium)
  3. api-gateway: 0.456 (medium)

🌐 サービス影響度分析:
  database:
    外向き影響: 2.341
    内向き影響: 0.123
    影響比率: 19.024
  user-service:
    外向き影響: 1.245
    内向き影響: 1.876
    影響比率: 0.664

⏰ 時系列因果関係分析:
  変化点:
    database (時刻10): 2.156
    user-service (時刻12): 1.534
    api-gateway (時刻14): 0.923

🎉 因果関係抽出デモンストレーション完了!
```

## 結果の活用方法

### 1. アラート優先度付け

```python
def prioritize_alerts(causal_results, current_alerts):
    """因果関係に基づくアラート優先度付け"""
    
    prioritized_alerts = []
    root_causes = causal_results['summary']['likely_root_causes']
    
    for alert in current_alerts:
        service = alert['service']
        priority_boost = 0
        
        # 根本原因候補の場合は優先度を上げる
        for cause in root_causes:
            if cause['service'] == service:
                priority_boost = cause['probability'] * 10
                break
        
        alert['priority'] += priority_boost
        prioritized_alerts.append(alert)
    
    return sorted(prioritized_alerts, key=lambda x: x['priority'], reverse=True)
```

### 2. 自動修復アクション

```python
def suggest_remediation_actions(causal_results):
    """因果関係に基づく修復アクション提案"""
    
    actions = []
    top_cause = causal_results['summary']['top_root_cause']
    
    if top_cause:
        service = top_cause['service']
        confidence = top_cause['probability']
        
        if service == 'database' and confidence > 0.8:
            actions.append({
                'action': 'scale_database_replicas',
                'priority': 'high',
                'description': 'データベースレプリカのスケーリング'
            })
            actions.append({
                'action': 'enable_query_caching',
                'priority': 'medium',
                'description': 'クエリキャッシュの有効化'
            })
        
        elif service == 'api-gateway' and confidence > 0.7:
            actions.append({
                'action': 'increase_rate_limits',
                'priority': 'high',
                'description': 'レート制限の調整'
            })
    
    return actions
```

### 3. ダッシュボード統合

```python
def create_causality_dashboard(causal_results):
    """因果関係ダッシュボードの作成"""
    
    dashboard_data = {
        'timestamp': datetime.now().isoformat(),
        'root_cause_summary': causal_results['summary'],
        'service_health_scores': {},
        'causal_graph': {},
        'recommendations': suggest_remediation_actions(causal_results)
    }
    
    # サービス健全性スコアの計算
    influences = causal_results['service_influences']
    for service, influence in influences.items():
        health_score = 1.0 - (influence['incoming_influence'] / 10.0)
        dashboard_data['service_health_scores'][service] = max(0, min(1, health_score))
    
    return dashboard_data
```

## まとめ

OCEANによる因果関係抽出は以下の手順で実行されます：

1. **データ準備**: 時系列メトリクス、ログ、サービスグラフの用意
2. **モデル実行**: OCEANモデルによる推論
3. **因果関係抽出**: 注意重み、グラフ構造、時系列パターンの分析
4. **結果解釈**: 根本原因の特定と影響度の評価
5. **アクション提案**: 修復アクションの優先度付け

この手法により、複雑なマイクロサービス環境における異常の根本原因を効果的に特定し、迅速な問題解決を支援できます。