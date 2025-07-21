#!/usr/bin/env python3
"""
OCEAN因果関係抽出の実践的な例

このスクリプトでは、OCEANモデルを使用して実際にマイクロサービス間の
因果関係を抽出し、根本原因分析を行う方法を示します。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from ocean.configs.default_config import default_config
from ocean.models.ocean_model import OCEANModel
from ocean.data.data_types import ServiceGraph, DatasetSample
from ocean.evaluation.metrics import PerformanceMetrics


class CausalRelationshipExtractor:
    """
    因果関係抽出のためのユーティリティクラス
    
    OCEANモデルの出力から因果関係を解釈し、
    根本原因分析を実行します。
    """
    
    def __init__(self, model: OCEANModel, service_names: List[str]):
        self.model = model
        self.service_names = service_names
        self.num_services = len(service_names)
        
    def extract_causal_relationships(self, 
                                   metrics: torch.Tensor,
                                   service_graph: ServiceGraph,
                                   logs: Optional[torch.Tensor] = None,
                                   threshold: float = 0.5) -> Dict:
        """
        因果関係を抽出し、根本原因を特定
        
        Args:
            metrics: 時系列メトリクスデータ (batch, seq_len, features)
            service_graph: サービス依存グラフ
            logs: ログデータ (batch, seq_len, log_dim)
            threshold: 根本原因の確信度閾値
            
        Returns:
            因果関係分析結果を含む辞書
        """
        self.model.eval()
        
        with torch.no_grad():
            # 中間表現も取得するためのフォワードパス
            # ログデータがない場合はダミーデータを作成
            if logs is None:
                # 適当なログ特徴量を生成（実際のプロジェクトでは実データを使用）
                batch_size, seq_len = metrics.shape[:2]
                logs = torch.randn(batch_size, seq_len, 768) * 0.1  # BERT次元
            
            outputs = self.model(
                metrics, 
                service_graph, 
                logs, 
                return_intermediate=True
            )
            
            # 根本原因確率
            root_cause_probs = outputs['root_cause_probs'].cpu().numpy()
            
            # 中間表現から因果関係を抽出
            intermediate = outputs['intermediate']
            
            # 注意重みから因果関係を分析
            causal_analysis = self._analyze_attention_weights(
                intermediate, root_cause_probs, threshold
            )
            
            # サービス間の影響度を計算
            service_influences = self._compute_service_influences(
                intermediate, service_graph
            )
            
            # 時系列的な因果関係を抽出
            temporal_causality = self._extract_temporal_causality(
                intermediate, metrics
            )
            
        return {
            'root_cause_probabilities': root_cause_probs,
            'causal_relationships': causal_analysis,
            'service_influences': service_influences,
            'temporal_causality': temporal_causality,
            'summary': self._generate_causality_summary(
                root_cause_probs, causal_analysis, threshold
            )
        }
    
    def _analyze_attention_weights(self, 
                                 intermediate: Dict,
                                 root_cause_probs: np.ndarray,
                                 threshold: float) -> Dict:
        """注意重みから因果関係を分析"""
        
        # Multi-factor Attentionの重みを取得
        attention_details = intermediate.get('fusion_details', {})
        
        causal_relationships = {
            'cross_modal_attention': {},
            'temporal_dependencies': {},
            'spatial_relationships': {}
        }
        
        # クロスモーダル注意重みの分析
        if 'attention_weights' in attention_details:
            attention_weights = attention_details['attention_weights']
            
            # 各サービスに対する注意重みを分析
            for i, service in enumerate(self.service_names):
                if root_cause_probs[0, i] > threshold:
                    # 高い根本原因確率を持つサービスの注意パターンを分析
                    service_attention = attention_weights[0, :, i]  # (seq_len,)
                    
                    causal_relationships['cross_modal_attention'][service] = {
                        'attention_pattern': service_attention.cpu().numpy().tolist(),
                        'peak_attention_time': int(torch.argmax(service_attention)),
                        'average_attention': float(torch.mean(service_attention))
                    }
        
        return causal_relationships
    
    def _compute_service_influences(self, 
                                  intermediate: Dict,
                                  service_graph: ServiceGraph) -> Dict:
        """サービス間の影響度を計算"""
        
        # グラフ特徴量から影響度を計算
        spatial_features = intermediate['spatial_features']  # (batch, num_services, features)
        
        # サービス間の類似度行列を計算
        # spatial_featuresは(batch_size, spatial_dim)形状なので、サービス毎に分割する必要がある
        # 簡略化: 全サービスに同じ特徴量を適用
        spatial_feature = spatial_features[0]  # (spatial_dim,)
        num_services = len(self.service_names)
        
        # 各サービスに同じ特徴量を複製
        service_features = spatial_feature.unsqueeze(0).repeat(num_services, 1)  # (num_services, spatial_dim)
        
        # コサイン類似度を計算
        similarity_matrix = torch.cosine_similarity(
            service_features.unsqueeze(1),  # (num_services, 1, spatial_dim)
            service_features.unsqueeze(0),  # (1, num_services, spatial_dim)
            dim=2  # spatial_dim次元で計算
        )
        
        # 隣接行列との組み合わせで実際の影響度を計算
        adjacency = service_graph.adjacency_matrix
        influence_matrix = similarity_matrix * adjacency
        
        # 各サービスの影響度スコアを計算
        service_influences = {}
        for i, service in enumerate(self.service_names):
            # そのサービスが他のサービスに与える影響
            outgoing_influence = float(torch.sum(influence_matrix[i, :]))
            # そのサービスが他のサービスから受ける影響
            incoming_influence = float(torch.sum(influence_matrix[:, i]))
            
            service_influences[service] = {
                'outgoing_influence': outgoing_influence,
                'incoming_influence': incoming_influence,
                'influence_ratio': outgoing_influence / (incoming_influence + 1e-8),
                'connected_services': [
                    self.service_names[j] for j in range(self.num_services)
                    if adjacency[i, j] > 0 and i != j
                ]
            }
        
        return service_influences
    
    def _extract_temporal_causality(self, 
                                  intermediate: Dict,
                                  metrics: torch.Tensor) -> Dict:
        """時系列的な因果関係を抽出"""
        
        temporal_features = intermediate['temporal_features']  # (batch, seq_len, features)
        seq_len = temporal_features.shape[1]
        
        # 時刻間の因果関係を分析
        temporal_causality = {
            'lag_analysis': {},
            'change_points': [],
            'trend_analysis': {}
        }
        
        # 各サービスの時系列パターンを分析
        # メトリクスをサービス数で分割（12次元を5サービスで分割できないため、適切に処理）
        metrics_per_service = metrics.shape[-1] // len(self.service_names)  # 12 // 5 = 2
        
        for i, service in enumerate(self.service_names):
            # 各サービスに対して2-3次元のメトリクスを割り当て
            start_idx = i * metrics_per_service
            end_idx = min((i + 1) * metrics_per_service, metrics.shape[-1])
            if i == len(self.service_names) - 1:  # 最後のサービスは残りを全て使用
                end_idx = metrics.shape[-1]
                
            service_metrics = metrics[0, :, start_idx:end_idx]
            
            # 変化点の検出（簡単な差分ベース）
            diff = torch.diff(torch.mean(service_metrics, dim=1))
            change_indices = torch.where(torch.abs(diff) > torch.std(diff) * 2)[0]
            
            temporal_causality['change_points'].extend([
                {
                    'service': service,
                    'time_index': int(idx),
                    'magnitude': float(diff[idx])
                }
                for idx in change_indices
            ])
            
            # トレンド分析
            service_avg = torch.mean(service_metrics, dim=1).cpu().numpy()
            x_range = range(len(service_avg))  # service_metricsの実際の長さを使用
            trend = np.polyfit(x_range, service_avg, 1)[0]
            temporal_causality['trend_analysis'][service] = {
                'trend_slope': float(trend),
                'trend_direction': 'increasing' if trend > 0 else 'decreasing'
            }
        
        return temporal_causality
    
    def _generate_causality_summary(self, 
                                  root_cause_probs: np.ndarray,
                                  causal_analysis: Dict,
                                  threshold: float) -> Dict:
        """因果関係分析の要約を生成"""
        
        # 根本原因の候補を特定
        likely_root_causes = []
        for i, service in enumerate(self.service_names):
            prob = root_cause_probs[0, i]
            if prob > threshold:
                likely_root_causes.append({
                    'service': service,
                    'probability': float(prob),
                    'confidence': 'high' if prob > 0.8 else 'medium'
                })
        
        # 確率順にソート
        likely_root_causes.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'likely_root_causes': likely_root_causes,
            'top_root_cause': likely_root_causes[0] if likely_root_causes else None,
            'num_potential_causes': len(likely_root_causes),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def visualize_causal_relationships(self, 
                                     causal_results: Dict,
                                     save_path: Optional[str] = None):
        """因果関係の可視化"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 根本原因確率のバープロット
        probs = causal_results['root_cause_probabilities'][0]
        ax1.bar(self.service_names, probs)
        ax1.set_title('Root Cause Probabilities')
        ax1.set_ylabel('Probability')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. サービス影響度マトリックス
        influences = causal_results['service_influences']
        influence_matrix = np.zeros((self.num_services, self.num_services))
        
        for i, service in enumerate(self.service_names):
            if service in influences:
                influence_matrix[i, i] = influences[service]['outgoing_influence']
        
        im2 = ax2.imshow(influence_matrix, cmap='RdYlBu_r')
        ax2.set_title('Service Influence Matrix')
        ax2.set_xticks(range(self.num_services))
        ax2.set_yticks(range(self.num_services))
        ax2.set_xticklabels(self.service_names, rotation=45)
        ax2.set_yticklabels(self.service_names)
        plt.colorbar(im2, ax=ax2)
        
        # 3. 時系列トレンド
        trends = causal_results['temporal_causality']['trend_analysis']
        trend_slopes = [trends[service]['trend_slope'] for service in self.service_names]
        
        colors = ['red' if slope > 0 else 'blue' for slope in trend_slopes]
        ax3.bar(self.service_names, trend_slopes, color=colors)
        ax3.set_title('Service Trend Analysis')
        ax3.set_ylabel('Trend Slope')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 要約テキスト
        summary = causal_results['summary']
        
        # 確信度の文字列を事前に準備
        if summary['top_root_cause']:
            service_name = summary['top_root_cause']['service']
            confidence = f"{summary['top_root_cause']['probability']:.3f}"
        else:
            service_name = 'None'
            confidence = 'N/A'
        
        summary_text = f"""
        分析結果要約:
        
        最有力根本原因: {service_name}
        確信度: {confidence}
        
        候補数: {summary['num_potential_causes']}
        
        上位3候補:
        """
        
        for i, cause in enumerate(summary['likely_root_causes'][:3]):
            summary_text += f"\n{i+1}. {cause['service']}: {cause['probability']:.3f}"
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='center')
        ax4.set_title('Analysis Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_sample_scenario():
    """サンプルシナリオの作成"""
    
    # 5つのマイクロサービス構成
    service_names = ['web-frontend', 'api-gateway', 'user-service', 'database', 'cache']
    num_services = len(service_names)
    
    # サービス依存グラフの作成
    # web -> api -> user-service -> database
    #              api -> cache
    adjacency_matrix = torch.zeros(num_services, num_services)
    adjacency_matrix[0, 1] = 1  # web -> api
    adjacency_matrix[1, 2] = 1  # api -> user-service
    adjacency_matrix[2, 3] = 1  # user-service -> database
    adjacency_matrix[1, 4] = 1  # api -> cache
    
    # ノード特徴量（各サービスの基本メトリクス）
    node_features = torch.randn(num_services, 12)  # 12次元の特徴量
    
    service_graph = ServiceGraph(
        adjacency_matrix=adjacency_matrix,
        node_features=node_features,
        service_names=service_names
    )
    
    # 異常シナリオの時系列データを生成
    seq_len = 20
    batch_size = 1
    # DCNNが期待する12次元の入力特徴量に合わせる
    input_features = 12
    
    # 正常時のベースライン
    baseline_metrics = torch.randn(batch_size, seq_len, input_features) * 0.1
    
    # データベースで異常が発生し、それが他のサービスに波及するシナリオ
    # 時刻10から異常が開始
    anomaly_start = 10
    
    # メトリクス全体に異常パターンを追加
    # 最初の3次元をCPU使用率、次の3次元をメモリ使用率、次の3次元をレスポンス時間、残りをその他メトリクスとする
    
    # データベースの異常（メトリクス9-11をデータベース関連とする）
    baseline_metrics[0, anomaly_start:, 9:12] += 2.0
    
    # user-serviceへの影響（メトリクス6-8を関連とする）
    baseline_metrics[0, anomaly_start+2:, 6:9] += 1.5
    
    # api-gatewayへの影響（メトリクス3-5を関連とする）
    baseline_metrics[0, anomaly_start+4:, 3:6] += 1.0
    
    # web-frontendへの影響（メトリクス0-2を関連とする）
    baseline_metrics[0, anomaly_start+6:, 0:3] += 0.5
    
    return service_graph, baseline_metrics, service_names


def main():
    """メイン実行関数"""
    
    print("🔍 OCEAN因果関係抽出のデモンストレーション")
    print("=" * 50)
    
    # 1. サンプルシナリオの作成
    print("📊 サンプルシナリオを作成中...")
    service_graph, metrics, service_names = create_sample_scenario()
    
    # 2. OCEANモデルの初期化
    print("🤖 OCEANモデルを初期化中...")
    config = default_config()
    # サンプルシナリオに合わせて設定を調整
    config.model.num_services = len(service_names)
    config.data.sequence_length = 20
    # 全ての次元を統一して32に設定
    config.model.gnn_hidden_dim = 32
    config.model.hidden_dim = 32
    config.model.attention_dim = 32
    config.model.temporal_dim = 32
    config.model.spatial_dim = 32
    try:
        model = OCEANModel(config)
        model.eval()
        print("✅ モデル初期化成功")
    except Exception as e:
        print(f"❌ モデル初期化でエラー: {e}")
        # フォールバック: より小さな設定を試す
        config.model.gnn_hidden_dim = 16
        config.model.hidden_dim = 16
        config.model.attention_dim = 16
        config.model.temporal_dim = 16
        config.model.spatial_dim = 16
        model = OCEANModel(config)
        model.eval()
        print("✅ フォールバック設定でモデル初期化成功")
    
    # 3. 因果関係抽出器の初期化
    print("🔬 因果関係抽出器を初期化中...")
    extractor = CausalRelationshipExtractor(model, service_names)
    
    # 4. 因果関係の抽出
    print("🎯 因果関係を抽出中...")
    causal_results = extractor.extract_causal_relationships(
        metrics=metrics,
        service_graph=service_graph,
        threshold=0.3  # 閾値を低く設定
    )
    
    # 5. 結果の表示
    print("\n📋 分析結果:")
    print("-" * 30)
    
    summary = causal_results['summary']
    print(f"最有力根本原因: {summary['top_root_cause']['service'] if summary['top_root_cause'] else 'None'}")
    
    if summary['top_root_cause']:
        print(f"確信度: {summary['top_root_cause']['probability']:.3f}")
    
    print(f"\n根本原因候補数: {summary['num_potential_causes']}")
    
    print("\n上位根本原因候補:")
    for i, cause in enumerate(summary['likely_root_causes'][:3]):
        print(f"  {i+1}. {cause['service']}: {cause['probability']:.3f} ({cause['confidence']})")
    
    # 6. サービス影響度の表示
    print("\n🌐 サービス影響度分析:")
    influences = causal_results['service_influences']
    for service, influence in influences.items():
        print(f"  {service}:")
        print(f"    外向き影響: {influence['outgoing_influence']:.3f}")
        print(f"    内向き影響: {influence['incoming_influence']:.3f}")
        print(f"    影響比率: {influence['influence_ratio']:.3f}")
    
    # 7. 時系列因果関係の表示
    print("\n⏰ 時系列因果関係分析:")
    temporal = causal_results['temporal_causality']
    
    print("  変化点:")
    for change in temporal['change_points'][:5]:  # 上位5つ
        print(f"    {change['service']} (時刻{change['time_index']}): {change['magnitude']:.3f}")
    
    print("\n  トレンド分析:")
    for service, trend in temporal['trend_analysis'].items():
        direction = "↗️" if trend['trend_direction'] == 'increasing' else "↘️"
        print(f"    {service}: {direction} {trend['trend_slope']:.3f}")
    
    # 8. 可視化
    print("\n📈 因果関係を可視化中...")
    try:
        extractor.visualize_causal_relationships(
            causal_results, 
            save_path="causal_analysis_results.png"
        )
        print("✅ 可視化結果を 'causal_analysis_results.png' に保存しました")
    except Exception as e:
        print(f"⚠️ 可視化でエラーが発生しました: {e}")
    
    print("\n🎉 因果関係抽出デモンストレーション完了!")
    
    return causal_results


if __name__ == "__main__":
    main()