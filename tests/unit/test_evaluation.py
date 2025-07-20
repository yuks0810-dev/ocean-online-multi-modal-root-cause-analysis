"""
Test evaluation framework.
"""

import pytest
import torch
import numpy as np
from ocean.evaluation.metrics import PerformanceMetrics, BaselineComparator, StatisticalSignificance
from ocean.evaluation.evaluator import Evaluator
from ocean.evaluation.profiler import ModelProfiler, MemoryProfiler


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = PerformanceMetrics()
        assert len(metrics.predictions) == 0
        assert len(metrics.ground_truth) == 0
    
    def test_add_predictions(self):
        """Test adding predictions."""
        metrics = PerformanceMetrics()
        
        predictions = torch.tensor([1, 0, 1, 1, 0])
        ground_truth = torch.tensor([1, 0, 1, 0, 0])
        scores = torch.tensor([0.9, 0.1, 0.8, 0.6, 0.2])
        
        metrics.add_predictions(predictions, ground_truth, scores)
        
        assert len(metrics.predictions) == 5
        assert len(metrics.ground_truth) == 5
        assert len(metrics.prediction_scores) == 5
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        metrics = PerformanceMetrics()
        
        # Perfect predictions
        predictions = torch.tensor([1, 0, 1, 0])
        ground_truth = torch.tensor([1, 0, 1, 0])
        scores = torch.tensor([0.9, 0.1, 0.8, 0.2])
        
        metrics.add_predictions(predictions, ground_truth, scores)
        
        results = metrics.compute_classification_metrics()
        
        assert results['accuracy'] == 1.0
        assert results['precision'] == 1.0
        assert results['recall'] == 1.0
        assert results['f1_score'] == 1.0
    
    def test_ranking_metrics(self):
        """Test ranking metrics computation."""
        metrics = PerformanceMetrics()
        
        predictions = torch.tensor([1, 0, 1, 0, 1])
        ground_truth = torch.tensor([1, 0, 1, 1, 0])
        scores = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.3])
        
        metrics.add_predictions(predictions, ground_truth, scores)
        
        ranking_results = metrics.compute_ranking_metrics([1, 3, 5])
        
        assert 'precision@1' in ranking_results
        assert 'precision@3' in ranking_results
        assert 'precision@5' in ranking_results
        assert 'hit_rate@1' in ranking_results
    
    def test_temporal_metrics(self):
        """Test temporal metrics computation."""
        metrics = PerformanceMetrics()
        
        # Add enough predictions for temporal analysis
        for i in range(150):
            pred = torch.tensor([i % 2])
            truth = torch.tensor([i % 2])
            score = torch.tensor([0.9 if i % 2 else 0.1])
            metrics.add_predictions(pred, truth, score)
        
        temporal_results = metrics.compute_temporal_metrics(window_size=50)
        
        assert 'accuracy' in temporal_results
        assert len(temporal_results['accuracy']) > 0


class TestBaselineComparator:
    """Test baseline comparison."""
    
    def test_comparator_creation(self):
        """Test comparator creation."""
        comparator = BaselineComparator()
        assert len(comparator.baselines) == 0
        assert comparator.ocean_metrics is None
    
    def test_add_baseline(self):
        """Test adding baseline results."""
        comparator = BaselineComparator()
        
        baseline_metrics = {
            'accuracy': 0.8,
            'f1_score': 0.75,
            'precision': 0.7
        }
        
        comparator.add_baseline('random_forest', baseline_metrics)
        
        assert 'random_forest' in comparator.baselines
        assert comparator.baselines['random_forest'] == baseline_metrics
    
    def test_comparison(self):
        """Test baseline comparison."""
        comparator = BaselineComparator()
        
        ocean_metrics = {
            'accuracy': 0.9,
            'f1_score': 0.85,
            'precision': 0.8
        }
        
        baseline_metrics = {
            'accuracy': 0.8,
            'f1_score': 0.75,
            'precision': 0.7
        }
        
        comparator.set_ocean_metrics(ocean_metrics)
        comparator.add_baseline('baseline', baseline_metrics)
        
        comparisons = comparator.compare_all()
        
        assert 'baseline' in comparisons
        assert 'accuracy_improvement' in comparisons['baseline']
        assert comparisons['baseline']['accuracy_improvement'] > 0  # OCEAN should be better


class TestStatisticalSignificance:
    """Test statistical significance testing."""
    
    def test_significance_creation(self):
        """Test significance tester creation."""
        tester = StatisticalSignificance(alpha=0.05)
        assert tester.alpha == 0.05
    
    def test_paired_t_test(self):
        """Test paired t-test."""
        tester = StatisticalSignificance()
        
        # Create two sets of scores (OCEAN should be better)
        ocean_scores = [0.9, 0.85, 0.88, 0.92, 0.87]
        baseline_scores = [0.8, 0.75, 0.78, 0.82, 0.77]
        
        results = tester.paired_t_test(ocean_scores, baseline_scores)
        
        assert 'statistic' in results
        assert 'p_value' in results
        assert 'cohen_d' in results
        assert 'is_significant' in results
        assert isinstance(results['is_significant'], bool)
    
    def test_wilcoxon_test(self):
        """Test Wilcoxon test."""
        tester = StatisticalSignificance()
        
        ocean_scores = [0.9, 0.85, 0.88, 0.92, 0.87]
        baseline_scores = [0.8, 0.75, 0.78, 0.82, 0.77]
        
        results = tester.wilcoxon_test(ocean_scores, baseline_scores)
        
        assert 'statistic' in results
        assert 'p_value' in results
        assert 'is_significant' in results
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction."""
        tester = StatisticalSignificance()
        
        p_values = [0.01, 0.03, 0.08, 0.12, 0.02]
        
        results = tester.multiple_comparison_correction(p_values, method='bonferroni')
        
        assert 'corrected_p_values' in results
        assert 'significant_tests' in results
        assert len(results['corrected_p_values']) == len(p_values)


class TestEvaluator:
    """Test main evaluator."""
    
    def test_evaluator_creation(self, ocean_model, config, mock_data_loader):
        """Test evaluator creation."""
        # Create a mock data loader
        from ocean.data.loaders.multimodal_dataset import MultiModalDataLoader
        
        # For testing, we'll create a simple mock
        class MockMultiModalDataLoader:
            def get_all_loaders(self):
                return mock_data_loader, mock_data_loader, mock_data_loader
            
            def get_split_statistics(self):
                return {'train_samples': 100, 'val_samples': 20, 'test_samples': 20}
        
        mock_multimodal_loader = MockMultiModalDataLoader()
        
        evaluator = Evaluator(
            model=ocean_model,
            config=config,
            data_loader=mock_multimodal_loader
        )
        
        assert evaluator.model == ocean_model
        assert evaluator.config == config
    
    def test_model_profiler_creation(self, ocean_model):
        """Test model profiler creation."""
        profiler = ModelProfiler(ocean_model)
        assert profiler.model == ocean_model
    
    def test_memory_profiler_creation(self):
        """Test memory profiler creation."""
        profiler = MemoryProfiler()
        assert hasattr(profiler, 'memory_snapshots')
        assert hasattr(profiler, 'memory_timeline')
    
    def test_memory_profiling(self):
        """Test memory profiling functionality."""
        profiler = MemoryProfiler()
        
        def simple_operation():
            x = torch.randn(100, 100)
            y = torch.matmul(x, x.T)
            return y
        
        results = profiler.profile_memory_usage(simple_operation, num_runs=3)
        
        assert 'avg_memory_usage_mb' in results
        assert 'memory_usage_samples' in results
        assert len(results['memory_usage_samples']) == 3