#!/usr/bin/env python3
"""
Simple test to verify basic OCEAN functionality.
"""

import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from ocean.configs.default_config import default_config, get_device
        from ocean.data.data_types import ServiceGraph, DatasetSample
        from ocean.models.components.dilated_cnn import DilatedCNN
        from ocean.models.components.graph_neural_network import GraphNeuralNetwork
        from ocean.models.components.multi_factor_attention import MultiFactorAttention
        from ocean.models.components.graph_fusion import GraphFusionModule
        from ocean.evaluation.metrics import PerformanceMetrics
        
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration creation."""
    logger.info("Testing configuration...")
    
    try:
        from ocean.configs.default_config import default_config, get_device
        
        config = default_config()
        device = get_device()
        
        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'training')
        assert hasattr(config, 'system')
        
        logger.info("✓ Configuration test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_data_types():
    """Test data types creation."""
    logger.info("Testing data types...")
    
    try:
        from ocean.data.data_types import ServiceGraph, DatasetSample
        
        # Create service graph
        adjacency_matrix = torch.eye(3)  # 3x3 identity matrix
        node_features = torch.randn(3, 12)
        
        service_graph = ServiceGraph(
            adjacency_matrix=adjacency_matrix,
            node_features=node_features,
            service_names=['service_a', 'service_b', 'service_c']
        )
        
        assert service_graph.num_services == 3
        
        # Create dataset sample
        from datetime import datetime
        
        sample = DatasetSample(
            timestamp=datetime.now(),
            metrics=torch.randn(10, 12),  # (seq_len, num_metrics)
            logs=torch.randn(5, 768),     # (num_logs, log_embedding_dim)  
            graph=service_graph,
            label=torch.tensor([1.0, 0.0, 0.0])  # (num_services,) binary labels
        )
        
        assert sample.metrics.shape == (10, 12)
        assert sample.logs.shape == (5, 768)
        assert sample.label.shape == (3,)
        
        logger.info("✓ Data types test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Data types test failed: {e}")
        return False

def test_components():
    """Test individual components."""
    logger.info("Testing model components...")
    
    try:
        from ocean.models.components.dilated_cnn import DilatedCNN
        from ocean.models.components.graph_neural_network import GraphNeuralNetwork
        from ocean.models.components.multi_factor_attention import MultiFactorAttention
        from ocean.models.components.graph_fusion import GraphFusionModule
        
        # Test DilatedCNN
        dcnn = DilatedCNN(
            input_dim=12,
            channels=[32, 64],
            dilation_rates=[1, 2],  # Must match channels length
            output_dim=128
        )
        
        x_temporal = torch.randn(2, 10, 12)  # (batch, seq, features)
        temporal_out = dcnn(x_temporal)
        assert temporal_out.shape == (2, 128)
        
        # Test MultiFactorAttention
        attention = MultiFactorAttention(
            temporal_dim=128,
            spatial_dim=128,
            log_dim=768,
            attention_dim=64
        )
        
        temporal_feat = torch.randn(2, 128)
        spatial_feat = torch.randn(2, 128)
        log_feat = torch.randn(2, 768)
        
        attention_out = attention(temporal_feat, spatial_feat, log_feat)
        assert attention_out.shape[0] == 2  # batch size preserved
        
        # Test GraphFusionModule
        fusion = GraphFusionModule(
            temporal_dim=128,
            spatial_dim=128,
            log_dim=768,
            fusion_dim=256
        )
        
        fusion_result = fusion(temporal_feat, spatial_feat, log_feat)
        assert isinstance(fusion_result, dict)
        assert 'fused_representation' in fusion_result
        fused_out = fusion_result['fused_representation']
        assert fused_out.shape == (2, 256)
        
        logger.info("✓ Components test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation():
    """Test evaluation metrics."""
    logger.info("Testing evaluation framework...")
    
    try:
        from ocean.evaluation.metrics import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        # Add some predictions
        predictions = torch.tensor([1, 0, 1, 0, 1])
        ground_truth = torch.tensor([1, 0, 1, 1, 0])
        scores = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.3])
        
        metrics.add_predictions(predictions, ground_truth, scores)
        
        # Compute metrics
        classification_metrics = metrics.compute_classification_metrics()
        ranking_metrics = metrics.compute_ranking_metrics([1, 3, 5])
        
        assert 'accuracy' in classification_metrics
        assert 'f1_score' in classification_metrics
        assert 'precision@1' in ranking_metrics
        
        logger.info("✓ Evaluation test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Evaluation test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    logger.info("🧪 Running OCEAN basic functionality tests")
    
    tests = [
        test_imports,
        test_configuration,
        test_data_types,
        test_components,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            break  # Stop on first failure for easier debugging
    
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All basic tests passed! OCEAN implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)