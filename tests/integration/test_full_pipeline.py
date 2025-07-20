#!/usr/bin/env python3
"""
Full integration test for OCEAN model pipeline.
Tests the complete workflow from data loading to evaluation.
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ocean.configs.default_config import default_config
from ocean.models.ocean_model import OCEANModel, OCEANVariant
from ocean.models.training.trainer import Trainer, LossFunction
from ocean.models.training.online_learner import OnlineLearner
from ocean.evaluation.evaluator import Evaluator
from ocean.evaluation.metrics import PerformanceMetrics
from ocean.data.data_types import ServiceGraph, DatasetSample
import torch_geometric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data(config, num_samples=100):
    """Create synthetic data for testing."""
    logger.info(f"Creating {num_samples} synthetic samples")
    
    # Create service graph
    num_nodes = 5
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    
    node_features = torch.randn(num_nodes, 16)
    edge_features = torch.randn(edge_index.size(1), 8)
    
    graph_data = torch_geometric.data.Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features
    )
    
    # Create adjacency matrix from edge_index
    adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    for i, j in edge_index.T:
        adjacency_matrix[i, j] = 1.0
    
    service_graph = ServiceGraph(
        adjacency_matrix=adjacency_matrix,
        node_features=node_features,
        service_names=[f'service_{i}' for i in range(num_nodes)]
    )
    
    # Create dataset samples
    samples = []
    for i in range(num_samples):
        sample = DatasetSample(
            timestamp=torch.tensor(1640000000.0 + i),
            metrics=torch.randn(config.model.temporal_dim),
            logs=torch.randn(config.model.log_dim),
            graph=service_graph,
            label=torch.tensor(float(i % 2)),  # Alternating labels
            service_id=f'service_{i % num_nodes}',
            anomaly_type='performance' if i % 2 else 'normal'
        )
        samples.append(sample)
    
    return samples


def create_mock_data_loader(samples, batch_size=4):
    """Create mock data loader from samples."""
    class MockDataLoader:
        def __init__(self, samples, batch_size):
            self.samples = samples
            self.batch_size = batch_size
        
        def __iter__(self):
            for i in range(0, len(self.samples), self.batch_size):
                batch_samples = self.samples[i:i + self.batch_size]
                
                # Create batch tensors
                metrics = torch.stack([s.metrics for s in batch_samples])
                logs = torch.stack([s.logs for s in batch_samples])
                labels = torch.stack([s.label for s in batch_samples])
                graphs = [s.graph for s in batch_samples]
                
                yield {
                    'metrics': metrics,
                    'logs': logs,
                    'labels': labels,
                    'graphs': graphs,
                    'timestamps': torch.stack([s.timestamp for s in batch_samples]),
                    'service_ids': [s.service_id for s in batch_samples]
                }
        
        def __len__(self):
            return (len(self.samples) + self.batch_size - 1) // self.batch_size
    
    return MockDataLoader(samples, batch_size)


def create_mock_multimodal_loader(train_samples, val_samples, test_samples, batch_size=4):
    """Create mock multimodal data loader."""
    class MockMultiModalDataLoader:
        def __init__(self, train_samples, val_samples, test_samples, batch_size):
            self.train_loader = create_mock_data_loader(train_samples, batch_size)
            self.val_loader = create_mock_data_loader(val_samples, batch_size)
            self.test_loader = create_mock_data_loader(test_samples, batch_size)
        
        def get_all_loaders(self):
            return self.train_loader, self.val_loader, self.test_loader
        
        def get_split_statistics(self):
            return {
                'train_samples': len(self.train_loader.samples),
                'val_samples': len(self.val_loader.samples),
                'test_samples': len(self.test_loader.samples)
            }
    
    return MockMultiModalDataLoader(train_samples, val_samples, test_samples, batch_size)


def test_model_creation_and_forward():
    """Test model creation and forward pass."""
    logger.info("Testing model creation and forward pass")
    
    config = default_config()
    device = torch.device('cpu')
    
    # Test OCEAN model
    model = OCEANModel(config)
    model.to(device)
    
    # Test forward pass
    batch_size = 4
    # Use correct input dimensions based on what the model expects
    metrics_features = 12  # Default metrics features
    log_features = 768     # BERT embedding dimension
    metrics = torch.randn(batch_size, metrics_features)
    logs = torch.randn(batch_size, log_features)
    
    # Create simple graph
    num_nodes = 5
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    graph_data = torch_geometric.data.Data(
        x=torch.randn(num_nodes, 16),
        edge_index=edge_index
    )
    
    # Create simple adjacency matrix
    adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes - 1):
        adjacency_matrix[i, i + 1] = 1.0  # Simple chain
    
    # Use correct node features dimension (should match graph_input_dim)
    graph_features = 12  # Default graph features to match model config
    service_graph = ServiceGraph(
        adjacency_matrix=adjacency_matrix,
        node_features=torch.randn(num_nodes, graph_features),
        service_names=[f'service_{i}' for i in range(num_nodes)]
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        # Add sequence dimension for temporal data
        metrics_seq = metrics.unsqueeze(1)  # (batch_size, seq_len=1, features)
        logs_seq = logs.unsqueeze(1)       # (batch_size, seq_len=1, features)
        outputs = model(metrics_seq, service_graph, logs_seq)
    
    assert 'root_cause_probs' in outputs
    assert outputs['root_cause_probs'].shape == (batch_size,)
    
    logger.info("✓ Model creation and forward pass test passed")


def test_training_pipeline():
    """Test training pipeline."""
    logger.info("Testing training pipeline")
    
    config = default_config()
    config.training.num_epochs = 2  # Quick test
    config.data.batch_size = 4
    
    device = torch.device('cpu')
    
    # Create model
    model = OCEANModel(config)
    model.to(device)
    
    # Create synthetic data
    train_samples = create_synthetic_data(config, num_samples=40)
    val_samples = create_synthetic_data(config, num_samples=20)
    test_samples = create_synthetic_data(config, num_samples=20)
    
    # Create data loader
    data_loader = create_mock_multimodal_loader(
        train_samples, val_samples, test_samples, batch_size=config.data.batch_size
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        data_loader=data_loader,
        device=device
    )
    
    # Train model
    training_history = trainer.train(num_epochs=2)
    
    assert 'train_history' in training_history
    assert 'val_history' in training_history
    
    logger.info("✓ Training pipeline test passed")


def test_online_learning():
    """Test online learning functionality."""
    logger.info("Testing online learning")
    
    config = default_config()
    device = torch.device('cpu')
    
    # Create model
    model = OCEANModel(config)
    model.to(device)
    
    # Create online learner
    online_learner = OnlineLearner(model, config)
    
    # Create streaming samples
    samples = create_synthetic_data(config, num_samples=20)
    
    # Process samples online
    results = []
    for sample in samples[:10]:  # Test with subset
        result = online_learner.process_sample(sample, update_model=True)
        results.append(result)
    
    assert len(results) == 10
    assert all('loss' in result for result in results)
    
    # Test performance summary
    summary = online_learner.get_performance_summary()
    assert 'total_updates' in summary
    assert summary['total_updates'] > 0
    
    logger.info("✓ Online learning test passed")


def test_evaluation_framework():
    """Test evaluation framework."""
    logger.info("Testing evaluation framework")
    
    config = default_config()
    device = torch.device('cpu')
    
    # Create model
    model = OCEANModel(config)
    model.to(device)
    
    # Create synthetic data
    train_samples = create_synthetic_data(config, num_samples=40)
    val_samples = create_synthetic_data(config, num_samples=20)
    test_samples = create_synthetic_data(config, num_samples=20)
    
    # Create data loader
    data_loader = create_mock_multimodal_loader(
        train_samples, val_samples, test_samples, batch_size=config.data.batch_size
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        config=config,
        data_loader=data_loader,
        device=device
    )
    
    # Evaluate model
    _, _, test_loader = data_loader.get_all_loaders()
    evaluation_results = evaluator.evaluate_model(
        test_loader=test_loader,
        save_results=False
    )
    
    assert 'classification_metrics' in evaluation_results
    assert 'ranking_metrics' in evaluation_results
    assert 'profiling_results' in evaluation_results
    
    # Test performance metrics
    metrics = PerformanceMetrics()
    predictions = torch.tensor([1, 0, 1, 0, 1])
    ground_truth = torch.tensor([1, 0, 1, 1, 0])
    scores = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.3])
    
    metrics.add_predictions(predictions, ground_truth, scores)
    classification_metrics = metrics.compute_classification_metrics()
    
    assert 'accuracy' in classification_metrics
    assert 'f1_score' in classification_metrics
    
    logger.info("✓ Evaluation framework test passed")


def test_ablation_study():
    """Test ablation study functionality."""
    logger.info("Testing ablation study")
    
    config = default_config()
    device = torch.device('cpu')
    
    # Create variant model for ablation
    model = OCEANVariant(config)
    model.to(device)
    
    # Create synthetic data
    test_samples = create_synthetic_data(config, num_samples=20)
    data_loader = create_mock_multimodal_loader(
        test_samples, test_samples, test_samples, batch_size=config.data.batch_size
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        config=config,
        data_loader=data_loader,
        device=device
    )
    
    # Define ablation configurations
    ablation_configs = [
        {'disable_attention': True},
        {'disable_graph_fusion': True},
        {'disable_attention': True, 'disable_graph_fusion': True}
    ]
    
    # Run ablation study
    _, _, test_loader = data_loader.get_all_loaders()
    ablation_results = evaluator.evaluate_ablation_study(
        ablation_configs=ablation_configs,
        test_loader=test_loader
    )
    
    assert 'ablation_results' in ablation_results
    assert 'analysis' in ablation_results
    assert len(ablation_results['ablation_results']) == len(ablation_configs)
    
    logger.info("✓ Ablation study test passed")


def main():
    """Run all integration tests."""
    logger.info("Starting OCEAN integration tests")
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run tests
        test_model_creation_and_forward()
        test_training_pipeline()
        test_online_learning()
        test_evaluation_framework()
        test_ablation_study()
        
        logger.info("🎉 All integration tests passed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)