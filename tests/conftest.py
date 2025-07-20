"""
Pytest configuration and fixtures for OCEAN tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

from ocean.configs.default_config import OCEANConfig, default_config
from ocean.data.data_types import ServiceGraph, DatasetSample
from ocean.models.ocean_model import OCEANModel


@pytest.fixture
def device():
    """Get test device (CPU only for tests)."""
    return torch.device('cpu')


@pytest.fixture
def config():
    """Get test configuration."""
    config = default_config()
    # Override for testing
    config.model.temporal_dim = 32
    config.model.spatial_dim = 16
    config.model.log_dim = 64
    config.model.fusion_dim = 64
    config.data.batch_size = 4
    config.data.sequence_length = 10
    config.training.num_epochs = 2
    config.training.learning_rate = 0.001
    return config


@pytest.fixture
def sample_service_graph():
    """Create a sample service graph for testing."""
    import torch_geometric
    
    # Create simple graph with 5 nodes
    num_nodes = 5
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    
    node_features = torch.randn(num_nodes, 16)
    edge_features = torch.randn(edge_index.size(1), 8)
    
    data = torch_geometric.data.Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features
    )
    
    return ServiceGraph(
        nodes=['service_a', 'service_b', 'service_c', 'service_d', 'service_e'],
        edges=[('service_a', 'service_b'), ('service_b', 'service_c'), 
               ('service_c', 'service_d'), ('service_d', 'service_e')],
        node_features=node_features,
        edge_features=edge_features,
        graph_data=data
    )


@pytest.fixture
def sample_dataset_sample(config, sample_service_graph):
    """Create a sample dataset sample for testing."""
    return DatasetSample(
        timestamp=torch.tensor(1640000000.0),
        metrics=torch.randn(config.model.temporal_dim),
        logs=torch.randn(config.model.log_dim),
        graph=sample_service_graph,
        label=torch.tensor(1.0),
        service_id='test_service',
        anomaly_type='performance'
    )


@pytest.fixture
def sample_batch_data(config, sample_service_graph):
    """Create sample batch data for testing."""
    batch_size = config.data.batch_size
    
    return {
        'metrics': torch.randn(batch_size, config.model.temporal_dim),
        'logs': torch.randn(batch_size, config.model.log_dim),
        'graphs': [sample_service_graph] * batch_size,
        'labels': torch.randint(0, 2, (batch_size,)).float(),
        'timestamps': torch.tensor([1640000000.0 + i for i in range(batch_size)]),
        'service_ids': [f'service_{i}' for i in range(batch_size)]
    }


@pytest.fixture
def ocean_model(config, device):
    """Create OCEAN model for testing."""
    model = OCEANModel(config)
    model.to(device)
    return model


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_data_loader(sample_batch_data):
    """Create mock data loader for testing."""
    class MockDataLoader:
        def __init__(self, batch_data):
            self.batch_data = batch_data
            
        def __iter__(self):
            yield self.batch_data
            
        def __len__(self):
            return 1
    
    return MockDataLoader(sample_batch_data)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


def pytest_configure(config):
    """Configure pytest."""
    # Set random seeds for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Disable CUDA for tests
    torch.cuda.is_available = lambda: False