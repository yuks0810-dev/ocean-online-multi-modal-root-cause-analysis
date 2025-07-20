"""
Test individual model components.
"""

import pytest
import torch
import torch.nn as nn

from ocean.models.components.dilated_cnn import DilatedCNN
from ocean.models.components.graph_neural_network import GraphNeuralNetwork
from ocean.models.components.multi_factor_attention import MultiFactorAttention
from ocean.models.components.graph_fusion import GraphFusionModule


class TestDilatedCNN:
    """Test Dilated CNN component."""
    
    def test_dilated_cnn_creation(self, config):
        """Test DilatedCNN creation."""
        cnn = DilatedCNN(
            input_dim=config.model.temporal_dim,
            hidden_dim=64,
            output_dim=32,
            num_layers=3
        )
        
        assert isinstance(cnn, nn.Module)
        assert len(cnn.conv_layers) == 3
    
    def test_dilated_cnn_forward(self, config):
        """Test DilatedCNN forward pass."""
        batch_size = 4
        seq_length = 10
        
        cnn = DilatedCNN(
            input_dim=config.model.temporal_dim,
            hidden_dim=64,
            output_dim=32,
            num_layers=3
        )
        
        # Input: (batch_size, seq_length, input_dim)
        x = torch.randn(batch_size, seq_length, config.model.temporal_dim)
        
        output = cnn(x)
        
        # Output should have shape (batch_size, seq_length, output_dim)
        assert output.shape == (batch_size, seq_length, 32)
    
    def test_dilated_cnn_different_configs(self):
        """Test DilatedCNN with different configurations."""
        configs = [
            {'input_dim': 64, 'hidden_dim': 32, 'output_dim': 16, 'num_layers': 2},
            {'input_dim': 128, 'hidden_dim': 64, 'output_dim': 32, 'num_layers': 4},
        ]
        
        for config in configs:
            cnn = DilatedCNN(**config)
            x = torch.randn(2, 5, config['input_dim'])
            output = cnn(x)
            assert output.shape == (2, 5, config['output_dim'])


class TestGraphNeuralNetwork:
    """Test Graph Neural Network component."""
    
    def test_gnn_creation(self, config):
        """Test GNN creation."""
        gnn = GraphNeuralNetwork(
            node_dim=config.model.spatial_dim,
            edge_dim=32,
            hidden_dim=64,
            output_dim=32,
            num_layers=2
        )
        
        assert isinstance(gnn, nn.Module)
    
    def test_gnn_forward(self, config, sample_service_graph):
        """Test GNN forward pass."""
        gnn = GraphNeuralNetwork(
            node_dim=16,  # Match sample graph node features
            edge_dim=8,   # Match sample graph edge features
            hidden_dim=32,
            output_dim=16,
            num_layers=2
        )
        
        output = gnn(sample_service_graph.graph_data)
        
        # Output should have shape (num_nodes, output_dim)
        assert output.shape == (5, 16)


class TestMultiFactorAttention:
    """Test Multi-Factor Attention component."""
    
    def test_attention_creation(self, config):
        """Test attention mechanism creation."""
        attention = MultiFactorAttention(
            temporal_dim=config.model.temporal_dim,
            spatial_dim=config.model.spatial_dim,
            log_dim=config.model.log_dim,
            attention_dim=64,
            num_heads=4
        )
        
        assert isinstance(attention, nn.Module)
    
    def test_attention_forward(self, config):
        """Test attention forward pass."""
        batch_size = 4
        
        attention = MultiFactorAttention(
            temporal_dim=config.model.temporal_dim,
            spatial_dim=config.model.spatial_dim,
            log_dim=config.model.log_dim,
            attention_dim=64,
            num_heads=4
        )
        
        temporal_features = torch.randn(batch_size, config.model.temporal_dim)
        spatial_features = torch.randn(batch_size, config.model.spatial_dim)
        log_features = torch.randn(batch_size, config.model.log_dim)
        
        output = attention(temporal_features, spatial_features, log_features)
        
        # Output should have consistent batch dimension
        assert output.shape[0] == batch_size
    
    def test_attention_different_fusion_strategies(self, config):
        """Test different fusion strategies."""
        strategies = ['concatenate', 'add', 'multiply']
        batch_size = 2
        
        for strategy in strategies:
            attention = MultiFactorAttention(
                temporal_dim=config.model.temporal_dim,
                spatial_dim=config.model.spatial_dim,
                log_dim=config.model.log_dim,
                attention_dim=64,
                num_heads=4,
                fusion_strategy=strategy
            )
            
            temporal_features = torch.randn(batch_size, config.model.temporal_dim)
            spatial_features = torch.randn(batch_size, config.model.spatial_dim)
            log_features = torch.randn(batch_size, config.model.log_dim)
            
            output = attention(temporal_features, spatial_features, log_features)
            assert output.shape[0] == batch_size


class TestGraphFusionModule:
    """Test Graph Fusion Module."""
    
    def test_fusion_creation(self, config):
        """Test fusion module creation."""
        fusion = GraphFusionModule(
            temporal_dim=config.model.temporal_dim,
            spatial_dim=config.model.spatial_dim,
            log_dim=config.model.log_dim,
            fusion_dim=128
        )
        
        assert isinstance(fusion, nn.Module)
    
    def test_fusion_forward(self, config):
        """Test fusion forward pass."""
        batch_size = 4
        
        fusion = GraphFusionModule(
            temporal_dim=config.model.temporal_dim,
            spatial_dim=config.model.spatial_dim,
            log_dim=config.model.log_dim,
            fusion_dim=128
        )
        
        temporal_features = torch.randn(batch_size, config.model.temporal_dim)
        spatial_features = torch.randn(batch_size, config.model.spatial_dim)
        log_features = torch.randn(batch_size, config.model.log_dim)
        
        fused_output, contrastive_losses = fusion(
            temporal_features, spatial_features, log_features
        )
        
        assert fused_output.shape == (batch_size, 128)
        assert isinstance(contrastive_losses, dict)
    
    def test_fusion_contrastive_learning(self, config):
        """Test contrastive learning component."""
        batch_size = 8  # Larger batch for contrastive learning
        
        fusion = GraphFusionModule(
            temporal_dim=config.model.temporal_dim,
            spatial_dim=config.model.spatial_dim,
            log_dim=config.model.log_dim,
            fusion_dim=128
        )
        
        temporal_features = torch.randn(batch_size, config.model.temporal_dim)
        spatial_features = torch.randn(batch_size, config.model.spatial_dim)
        log_features = torch.randn(batch_size, config.model.log_dim)
        
        fused_output, contrastive_losses = fusion(
            temporal_features, spatial_features, log_features
        )
        
        # Check that contrastive losses are computed
        assert len(contrastive_losses) > 0
        for loss_name, loss_value in contrastive_losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.requires_grad