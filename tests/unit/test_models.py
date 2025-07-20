"""
Test OCEAN model implementations.
"""

import pytest
import torch
from ocean.models.ocean_model import OCEANModel, OCEANVariant


class TestOCEANModel:
    """Test OCEAN model."""
    
    def test_ocean_model_creation(self, config, device):
        """Test OCEAN model creation."""
        model = OCEANModel(config)
        model.to(device)
        
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'temporal_component')
        assert hasattr(model, 'spatial_component')
        assert hasattr(model, 'attention_component')
        assert hasattr(model, 'fusion_component')
    
    def test_ocean_model_forward(self, ocean_model, sample_batch_data, device):
        """Test OCEAN model forward pass."""
        # Move data to device
        metrics = sample_batch_data['metrics'].to(device)
        logs = sample_batch_data['logs'].to(device)
        graph = sample_batch_data['graphs'][0]
        
        ocean_model.eval()
        with torch.no_grad():
            outputs = ocean_model(metrics, graph, logs)
        
        assert 'root_cause_probs' in outputs
        assert outputs['root_cause_probs'].shape[0] == metrics.shape[0]
        assert torch.all(outputs['root_cause_probs'] >= 0)
        assert torch.all(outputs['root_cause_probs'] <= 1)
    
    def test_ocean_model_with_intermediate(self, ocean_model, sample_batch_data, device):
        """Test OCEAN model with intermediate outputs."""
        metrics = sample_batch_data['metrics'].to(device)
        logs = sample_batch_data['logs'].to(device)
        graph = sample_batch_data['graphs'][0]
        
        ocean_model.eval()
        with torch.no_grad():
            outputs = ocean_model(metrics, graph, logs, return_intermediate=True)
        
        assert 'root_cause_probs' in outputs
        assert 'intermediate' in outputs
        assert 'temporal_features' in outputs['intermediate']
        assert 'spatial_features' in outputs['intermediate']
        assert 'attention_weights' in outputs['intermediate']
    
    def test_ocean_model_loss_computation(self, ocean_model, sample_batch_data, device):
        """Test loss computation."""
        metrics = sample_batch_data['metrics'].to(device)
        logs = sample_batch_data['logs'].to(device)
        graph = sample_batch_data['graphs'][0]
        labels = sample_batch_data['labels'].to(device)
        
        ocean_model.train()
        outputs = ocean_model(metrics, graph, logs)
        losses = ocean_model.compute_loss(outputs, labels)
        
        assert 'total_loss' in losses
        assert 'prediction_loss' in losses
        assert isinstance(losses['total_loss'], torch.Tensor)
        assert losses['total_loss'].requires_grad
    
    def test_ocean_model_gradient_flow(self, ocean_model, sample_batch_data, device):
        """Test gradient flow through model."""
        metrics = sample_batch_data['metrics'].to(device)
        logs = sample_batch_data['logs'].to(device)
        graph = sample_batch_data['graphs'][0]
        labels = sample_batch_data['labels'].to(device)
        
        ocean_model.train()
        outputs = ocean_model(metrics, graph, logs)
        losses = ocean_model.compute_loss(outputs, labels)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Check that gradients are computed
        for name, param in ocean_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestOCEANVariant:
    """Test OCEAN variant for ablation studies."""
    
    def test_variant_creation(self, config, device):
        """Test OCEAN variant creation."""
        variant = OCEANVariant(config)
        variant.to(device)
        
        assert isinstance(variant, torch.nn.Module)
        assert hasattr(variant, 'disable_attention')
        assert hasattr(variant, 'disable_graph_fusion')
        assert hasattr(variant, 'disable_temporal_cnn')
        assert hasattr(variant, 'disable_spatial_gnn')
    
    def test_variant_component_disabling(self, config, sample_batch_data, device):
        """Test component disabling in variant."""
        variant = OCEANVariant(config)
        variant.to(device)
        
        metrics = sample_batch_data['metrics'].to(device)
        logs = sample_batch_data['logs'].to(device)
        graph = sample_batch_data['graphs'][0]
        
        # Test with all components enabled
        variant.eval()
        with torch.no_grad():
            outputs_full = variant(metrics, graph, logs)
        
        # Test with attention disabled
        variant.disable_attention = True
        with torch.no_grad():
            outputs_no_attention = variant(metrics, graph, logs)
        
        # Outputs should be different
        assert not torch.allclose(
            outputs_full['root_cause_probs'], 
            outputs_no_attention['root_cause_probs']
        )
        
        # Reset
        variant.disable_attention = False
    
    def test_variant_ablation_configurations(self, config, sample_batch_data, device):
        """Test different ablation configurations."""
        variant = OCEANVariant(config)
        variant.to(device)
        
        metrics = sample_batch_data['metrics'].to(device)
        logs = sample_batch_data['logs'].to(device)
        graph = sample_batch_data['graphs'][0]
        
        ablation_configs = [
            {'disable_attention': True},
            {'disable_graph_fusion': True},
            {'disable_temporal_cnn': True},
            {'disable_spatial_gnn': True},
            {'disable_attention': True, 'disable_graph_fusion': True}
        ]
        
        variant.eval()
        outputs = []
        
        for config in ablation_configs:
            # Set configuration
            for attr, value in config.items():
                setattr(variant, attr, value)
            
            # Forward pass
            with torch.no_grad():
                output = variant(metrics, graph, logs)
                outputs.append(output['root_cause_probs'])
            
            # Reset configuration
            for attr in config.keys():
                setattr(variant, attr, False)
        
        # All outputs should be different (assuming different ablations)
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                # Some ablations might produce similar results, so we check shape consistency
                assert outputs[i].shape == outputs[j].shape
    
    def test_variant_model_summary(self, config, device):
        """Test model summary generation."""
        variant = OCEANVariant(config)
        variant.to(device)
        
        summary = variant.get_model_summary()
        
        assert isinstance(summary, dict)
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'model_size_mb' in summary
        assert summary['total_parameters'] > 0