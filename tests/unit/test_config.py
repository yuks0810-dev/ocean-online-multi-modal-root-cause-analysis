"""
Test configuration management.
"""

import pytest
import torch
from ocean.configs.default_config import OCEANConfig, default_config, get_device


class TestOCEANConfig:
    """Test OCEAN configuration."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = default_config()
        
        assert isinstance(config, OCEANConfig)
        assert config.model.temporal_dim > 0
        assert config.model.spatial_dim > 0
        assert config.model.log_dim > 0
        assert config.data.batch_size > 0
        assert config.training.learning_rate > 0
    
    def test_config_sections(self):
        """Test configuration sections."""
        config = default_config()
        
        # Check all required sections exist
        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'training')
        assert hasattr(config, 'system')
    
    def test_model_config(self):
        """Test model configuration."""
        config = default_config()
        
        assert config.model.temporal_dim == 128
        assert config.model.spatial_dim == 64
        assert config.model.log_dim == 256
        assert config.model.attention_dim == 128
        assert config.model.num_heads == 8
    
    def test_data_config(self):
        """Test data configuration."""
        config = default_config()
        
        assert config.data.batch_size == 32
        assert config.data.sequence_length == 100
        assert config.data.test_split == 0.2
        assert config.data.val_split == 0.1
    
    def test_training_config(self):
        """Test training configuration."""
        config = default_config()
        
        assert config.training.num_epochs == 100
        assert config.training.learning_rate == 0.001
        assert config.training.weight_decay == 1e-4
        assert config.training.early_stopping_patience == 10
    
    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'data' in config_dict
        assert 'training' in config_dict
        assert 'system' in config_dict
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        # In test environment, should default to CPU
        assert device.type == 'cpu'