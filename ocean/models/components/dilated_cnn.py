"""
Dilated Convolutional Neural Network (DCNN) component for OCEAN model.
Extracts temporal features from metrics time series using dilated convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import logging


logger = logging.getLogger(__name__)


class DilatedConvBlock(nn.Module):
    """
    Single dilated convolution block with residual connection.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize dilated convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super(DilatedConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        # Main convolution path
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Residual connection
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.residual_bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dilated convolution block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        # Main path
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual:
            residual = x
        else:
            residual = self.residual_conv(x)
            residual = self.residual_bn(residual)
        
        return out + residual


class DilatedCNN(nn.Module):
    """
    Dilated Convolutional Neural Network for temporal feature extraction.
    Processes metrics time series with increasing dilation rates.
    """
    
    def __init__(self,
                 input_dim: int,
                 channels: List[int] = None,
                 kernel_size: int = 3,
                 dilation_rates: List[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 output_dim: Optional[int] = None,
                 global_pooling: str = 'mean'):
        """
        Initialize Dilated CNN.
        
        Args:
            input_dim: Number of input features (metrics)
            channels: List of channel dimensions for each layer
            kernel_size: Convolution kernel size
            dilation_rates: List of dilation rates for each layer
            dropout: Dropout probability
            activation: Activation function
            output_dim: Final output dimension (if None, uses last channel dim)
            global_pooling: Global pooling method ('mean', 'max', 'attention')
        """
        super(DilatedCNN, self).__init__()
        
        if channels is None:
            channels = [64, 128, 256, 512]
        
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
        
        if len(channels) != len(dilation_rates):
            raise ValueError("Length of channels and dilation_rates must match")
        
        self.input_dim = input_dim
        self.channels = channels
        self.dilation_rates = dilation_rates
        self.global_pooling = global_pooling
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, channels[0])
        
        # Dilated convolution layers
        self.conv_layers = nn.ModuleList()
        
        for i in range(len(channels)):
            in_channels = channels[i]
            out_channels = channels[i]  # Keep same channel size within layer
            dilation = dilation_rates[i]
            
            conv_block = DilatedConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                activation=activation
            )
            
            self.conv_layers.append(conv_block)
        
        # Channel transition layers (increase channels between stages)
        self.transition_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            transition = nn.Sequential(
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.transition_layers.append(transition)
        
        # Global pooling
        if global_pooling == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(channels[-1], channels[-1] // 4),
                nn.ReLU(),
                nn.Linear(channels[-1] // 4, 1),
                nn.Softmax(dim=-1)
            )
        
        # Output projection
        final_dim = output_dim if output_dim is not None else channels[-1]
        self.output_projection = nn.Linear(channels[-1], final_dim)
        
        logger.info(f"Initialized DilatedCNN with {len(channels)} layers, "
                   f"channels={channels}, dilations={dilation_rates}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Dilated CNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Input projection: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, channels[0])
        x = self.input_projection(x)
        
        # Transpose for convolution: (batch_size, channels[0], seq_len)
        x = x.transpose(1, 2)
        
        # Apply dilated convolution layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            
            # Apply transition layer if not the last layer
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)
        
        # Global pooling: (batch_size, channels[-1], seq_len) -> (batch_size, channels[-1])
        if self.global_pooling == 'mean':
            x = torch.mean(x, dim=2)
        elif self.global_pooling == 'max':
            x, _ = torch.max(x, dim=2)
        elif self.global_pooling == 'attention':
            # Transpose for attention: (batch_size, seq_len, channels[-1])
            x = x.transpose(1, 2)
            
            # Compute attention weights
            attention_weights = self.attention_pooling(x)  # (batch_size, seq_len, 1)
            
            # Apply attention
            x = torch.sum(x * attention_weights, dim=1)  # (batch_size, channels[-1])
        else:
            raise ValueError(f"Unsupported global pooling: {self.global_pooling}")
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def get_temporal_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate temporal features from each dilated layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            List of feature tensors from each layer
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Input projection
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        
        features = []
        
        # Apply dilated convolution layers and collect features
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            
            # Global pool current features
            if self.global_pooling == 'mean':
                pooled = torch.mean(x, dim=2)
            elif self.global_pooling == 'max':
                pooled, _ = torch.max(x, dim=2)
            else:
                pooled = torch.mean(x, dim=2)  # Default to mean
            
            features.append(pooled)
            
            # Apply transition layer if not the last layer
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)
        
        return features
    
    def compute_receptive_field(self) -> int:
        """Compute the theoretical receptive field of the network."""
        receptive_field = 1
        
        for i, dilation in enumerate(self.dilation_rates):
            kernel_size = 3  # Fixed kernel size
            receptive_field = receptive_field + (kernel_size - 1) * dilation
        
        return receptive_field
    
    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'DilatedCNN',
            'input_dim': self.input_dim,
            'channels': self.channels,
            'dilation_rates': self.dilation_rates,
            'num_layers': len(self.conv_layers),
            'receptive_field': self.compute_receptive_field(),
            'global_pooling': self.global_pooling,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class MultiScaleDilatedCNN(nn.Module):
    """
    Multi-scale Dilated CNN that processes data at different temporal scales.
    """
    
    def __init__(self,
                 input_dim: int,
                 scales: List[int] = None,
                 channels_per_scale: List[int] = None,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 fusion_method: str = 'concatenate'):
        """
        Initialize Multi-scale Dilated CNN.
        
        Args:
            input_dim: Number of input features
            scales: List of different temporal scales (downsampling factors)
            channels_per_scale: Number of channels for each scale
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            fusion_method: Method to fuse multi-scale features ('concatenate', 'attention')
        """
        super(MultiScaleDilatedCNN, self).__init__()
        
        if scales is None:
            scales = [1, 2, 4]  # Different temporal downsampling scales
        
        if channels_per_scale is None:
            channels_per_scale = [128] * len(scales)
        
        if len(scales) != len(channels_per_scale):
            raise ValueError("Length of scales and channels_per_scale must match")
        
        self.scales = scales
        self.fusion_method = fusion_method
        
        # Create separate CNN for each scale
        self.scale_cnns = nn.ModuleList()
        
        for scale, channels in zip(scales, channels_per_scale):
            # Adjust dilation rates based on scale
            dilation_rates = [1 * scale, 2 * scale, 4 * scale]
            channel_list = [channels // 2, channels, channels]
            
            cnn = DilatedCNN(
                input_dim=input_dim,
                channels=channel_list,
                kernel_size=kernel_size,
                dilation_rates=dilation_rates,
                dropout=dropout,
                output_dim=channels
            )
            
            self.scale_cnns.append(cnn)
        
        # Fusion layer
        if fusion_method == 'concatenate':
            self.fusion_dim = sum(channels_per_scale)
        elif fusion_method == 'attention':
            self.fusion_dim = max(channels_per_scale)
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=8,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        logger.info(f"Initialized MultiScaleDilatedCNN with scales={scales}, "
                   f"fusion_method={fusion_method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Multi-scale Dilated CNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Fused multi-scale features
        """
        scale_features = []
        
        # Process at each scale
        for i, (scale, cnn) in enumerate(zip(self.scales, self.scale_cnns)):
            # Downsample input if scale > 1
            if scale > 1:
                # Simple downsampling by taking every scale-th element
                x_scaled = x[:, ::scale, :]
            else:
                x_scaled = x
            
            # Process through scale-specific CNN
            features = cnn(x_scaled)
            scale_features.append(features)
        
        # Fuse multi-scale features
        if self.fusion_method == 'concatenate':
            fused_features = torch.cat(scale_features, dim=1)
        elif self.fusion_method == 'attention':
            # Pad smaller features to same size
            max_dim = max(f.size(1) for f in scale_features)
            padded_features = []
            
            for features in scale_features:
                if features.size(1) < max_dim:
                    padding = torch.zeros(features.size(0), max_dim - features.size(1), 
                                        device=features.device)
                    features = torch.cat([features, padding], dim=1)
                padded_features.append(features.unsqueeze(0))
            
            # Stack and apply attention
            stacked_features = torch.cat(padded_features, dim=0)  # (num_scales, batch_size, dim)
            
            attended_features, _ = self.scale_attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # Average across scales
            fused_features = torch.mean(attended_features, dim=0)
        
        return fused_features