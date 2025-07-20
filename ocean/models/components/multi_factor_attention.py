"""
Multi-factor Attention mechanism for OCEAN model.
Dynamically weights different metrics and log features for multi-modal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Any
import logging


logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initialize Scaled Dot-Product Attention.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply scaled dot-product attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_values, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multiple heads if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Apply attention to each head
        attended_values, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.w_o(attended_values)
        output = self.dropout(output)
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between different data modalities.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        """
        Initialize Cross-Modal Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossModalAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.ff_layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                query_modality: torch.Tensor, 
                key_value_modality: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            query_modality: Query modality features
            key_value_modality: Key-value modality features
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Multi-head attention with residual connection
        attended, attention_weights = self.multihead_attention(
            query_modality, key_value_modality, key_value_modality, mask
        )
        attended = self.layer_norm(query_modality + attended)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(attended)
        output = self.ff_layer_norm(attended + ff_output)
        
        return output, attention_weights


class MultiFactorAttention(nn.Module):
    """
    Multi-factor Attention mechanism for OCEAN model.
    Dynamically weights different metrics and log features for multi-modal fusion.
    """
    
    def __init__(self,
                 temporal_dim: int,
                 spatial_dim: int,
                 log_dim: int,
                 attention_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 fusion_strategy: str = 'concatenate'):
        """
        Initialize Multi-factor Attention.
        
        Args:
            temporal_dim: Dimension of temporal features (from DCNN)
            spatial_dim: Dimension of spatial features (from GNN)
            log_dim: Dimension of log features
            attention_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
            fusion_strategy: Strategy for fusing modalities ('concatenate', 'add', 'gate')
        """
        super(MultiFactorAttention, self).__init__()
        
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.log_dim = log_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.fusion_strategy = fusion_strategy
        
        # Project different modalities to same dimension
        self.temporal_projection = nn.Linear(temporal_dim, attention_dim)
        self.spatial_projection = nn.Linear(spatial_dim, attention_dim)
        self.log_projection = nn.Linear(log_dim, attention_dim)
        
        # Self-attention for each modality
        self.temporal_self_attention = MultiHeadAttention(attention_dim, num_heads, dropout)
        self.spatial_self_attention = MultiHeadAttention(attention_dim, num_heads, dropout)
        self.log_self_attention = MultiHeadAttention(attention_dim, num_heads, dropout)
        
        # Cross-modal attention
        self.temporal_spatial_attention = CrossModalAttention(attention_dim, num_heads, dropout)
        self.temporal_log_attention = CrossModalAttention(attention_dim, num_heads, dropout)
        self.spatial_log_attention = CrossModalAttention(attention_dim, num_heads, dropout)
        
        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(3))  # temporal, spatial, log
        
        # Feature importance attention
        self.feature_attention = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Linear(attention_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Fusion layers
        if fusion_strategy == 'concatenate':
            self.fusion_dim = attention_dim * 3
            self.fusion_layer = nn.Linear(self.fusion_dim, attention_dim)
        elif fusion_strategy == 'add':
            self.fusion_dim = attention_dim
            self.fusion_layer = nn.Identity()
        elif fusion_strategy == 'gate':
            self.fusion_dim = attention_dim
            self.gate_layer = nn.Sequential(
                nn.Linear(attention_dim * 3, attention_dim),
                nn.Sigmoid()
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, attention_dim)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            'temporal': nn.LayerNorm(attention_dim),
            'spatial': nn.LayerNorm(attention_dim),
            'log': nn.LayerNorm(attention_dim),
            'output': nn.LayerNorm(attention_dim)
        })
        
        logger.info(f"Initialized MultiFactorAttention with dims: temporal={temporal_dim}, "
                   f"spatial={spatial_dim}, log={log_dim}, attention={attention_dim}")
    
    def forward(self, 
                temporal_features: torch.Tensor,
                spatial_features: torch.Tensor,
                log_features: torch.Tensor,
                return_attention_weights: bool = False) -> torch.Tensor:
        """
        Apply multi-factor attention to fuse multi-modal features.
        
        Args:
            temporal_features: Temporal features from DCNN (batch_size, temporal_dim)
            spatial_features: Spatial features from GNN (batch_size, spatial_dim)
            log_features: Log features (batch_size, log_dim)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Fused multi-modal features (batch_size, attention_dim)
        """
        batch_size = temporal_features.size(0)
        
        # Project to common dimension
        temporal_proj = self.temporal_projection(temporal_features)  # (batch_size, attention_dim)
        spatial_proj = self.spatial_projection(spatial_features)     # (batch_size, attention_dim)
        log_proj = self.log_projection(log_features)                 # (batch_size, attention_dim)
        
        # Add sequence dimension for attention
        temporal_proj = temporal_proj.unsqueeze(1)  # (batch_size, 1, attention_dim)
        spatial_proj = spatial_proj.unsqueeze(1)    # (batch_size, 1, attention_dim)
        log_proj = log_proj.unsqueeze(1)            # (batch_size, 1, attention_dim)
        
        # Self-attention for each modality
        temporal_self, temporal_attn = self.temporal_self_attention(
            temporal_proj, temporal_proj, temporal_proj
        )
        spatial_self, spatial_attn = self.spatial_self_attention(
            spatial_proj, spatial_proj, spatial_proj
        )
        log_self, log_attn = self.log_self_attention(
            log_proj, log_proj, log_proj
        )
        
        # Apply layer normalization
        temporal_self = self.layer_norms['temporal'](temporal_self.squeeze(1))
        spatial_self = self.layer_norms['spatial'](spatial_self.squeeze(1))
        log_self = self.layer_norms['log'](log_self.squeeze(1))
        
        # Cross-modal attention
        # Temporal attending to spatial and log
        temporal_enhanced, _ = self._apply_cross_modal_attention(
            temporal_self, spatial_self, log_self
        )
        
        # Spatial attending to temporal and log
        spatial_enhanced, _ = self._apply_cross_modal_attention(
            spatial_self, temporal_self, log_self
        )
        
        # Log attending to temporal and spatial
        log_enhanced, _ = self._apply_cross_modal_attention(
            log_self, temporal_self, spatial_self
        )
        
        # Apply modality importance weights
        modality_weights = F.softmax(self.modality_weights, dim=0)
        temporal_weighted = temporal_enhanced * modality_weights[0]
        spatial_weighted = spatial_enhanced * modality_weights[1]
        log_weighted = log_enhanced * modality_weights[2]
        
        # Feature-level attention
        temporal_importance = self.feature_attention(temporal_weighted)
        spatial_importance = self.feature_attention(spatial_weighted)
        log_importance = self.feature_attention(log_weighted)
        
        temporal_attended = temporal_weighted * temporal_importance
        spatial_attended = spatial_weighted * spatial_importance
        log_attended = log_weighted * log_importance
        
        # Fusion
        if self.fusion_strategy == 'concatenate':
            fused = torch.cat([temporal_attended, spatial_attended, log_attended], dim=1)
            fused = self.fusion_layer(fused)
        elif self.fusion_strategy == 'add':
            fused = temporal_attended + spatial_attended + log_attended
        elif self.fusion_strategy == 'gate':
            concatenated = torch.cat([temporal_attended, spatial_attended, log_attended], dim=1)
            gate = self.gate_layer(concatenated)
            fused = gate * (temporal_attended + spatial_attended + log_attended)
        
        # Output projection and normalization
        output = self.output_projection(fused)
        output = self.layer_norms['output'](output)
        
        if return_attention_weights:
            attention_info = {
                'modality_weights': modality_weights,
                'feature_importance': {
                    'temporal': temporal_importance,
                    'spatial': spatial_importance,
                    'log': log_importance
                },
                'self_attention': {
                    'temporal': temporal_attn,
                    'spatial': spatial_attn,
                    'log': log_attn
                }
            }
            return output, attention_info
        
        return output
    
    def _apply_cross_modal_attention(self, 
                                   query_modality: torch.Tensor,
                                   key_modality1: torch.Tensor,
                                   key_modality2: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Apply cross-modal attention between modalities."""
        # Combine key modalities
        combined_keys = torch.stack([key_modality1, key_modality2], dim=1)  # (batch_size, 2, attention_dim)
        query_expanded = query_modality.unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        # Attention scores
        scores = torch.matmul(query_expanded, combined_keys.transpose(-2, -1))  # (batch_size, 1, 2)
        attention_weights = F.softmax(scores / math.sqrt(self.attention_dim), dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, combined_keys).squeeze(1)  # (batch_size, attention_dim)
        
        # Combine with original query
        enhanced = query_modality + attended
        
        return enhanced, {'attention_weights': attention_weights}
    
    def get_attention_analysis(self, 
                              temporal_features: torch.Tensor,
                              spatial_features: torch.Tensor,
                              log_features: torch.Tensor) -> Dict[str, Any]:
        """
        Get detailed attention analysis for interpretability.
        
        Returns:
            Dictionary with attention weights and importance scores
        """
        with torch.no_grad():
            output, attention_info = self.forward(
                temporal_features, spatial_features, log_features, 
                return_attention_weights=True
            )
            
            analysis = {
                'modality_importance': {
                    'temporal': attention_info['modality_weights'][0].item(),
                    'spatial': attention_info['modality_weights'][1].item(),
                    'log': attention_info['modality_weights'][2].item()
                },
                'feature_importance': {
                    'temporal': attention_info['feature_importance']['temporal'].mean().item(),
                    'spatial': attention_info['feature_importance']['spatial'].mean().item(),
                    'log': attention_info['feature_importance']['log'].mean().item()
                }
            }
            
            return analysis
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'MultiFactorAttention',
            'temporal_dim': self.temporal_dim,
            'spatial_dim': self.spatial_dim,
            'log_dim': self.log_dim,
            'attention_dim': self.attention_dim,
            'num_heads': self.num_heads,
            'fusion_strategy': self.fusion_strategy,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that learns to focus on important time steps and features.
    """
    
    def __init__(self, 
                 feature_dim: int, 
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize Adaptive Attention.
        
        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(AdaptiveAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Temporal attention
        self.temporal_attention = MultiHeadAttention(feature_dim, num_heads, dropout)
        
        # Feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Adaptive weighting
        self.adaptive_weights = nn.Parameter(torch.ones(2))  # temporal, feature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive attention.
        
        Args:
            x: Input features (batch_size, seq_len, feature_dim)
            
        Returns:
            Attended features
        """
        # Temporal attention
        temporal_attended, _ = self.temporal_attention(x, x, x)
        
        # Feature attention
        feature_weights = self.feature_attention(x)
        feature_attended = x * feature_weights
        
        # Adaptive combination
        weights = F.softmax(self.adaptive_weights, dim=0)
        output = weights[0] * temporal_attended + weights[1] * feature_attended
        
        return output


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for multi-scale feature processing.
    """
    
    def __init__(self, 
                 feature_dims: List[int], 
                 attention_dim: int = 128,
                 num_heads: int = 8):
        """
        Initialize Hierarchical Attention.
        
        Args:
            feature_dims: List of feature dimensions at different scales
            attention_dim: Hidden attention dimension
            num_heads: Number of attention heads
        """
        super(HierarchicalAttention, self).__init__()
        
        self.feature_dims = feature_dims
        self.attention_dim = attention_dim
        
        # Projection layers for different scales
        self.projections = nn.ModuleList([
            nn.Linear(dim, attention_dim) for dim in feature_dims
        ])
        
        # Multi-head attention for each scale
        self.scale_attentions = nn.ModuleList([
            MultiHeadAttention(attention_dim, num_heads) for _ in feature_dims
        ])
        
        # Cross-scale attention
        self.cross_scale_attention = MultiHeadAttention(attention_dim, num_heads)
        
        # Output fusion
        self.fusion = nn.Linear(attention_dim * len(feature_dims), attention_dim)
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply hierarchical attention to multi-scale features.
        
        Args:
            multi_scale_features: List of feature tensors at different scales
            
        Returns:
            Fused hierarchical features
        """
        # Project features to common dimension
        projected_features = []
        for features, projection in zip(multi_scale_features, self.projections):
            projected = projection(features)
            if projected.dim() == 2:
                projected = projected.unsqueeze(1)
            projected_features.append(projected)
        
        # Apply scale-specific attention
        scale_attended = []
        for features, attention in zip(projected_features, self.scale_attentions):
            attended, _ = attention(features, features, features)
            scale_attended.append(attended)
        
        # Cross-scale attention
        if len(scale_attended) > 1:
            # Combine scales for cross-attention
            combined_scales = torch.cat(scale_attended, dim=1)
            cross_attended, _ = self.cross_scale_attention(
                combined_scales, combined_scales, combined_scales
            )
            
            # Split back to individual scales
            split_sizes = [feat.size(1) for feat in scale_attended]
            cross_attended_split = torch.split(cross_attended, split_sizes, dim=1)
            
            # Update scale features
            for i, cross_feat in enumerate(cross_attended_split):
                scale_attended[i] = scale_attended[i] + cross_feat
        
        # Fusion
        concatenated = torch.cat([feat.mean(dim=1) for feat in scale_attended], dim=1)
        output = self.fusion(concatenated)
        
        return output