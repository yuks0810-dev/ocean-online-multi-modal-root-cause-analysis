"""
Graph Fusion Module with contrastive learning for OCEAN model.
Fuses multi-modal graph representations using contrastive mutual information maximization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging


logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features to a lower-dimensional space for contrastive comparison.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 projection_dim: int = 256, 
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2,
                 activation: str = 'relu',
                 dropout: float = 0.1):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input feature dimension
            projection_dim: Output projection dimension
            hidden_dim: Hidden layer dimension (defaults to input_dim)
            num_layers: Number of layers in projection head
            activation: Activation function
            dropout: Dropout probability
        """
        super(ProjectionHead, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            # Add linear layer
            if i == num_layers - 1:
                # Final layer
                layers.append(nn.Linear(current_dim, projection_dim))
            else:
                # Hidden layers
                layers.append(nn.Linear(current_dim, hidden_dim))
                
                # Add activation
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                
                # Add dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                
                current_dim = hidden_dim
        
        self.projection = nn.Sequential(*layers)
        
        # L2 normalization
        self.normalize = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply projection head.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Projected features (batch_size, projection_dim)
        """
        projected = self.projection(x)
        
        if self.normalize:
            projected = F.normalize(projected, p=2, dim=1)
        
        return projected


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.1, reduction: str = 'mean'):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for softmax
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor: Anchor representations (batch_size, dim)
            positive: Positive representations (batch_size, dim)
            negative: Optional negative representations (num_negatives, dim)
            
        Returns:
            InfoNCE loss
        """
        batch_size = anchor.size(0)
        
        # Normalize features
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute positive similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (batch_size,)
        
        if negative is not None:
            # Use provided negatives
            negative = F.normalize(negative, p=2, dim=1)
            neg_sim = torch.matmul(anchor, negative.t()) / self.temperature  # (batch_size, num_negatives)
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_negatives)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        else:
            # Use in-batch negatives
            # Compute all pairwise similarities
            all_sim = torch.matmul(anchor, positive.t()) / self.temperature  # (batch_size, batch_size)
            
            # Create mask for positive pairs (diagonal)
            pos_mask = torch.eye(batch_size, device=anchor.device).bool()
            
            # Extract positive similarities (diagonal)
            pos_sim = all_sim[pos_mask]  # (batch_size,)
            
            # Use all similarities as logits
            logits = all_sim
            labels = torch.arange(batch_size, device=anchor.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss


class ContrastiveLearningModule(nn.Module):
    """
    Contrastive learning module for multi-modal representation learning.
    """
    
    def __init__(self,
                 modality_dims: Dict[str, int],
                 projection_dim: int = 256,
                 temperature: float = 0.1,
                 num_negatives: int = 16):
        """
        Initialize contrastive learning module.
        
        Args:
            modality_dims: Dictionary of modality names and their dimensions
            projection_dim: Dimension of projection space
            temperature: Temperature for InfoNCE loss
            num_negatives: Number of negative samples per positive pair
        """
        super(ContrastiveLearningModule, self).__init__()
        
        self.modality_dims = modality_dims
        self.projection_dim = projection_dim
        self.num_negatives = num_negatives
        
        # Projection heads for each modality
        self.projection_heads = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.projection_heads[modality] = ProjectionHead(dim, projection_dim)
        
        # InfoNCE loss
        self.infonce_loss = InfoNCELoss(temperature)
        
        # Negative sample memory bank (for efficiency)
        self.register_buffer('memory_bank', torch.randn(num_negatives, projection_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        logger.info(f"Initialized ContrastiveLearningModule with modalities: {list(modality_dims.keys())}")
    
    def forward(self, 
                modality_features: Dict[str, torch.Tensor],
                return_projections: bool = False) -> Dict[str, torch.Tensor]:
        """
        Apply contrastive learning to multi-modal features.
        
        Args:
            modality_features: Dictionary of modality features
            return_projections: Whether to return projected features
            
        Returns:
            Dictionary containing contrastive losses and optionally projections
        """
        # Project each modality
        projections = {}
        for modality, features in modality_features.items():
            if modality in self.projection_heads:
                projections[modality] = self.projection_heads[modality](features)
        
        # Compute contrastive losses between modality pairs
        losses = {}
        modality_names = list(projections.keys())
        
        for i, mod1 in enumerate(modality_names):
            for j, mod2 in enumerate(modality_names):
                if i < j:  # Avoid duplicate pairs
                    loss_name = f"{mod1}_{mod2}_contrastive"
                    losses[loss_name] = self.infonce_loss(
                        projections[mod1], 
                        projections[mod2],
                        self._get_negatives()
                    )
        
        # Update memory bank
        if self.training:
            self._update_memory_bank(projections)
        
        results = {'losses': losses}
        if return_projections:
            results['projections'] = projections
        
        return results
    
    def _get_negatives(self) -> torch.Tensor:
        """Get negative samples from memory bank."""
        return self.memory_bank.clone().detach()
    
    def _update_memory_bank(self, projections: Dict[str, torch.Tensor]) -> None:
        """Update memory bank with current batch features."""
        if not projections:
            return
        
        # Combine all projections
        all_projections = torch.cat(list(projections.values()), dim=0)
        batch_size = all_projections.size(0)
        
        ptr = int(self.memory_ptr)
        
        # Replace oldest entries in memory bank
        if ptr + batch_size <= self.num_negatives:
            self.memory_bank[ptr:ptr + batch_size] = all_projections.detach()
            ptr = (ptr + batch_size) % self.num_negatives
        else:
            # Wrap around
            remaining = self.num_negatives - ptr
            self.memory_bank[ptr:] = all_projections[:remaining].detach()
            self.memory_bank[:batch_size - remaining] = all_projections[remaining:].detach()
            ptr = batch_size - remaining
        
        self.memory_ptr[0] = ptr


class GraphFusionModule(nn.Module):
    """
    Graph Fusion Module that fuses multi-modal graph representations 
    using contrastive mutual information maximization.
    """
    
    def __init__(self,
                 temporal_dim: int,
                 spatial_dim: int,
                 log_dim: int,
                 fusion_dim: int = 256,
                 projection_dim: int = 128,
                 temperature: float = 0.1,
                 fusion_strategy: str = 'attention',
                 dropout: float = 0.1):
        """
        Initialize Graph Fusion Module.
        
        Args:
            temporal_dim: Dimension of temporal features
            spatial_dim: Dimension of spatial features  
            log_dim: Dimension of log features
            fusion_dim: Dimension of fused representation
            projection_dim: Dimension for contrastive learning projections
            temperature: Temperature for contrastive learning
            fusion_strategy: Strategy for fusion ('concatenate', 'attention', 'gate')
            dropout: Dropout probability
        """
        super(GraphFusionModule, self).__init__()
        
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.log_dim = log_dim
        self.fusion_dim = fusion_dim
        self.fusion_strategy = fusion_strategy
        
        # Feature preprocessing
        self.temporal_processor = nn.Sequential(
            nn.Linear(temporal_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim)
        )
        
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim)
        )
        
        self.log_processor = nn.Sequential(
            nn.Linear(log_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim)
        )
        
        # Contrastive learning module
        modality_dims = {
            'temporal': fusion_dim,
            'spatial': fusion_dim,
            'log': fusion_dim
        }
        self.contrastive_module = ContrastiveLearningModule(
            modality_dims, projection_dim, temperature
        )
        
        # Fusion mechanism
        if fusion_strategy == 'concatenate':
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(fusion_dim)
            )
        elif fusion_strategy == 'attention':
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(fusion_dim)
        elif fusion_strategy == 'gate':
            self.gate_network = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.Sigmoid()
            )
            self.value_network = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.Tanh()
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Mutual information estimation (optional)
        self.mi_estimator = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1)
        )
        
        logger.info(f"Initialized GraphFusionModule with fusion_strategy={fusion_strategy}, "
                   f"fusion_dim={fusion_dim}")
    
    def forward(self, 
                temporal_features: torch.Tensor,
                spatial_features: torch.Tensor,
                log_features: torch.Tensor,
                compute_contrastive_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-modal graph representations with contrastive learning.
        
        Args:
            temporal_features: Temporal features (batch_size, temporal_dim)
            spatial_features: Spatial features (batch_size, spatial_dim)
            log_features: Log features (batch_size, log_dim)
            compute_contrastive_loss: Whether to compute contrastive losses
            
        Returns:
            Dictionary containing fused representation and losses
        """
        # Preprocess features
        temporal_processed = self.temporal_processor(temporal_features)
        spatial_processed = self.spatial_processor(spatial_features)
        log_processed = self.log_processor(log_features)
        
        # Contrastive learning
        contrastive_results = {}
        if compute_contrastive_loss:
            modality_features = {
                'temporal': temporal_processed,
                'spatial': spatial_processed,
                'log': log_processed
            }
            contrastive_results = self.contrastive_module(
                modality_features, return_projections=True
            )
        
        # Fusion
        if self.fusion_strategy == 'concatenate':
            concatenated = torch.cat([temporal_processed, spatial_processed, log_processed], dim=1)
            fused = self.fusion_layer(concatenated)
        
        elif self.fusion_strategy == 'attention':
            # Stack features for attention
            stacked_features = torch.stack([
                temporal_processed, spatial_processed, log_processed
            ], dim=1)  # (batch_size, 3, fusion_dim)
            
            # Apply self-attention
            attended, attention_weights = self.attention_fusion(
                stacked_features, stacked_features, stacked_features
            )
            
            # Aggregate attended features
            fused = torch.mean(attended, dim=1)  # (batch_size, fusion_dim)
            fused = self.attention_norm(fused)
        
        elif self.fusion_strategy == 'gate':
            concatenated = torch.cat([temporal_processed, spatial_processed, log_processed], dim=1)
            gate = self.gate_network(concatenated)
            value = self.value_network(concatenated)
            fused = gate * value
        
        # Output projection
        output = self.output_projection(fused)
        
        # Compute mutual information (optional)
        mi_scores = {}
        if compute_contrastive_loss:
            # MI between fused representation and each modality
            mi_scores['temporal_mi'] = self._compute_mutual_information(output, temporal_processed)
            mi_scores['spatial_mi'] = self._compute_mutual_information(output, spatial_processed)
            mi_scores['log_mi'] = self._compute_mutual_information(output, log_processed)
        
        results = {
            'fused_representation': output,
            'processed_features': {
                'temporal': temporal_processed,
                'spatial': spatial_processed,
                'log': log_processed
            }
        }
        
        if compute_contrastive_loss:
            results['contrastive_losses'] = contrastive_results.get('losses', {})
            results['projections'] = contrastive_results.get('projections', {})
            results['mutual_information'] = mi_scores
        
        return results
    
    def _compute_mutual_information(self, 
                                  representation1: torch.Tensor, 
                                  representation2: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information between two representations.
        
        Args:
            representation1: First representation
            representation2: Second representation
            
        Returns:
            Mutual information estimate
        """
        # Positive pairs (aligned)
        positive_pairs = torch.cat([representation1, representation2], dim=1)
        positive_scores = self.mi_estimator(positive_pairs)
        
        # Negative pairs (shuffled)
        batch_size = representation1.size(0)
        shuffled_indices = torch.randperm(batch_size, device=representation1.device)
        negative_pairs = torch.cat([representation1, representation2[shuffled_indices]], dim=1)
        negative_scores = self.mi_estimator(negative_pairs)
        
        # MI estimation using MINE (Mutual Information Neural Estimation)
        mi_estimate = positive_scores.mean() - torch.log(torch.exp(negative_scores).mean() + 1e-8)
        
        return mi_estimate
    
    def get_fusion_analysis(self, 
                           temporal_features: torch.Tensor,
                           spatial_features: torch.Tensor,
                           log_features: torch.Tensor) -> Dict[str, Any]:
        """
        Get analysis of fusion process for interpretability.
        
        Returns:
            Dictionary with fusion analysis
        """
        with torch.no_grad():
            results = self.forward(temporal_features, spatial_features, log_features)
            
            analysis = {
                'modality_contributions': {
                    'temporal': torch.norm(results['processed_features']['temporal'], dim=1).mean().item(),
                    'spatial': torch.norm(results['processed_features']['spatial'], dim=1).mean().item(),
                    'log': torch.norm(results['processed_features']['log'], dim=1).mean().item()
                },
                'fusion_quality': {
                    'representation_norm': torch.norm(results['fused_representation'], dim=1).mean().item()
                }
            }
            
            if 'mutual_information' in results:
                analysis['mutual_information'] = {
                    k: v.item() for k, v in results['mutual_information'].items()
                }
            
            return analysis
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'GraphFusionModule',
            'temporal_dim': self.temporal_dim,
            'spatial_dim': self.spatial_dim,
            'log_dim': self.log_dim,
            'fusion_dim': self.fusion_dim,
            'fusion_strategy': self.fusion_strategy,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class MultiModalGraphFusion(nn.Module):
    """
    Advanced multi-modal graph fusion with hierarchical contrastive learning.
    """
    
    def __init__(self,
                 modality_configs: List[Dict[str, Any]],
                 fusion_dim: int = 256,
                 num_fusion_layers: int = 2,
                 contrastive_levels: List[str] = None):
        """
        Initialize advanced multi-modal graph fusion.
        
        Args:
            modality_configs: List of modality configurations
            fusion_dim: Fusion dimension
            num_fusion_layers: Number of fusion layers
            contrastive_levels: Levels for contrastive learning
        """
        super(MultiModalGraphFusion, self).__init__()
        
        if contrastive_levels is None:
            contrastive_levels = ['feature', 'representation', 'global']
        
        self.modality_configs = modality_configs
        self.fusion_dim = fusion_dim
        self.contrastive_levels = contrastive_levels
        
        # Modality processors
        self.modality_processors = nn.ModuleList()
        for config in modality_configs:
            processor = nn.Sequential(
                nn.Linear(config['input_dim'], fusion_dim),
                nn.ReLU(),
                nn.LayerNorm(fusion_dim)
            )
            self.modality_processors.append(processor)
        
        # Hierarchical fusion layers
        self.fusion_layers = nn.ModuleList()
        for i in range(num_fusion_layers):
            fusion_layer = GraphFusionModule(
                temporal_dim=fusion_dim,
                spatial_dim=fusion_dim,
                log_dim=fusion_dim,
                fusion_dim=fusion_dim
            )
            self.fusion_layers.append(fusion_layer)
        
        # Multi-level contrastive learning
        self.contrastive_modules = nn.ModuleDict()
        for level in contrastive_levels:
            modality_dims = {f'modality_{i}': fusion_dim for i in range(len(modality_configs))}
            self.contrastive_modules[level] = ContrastiveLearningModule(modality_dims, 128)
        
        logger.info(f"Initialized MultiModalGraphFusion with {len(modality_configs)} modalities")
    
    def forward(self, modality_inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical multi-modal fusion.
        
        Args:
            modality_inputs: List of modality input tensors
            
        Returns:
            Fusion results
        """
        # Process modalities
        processed_modalities = []
        for i, (inputs, processor) in enumerate(zip(modality_inputs, self.modality_processors)):
            processed = processor(inputs)
            processed_modalities.append(processed)
        
        # Hierarchical fusion
        current_representations = processed_modalities
        fusion_outputs = []
        
        for fusion_layer in self.fusion_layers:
            # Apply fusion (assuming first 3 modalities)
            if len(current_representations) >= 3:
                fusion_result = fusion_layer(
                    current_representations[0],
                    current_representations[1], 
                    current_representations[2]
                )
                fused_repr = fusion_result['fused_representation']
                fusion_outputs.append(fused_repr)
                
                # Update representations for next layer
                current_representations = [fused_repr] + current_representations[3:]
        
        # Multi-level contrastive learning
        contrastive_losses = {}
        for level, contrastive_module in self.contrastive_modules.items():
            if level == 'feature':
                features = {f'modality_{i}': feat for i, feat in enumerate(processed_modalities)}
            elif level == 'representation':
                features = {f'fusion_{i}': repr for i, repr in enumerate(fusion_outputs)}
            elif level == 'global':
                features = {'global': fusion_outputs[-1]} if fusion_outputs else {}
            
            if features:
                contrastive_result = contrastive_module(features)
                contrastive_losses.update(contrastive_result['losses'])
        
        return {
            'final_representation': fusion_outputs[-1] if fusion_outputs else processed_modalities[0],
            'intermediate_representations': fusion_outputs,
            'processed_modalities': processed_modalities,
            'contrastive_losses': contrastive_losses
        }