"""
Unified OCEAN (Online Multi-modal Causal structure lEArNiNG) model.
Integrates all components for end-to-end root cause analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from .components import (
    DilatedCNN, GraphNeuralNetwork, MultiFactorAttention, GraphFusionModule
)
from ..data.data_types import ServiceGraph, MultiModalFeatures
from ..configs import OCEANConfig


logger = logging.getLogger(__name__)


class OCEANModel(nn.Module):
    """
    Unified OCEAN model that combines all components for multi-modal root cause analysis.
    Processes metrics (temporal), service graphs (spatial), and logs for fault diagnosis.
    """
    
    def __init__(self, config: OCEANConfig):
        """
        Initialize OCEAN model.
        
        Args:
            config: OCEAN configuration object
        """
        super(OCEANModel, self).__init__()
        
        self.config = config
        
        # Model dimensions
        self.metrics_input_dim = getattr(config.data, 'metrics_features', 12)  # Default metrics features
        self.log_input_dim = getattr(config.data, 'log_embedding_dim', 768)   # BERT embedding dimension
        self.graph_input_dim = getattr(config.data, 'graph_features', 12)     # Service graph features
        
        # Component dimensions
        self.temporal_dim = config.model.hidden_dim
        self.spatial_dim = config.model.hidden_dim
        self.attention_dim = config.model.attention_dim
        self.fusion_dim = config.model.fusion_projection_dim
        
        # Initialize components
        self._init_temporal_component()
        self._init_spatial_component()
        self._init_attention_component()
        self._init_fusion_component()
        self._init_prediction_head()
        
        # Loss weights
        self.prediction_loss_weight = config.training.prediction_loss_weight
        self.contrastive_loss_weight = config.training.contrastive_loss_weight
        
        logger.info(f"Initialized OCEAN model with temporal_dim={self.temporal_dim}, "
                   f"spatial_dim={self.spatial_dim}, attention_dim={self.attention_dim}")
    
    def _init_temporal_component(self):
        """Initialize Dilated CNN for temporal feature extraction."""
        self.temporal_encoder = DilatedCNN(
            input_dim=self.metrics_input_dim,
            channels=self.config.model.dcnn_channels,
            kernel_size=self.config.model.dcnn_kernel_size,
            dilation_rates=self.config.model.dcnn_dilation_rates,
            dropout=self.config.model.dcnn_dropout,
            output_dim=self.temporal_dim
        )
    
    def _init_spatial_component(self):
        """Initialize Graph Neural Network for spatial relationship modeling."""
        self.spatial_encoder = GraphNeuralNetwork(
            input_dim=self.graph_input_dim,
            hidden_dim=self.spatial_dim,
            num_layers=self.config.model.gnn_num_layers,
            num_heads=self.config.model.gnn_num_heads,
            dropout=self.config.model.gnn_dropout,
            conv_type='gat',  # Use Graph Attention Networks
            pooling='mean',  # Use simple mean pooling instead of attention
            batch_norm=False  # Disable batch normalization for demo
        )
    
    def _init_attention_component(self):
        """Initialize Multi-factor Attention mechanism."""
        self.attention_module = MultiFactorAttention(
            temporal_dim=self.temporal_dim,
            spatial_dim=self.spatial_dim,
            log_dim=self.log_input_dim,
            attention_dim=self.attention_dim,
            num_heads=self.config.model.attention_num_heads,
            dropout=self.config.model.attention_dropout,
            fusion_strategy='concatenate'
        )
    
    def _init_fusion_component(self):
        """Initialize Graph Fusion Module with contrastive learning."""
        self.fusion_module = GraphFusionModule(
            temporal_dim=self.attention_dim,  # Output from attention
            spatial_dim=self.attention_dim,
            log_dim=self.attention_dim,
            fusion_dim=self.fusion_dim,
            projection_dim=self.fusion_dim // 2,
            temperature=self.config.model.fusion_temperature,
            fusion_strategy='attention'
        )
    
    def _init_prediction_head(self):
        """Initialize prediction head for root cause scoring."""
        self.prediction_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim // 4, self.config.model.num_services)
        )
        
        # Output activation for root cause probabilities
        self.output_activation = nn.Sigmoid()
    
    def forward(self, 
                metrics: torch.Tensor,
                service_graph: ServiceGraph,
                logs: torch.Tensor,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through OCEAN model.
        
        Args:
            metrics: Metrics time series (batch_size, seq_len, metrics_dim)
            service_graph: Service dependency graph
            logs: Log embeddings (batch_size, log_dim)
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            Dictionary containing predictions and optionally intermediate results
        """
        batch_size = metrics.size(0)
        
        # 1. Temporal feature extraction (DCNN)
        temporal_features = self.temporal_encoder(metrics)  # (batch_size, temporal_dim)
        
        # 2. Spatial feature extraction (GNN)
        # Prepare graph data
        node_features = service_graph.node_features
        edge_index = service_graph.to_edge_index()
        
        # Handle batching for graph neural network
        if batch_size > 1:
            # Create batch vector for graph pooling
            batch_vector = torch.arange(batch_size, device=metrics.device).repeat_interleave(
                node_features.size(0)
            )
            # Repeat node features for each sample in batch
            batched_node_features = node_features.repeat(batch_size, 1)
            # Adjust edge indices for batched graphs
            num_nodes = node_features.size(0)
            batched_edge_index = edge_index.clone()
            for i in range(1, batch_size):
                batch_edges = edge_index + i * num_nodes
                batched_edge_index = torch.cat([batched_edge_index, batch_edges], dim=1)
        else:
            batched_node_features = node_features
            batched_edge_index = edge_index
            batch_vector = None
        
        spatial_features = self.spatial_encoder(
            batched_node_features, batched_edge_index, batch_vector
        )  # (batch_size, spatial_dim)
        
        # 3. Log feature aggregation if needed
        if logs is not None:
            if logs.dim() == 3:
                # Aggregate time series log features to batch level
                log_features = torch.mean(logs, dim=1)  # (batch_size, log_dim)
            else:
                log_features = logs
        else:
            # Create dummy log features if None
            log_features = torch.zeros(batch_size, self.log_input_dim, device=metrics.device)
        
        # 3. Multi-factor Attention
        attended_features = self.attention_module(
            temporal_features, spatial_features, log_features
        )  # (batch_size, attention_dim)
        
        # 4. Graph Fusion with Contrastive Learning
        # Use attended features instead of raw features
        fusion_results = self.fusion_module(
            attended_features,  # Already fused temporal+spatial+log features
            spatial_features,   # Spatial features
            attended_features,  # Use attended features as log input too
            compute_contrastive_loss=self.training
        )
        
        fused_representation = fusion_results['fused_representation']  # (batch_size, fusion_dim)
        
        # 5. Root Cause Prediction
        root_cause_logits = self.prediction_head(fused_representation)  # (batch_size, num_services)
        root_cause_probs = self.output_activation(root_cause_logits)
        
        # Prepare output
        outputs = {
            'root_cause_logits': root_cause_logits,
            'root_cause_probs': root_cause_probs,
            'prediction': root_cause_probs  # Main prediction
        }
        
        # Add contrastive losses if training
        if self.training and 'contrastive_losses' in fusion_results:
            outputs['contrastive_losses'] = fusion_results['contrastive_losses']
        
        # Add intermediate representations if requested
        if return_intermediate:
            outputs['intermediate'] = {
                'temporal_features': temporal_features,
                'spatial_features': spatial_features,
                'attended_features': attended_features,
                'fused_representation': fused_representation,
                'fusion_details': fusion_results
            }
        
        return outputs
    
    def compute_loss(self, 
                     predictions: Dict[str, torch.Tensor],
                     targets: torch.Tensor,
                     reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining prediction and contrastive losses.
        
        Args:
            predictions: Model predictions dictionary
            targets: Target root cause labels (batch_size, num_services)
            reduction: Loss reduction method
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Prediction loss (Binary Cross Entropy)
        pred_loss = F.binary_cross_entropy(
            predictions['root_cause_probs'], 
            targets.float(), 
            reduction=reduction
        )
        losses['prediction_loss'] = pred_loss
        
        # Contrastive losses (if available)
        total_contrastive_loss = 0.0
        if 'contrastive_losses' in predictions:
            for loss_name, loss_value in predictions['contrastive_losses'].items():
                losses[f'contrastive_{loss_name}'] = loss_value
                total_contrastive_loss += loss_value
        
        losses['contrastive_loss'] = total_contrastive_loss
        
        # Total weighted loss
        total_loss = (self.prediction_loss_weight * pred_loss + 
                     self.contrastive_loss_weight * total_contrastive_loss)
        losses['total_loss'] = total_loss
        
        return losses
    
    def predict_root_cause(self, 
                          metrics: torch.Tensor,
                          service_graph: ServiceGraph,
                          logs: torch.Tensor,
                          threshold: float = 0.5,
                          top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict root cause services with post-processing.
        
        Args:
            metrics: Metrics time series
            service_graph: Service dependency graph
            logs: Log embeddings
            threshold: Probability threshold for binary classification
            top_k: Return top-k most likely root causes
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(metrics, service_graph, logs)
            probs = outputs['root_cause_probs']
            
            # Binary predictions using threshold
            binary_preds = (probs > threshold).float()
            
            # Top-k predictions
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
            else:
                top_k_probs, top_k_indices = None, None
            
            results = {
                'probabilities': probs,
                'binary_predictions': binary_preds,
                'threshold': threshold
            }
            
            if top_k is not None:
                results.update({
                    'top_k_probabilities': top_k_probs,
                    'top_k_indices': top_k_indices,
                    'top_k': top_k
                })
            
            # Add service names if available
            if service_graph.service_names:
                results['service_names'] = service_graph.service_names
                
                if top_k is not None:
                    top_k_services = []
                    for batch_idx in range(top_k_indices.size(0)):
                        batch_services = [
                            service_graph.service_names[idx.item()] 
                            for idx in top_k_indices[batch_idx]
                        ]
                        top_k_services.append(batch_services)
                    results['top_k_services'] = top_k_services
            
            return results
    
    def get_attention_weights(self, 
                             metrics: torch.Tensor,
                             service_graph: ServiceGraph,
                             logs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Args:
            metrics: Metrics time series
            service_graph: Service dependency graph  
            logs: Log embeddings
            
        Returns:
            Dictionary containing attention weights
        """
        self.eval()
        with torch.no_grad():
            # Get intermediate representations
            outputs = self.forward(metrics, service_graph, logs, return_intermediate=True)
            
            # Get attention analysis from multi-factor attention
            attention_analysis = self.attention_module.get_attention_analysis(
                outputs['intermediate']['temporal_features'],
                outputs['intermediate']['spatial_features'],
                logs
            )
            
            # Get fusion analysis
            fusion_analysis = self.fusion_module.get_fusion_analysis(
                outputs['intermediate']['attended_features'],
                outputs['intermediate']['spatial_features'],
                logs
            )
            
            return {
                'attention_analysis': attention_analysis,
                'fusion_analysis': fusion_analysis,
                'temporal_features_norm': torch.norm(outputs['intermediate']['temporal_features'], dim=1),
                'spatial_features_norm': torch.norm(outputs['intermediate']['spatial_features'], dim=1),
                'fused_features_norm': torch.norm(outputs['intermediate']['fused_representation'], dim=1)
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Component information
        component_info = {
            'temporal_encoder': self.temporal_encoder.get_model_info(),
            'spatial_encoder': self.spatial_encoder.get_model_info(),
            'attention_module': self.attention_module.get_model_info(),
            'fusion_module': self.fusion_module.get_model_info()
        }
        
        # Model configuration
        model_config = {
            'temporal_dim': self.temporal_dim,
            'spatial_dim': self.spatial_dim,
            'attention_dim': self.attention_dim,
            'fusion_dim': self.fusion_dim,
            'num_services': self.config.model.num_services,
            'prediction_loss_weight': self.prediction_loss_weight,
            'contrastive_loss_weight': self.contrastive_loss_weight
        }
        
        return {
            'model_type': 'OCEAN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_efficiency': trainable_params / total_params,
            'model_config': model_config,
            'component_info': component_info
        }
    
    def freeze_component(self, component_name: str):
        """Freeze parameters of a specific component."""
        if component_name == 'temporal':
            for param in self.temporal_encoder.parameters():
                param.requires_grad = False
        elif component_name == 'spatial':
            for param in self.spatial_encoder.parameters():
                param.requires_grad = False
        elif component_name == 'attention':
            for param in self.attention_module.parameters():
                param.requires_grad = False
        elif component_name == 'fusion':
            for param in self.fusion_module.parameters():
                param.requires_grad = False
        elif component_name == 'prediction':
            for param in self.prediction_head.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown component: {component_name}")
        
        logger.info(f"Frozen component: {component_name}")
    
    def unfreeze_component(self, component_name: str):
        """Unfreeze parameters of a specific component."""
        if component_name == 'temporal':
            for param in self.temporal_encoder.parameters():
                param.requires_grad = True
        elif component_name == 'spatial':
            for param in self.spatial_encoder.parameters():
                param.requires_grad = True
        elif component_name == 'attention':
            for param in self.attention_module.parameters():
                param.requires_grad = True
        elif component_name == 'fusion':
            for param in self.fusion_module.parameters():
                param.requires_grad = True
        elif component_name == 'prediction':
            for param in self.prediction_head.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown component: {component_name}")
        
        logger.info(f"Unfrozen component: {component_name}")


class OCEANVariant(OCEANModel):
    """
    OCEAN model variant for ablation studies.
    Allows disabling specific components to study their contributions.
    """
    
    def __init__(self, 
                 config: OCEANConfig, 
                 disabled_components: List[str] = None):
        """
        Initialize OCEAN variant.
        
        Args:
            config: OCEAN configuration
            disabled_components: List of components to disable ('dcnn', 'gnn', 'attention', 'fusion')
        """
        super().__init__(config)
        
        self.disabled_components = disabled_components or []
        
        # Disable components as specified
        for component in self.disabled_components:
            if component == 'dcnn':
                # Replace DCNN with simple linear layer
                self.temporal_encoder = nn.Linear(self.metrics_input_dim, self.temporal_dim)
            elif component == 'gnn':
                # Replace GNN with simple aggregation
                self.spatial_encoder = nn.Sequential(
                    nn.Linear(self.graph_input_dim, self.spatial_dim),
                    nn.ReLU()
                )
            elif component == 'attention':
                # Replace attention with simple concatenation
                self.attention_module = nn.Linear(
                    self.temporal_dim + self.spatial_dim + self.log_input_dim,
                    self.attention_dim
                )
            elif component == 'fusion':
                # Replace fusion with simple linear combination
                self.fusion_module = nn.Linear(self.attention_dim * 3, self.fusion_dim)
        
        logger.info(f"Initialized OCEAN variant with disabled components: {self.disabled_components}")
    
    def forward(self, 
                metrics: torch.Tensor,
                service_graph: ServiceGraph,
                logs: torch.Tensor,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with component modifications for ablation."""
        
        if 'dcnn' in self.disabled_components:
            # Simple temporal processing
            temporal_features = self.temporal_encoder(metrics.mean(dim=1))
        else:
            temporal_features = self.temporal_encoder(metrics)
        
        if 'gnn' in self.disabled_components:
            # Simple spatial processing
            spatial_features = self.spatial_encoder(service_graph.node_features.mean(dim=0).repeat(metrics.size(0), 1))
        else:
            # Use original GNN processing (simplified for ablation)
            node_features = service_graph.node_features
            spatial_features = self.spatial_encoder(node_features, service_graph.to_edge_index())
            if spatial_features.size(0) != metrics.size(0):
                spatial_features = spatial_features.mean(dim=0).repeat(metrics.size(0), 1)
        
        if 'attention' in self.disabled_components:
            # Simple concatenation
            attended_features = self.attention_module(
                torch.cat([temporal_features, spatial_features, logs], dim=1)
            )
        else:
            attended_features = self.attention_module(temporal_features, spatial_features, logs)
        
        if 'fusion' in self.disabled_components:
            # Simple linear fusion
            fused_representation = self.fusion_module(
                torch.cat([attended_features, spatial_features, logs], dim=1)
            )
            fusion_results = {'fused_representation': fused_representation}
        else:
            fusion_results = self.fusion_module(attended_features, spatial_features, logs, compute_contrastive_loss=self.training)
            fused_representation = fusion_results['fused_representation']
        
        # Prediction (unchanged)
        root_cause_logits = self.prediction_head(fused_representation)
        root_cause_probs = self.output_activation(root_cause_logits)
        
        outputs = {
            'root_cause_logits': root_cause_logits,
            'root_cause_probs': root_cause_probs,
            'prediction': root_cause_probs
        }
        
        if self.training and 'contrastive_losses' in fusion_results:
            outputs['contrastive_losses'] = fusion_results['contrastive_losses']
        
        if return_intermediate:
            outputs['intermediate'] = {
                'temporal_features': temporal_features,
                'spatial_features': spatial_features,
                'attended_features': attended_features,
                'fused_representation': fused_representation
            }
        
        return outputs