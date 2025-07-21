"""
Graph Neural Network (GNN) component for OCEAN model.
Models spatial relationships between services using Graph Attention Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Tuple, Dict, Any
import logging


logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """
    Custom Graph Attention Layer with enhanced features.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_heads: int = 1,
                 dropout: float = 0.1,
                 alpha: float = 0.2,
                 concat: bool = True,
                 bias: bool = True):
        """
        Initialize Graph Attention Layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate or average multi-head outputs
            bias: Whether to use bias
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformations
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features * num_heads)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features * num_heads))
        else:
            self.register_parameter('bias', None)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Graph Attention Layer.
        
        Args:
            h: Node features (batch_size * num_nodes, in_features)
            adj: Adjacency matrix (batch_size * num_nodes, batch_size * num_nodes)
            
        Returns:
            Updated node features
        """
        Wh = torch.mm(h, self.W)  # (N, out_features * num_heads)
        N = h.size(0)
        
        # Reshape for multi-head attention
        Wh = Wh.view(N, self.num_heads, self.out_features)
        
        # Compute attention coefficients
        attention_outputs = []
        
        for head in range(self.num_heads):
            Wh_head = Wh[:, head, :]  # (N, out_features)
            
            # Compute pairwise attention scores
            a_input = self._prepare_attentional_mechanism_input(Wh_head)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            
            # Mask attention scores using adjacency matrix
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = self.dropout_layer(attention)
            
            # Apply attention to features
            h_prime = torch.matmul(attention, Wh_head)
            attention_outputs.append(h_prime)
        
        # Combine multi-head outputs
        if self.concat:
            output = torch.cat(attention_outputs, dim=1)
        else:
            output = torch.mean(torch.stack(attention_outputs), dim=0)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """Prepare input for attention mechanism."""
        N = Wh.size(0)
        
        # Create all pairs of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Concatenate to form input for attention computation
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for spatial relationship modeling in service networks.
    Uses Graph Attention Networks to propagate information between services.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 conv_type: str = 'gat',
                 pooling: str = 'mean',
                 residual: bool = True,
                 batch_norm: bool = True):
        """
        Initialize Graph Neural Network.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout probability
            activation: Activation function
            conv_type: Type of graph convolution ('gat', 'gcn', 'sage', 'graph')
            pooling: Graph pooling method ('mean', 'max', 'add', 'attention')
            residual: Whether to use residual connections
            batch_norm: Whether to use batch normalization
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.conv_type = conv_type
        self.pooling = pooling
        self.residual = residual
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        for i in range(num_layers):
            if conv_type == 'gat':
                conv = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            elif conv_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == 'sage':
                conv = SAGEConv(hidden_dim, hidden_dim)
            elif conv_type == 'graph':
                conv = GraphConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            
            self.conv_layers.append(conv)
            
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Graph pooling
        if pooling == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        
        # Output projection to ensure correct dimensionality
        # For GAT with concat=False on final layer, output dim is hidden_dim // num_heads
        final_dim = hidden_dim // num_heads if conv_type == 'gat' else hidden_dim
        self.output_projection = nn.Linear(final_dim, hidden_dim)
        
        logger.info(f"Initialized GraphNeuralNetwork with {num_layers} {conv_type.upper()} layers, "
                   f"hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Graph Neural Network.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch vector (num_nodes,) for batched graphs
            
        Returns:
            Graph-level representations (batch_size, hidden_dim)
        """
        # Input projection
        x = self.input_projection(x)
        x = self.activation(x)
        
        # Apply graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x_residual = x if self.residual else None
            
            # Graph convolution
            x = conv(x, edge_index)
            
            # Batch normalization
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            x = self.dropout(x)
            
            # Residual connection
            if self.residual and x_residual is not None and x.shape == x_residual.shape:
                x = x + x_residual
        
        # Graph pooling
        if batch is None:
            # Single graph - global pooling
            if self.pooling == 'mean':
                x = torch.mean(x, dim=0, keepdim=True)
            elif self.pooling == 'max':
                x = torch.max(x, dim=0, keepdim=True)[0]
            elif self.pooling == 'add':
                x = torch.sum(x, dim=0, keepdim=True)
            elif self.pooling == 'attention':
                weights = self.attention_pooling(x)
                weights = F.softmax(weights, dim=0)
                x = torch.sum(x * weights, dim=0, keepdim=True)
        else:
            # Batched graphs
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                x = global_max_pool(x, batch)
            elif self.pooling == 'add':
                x = global_add_pool(x, batch)
            elif self.pooling == 'attention':
                # Custom attention pooling for batched graphs
                x = self._attention_pool_batched(x, batch)
        
        # Apply output projection
        x = self.output_projection(x)
        
        return x
    
    def _attention_pool_batched(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling to batched graphs."""
        unique_batch = torch.unique(batch)
        pooled_outputs = []
        
        for b in unique_batch:
            mask = batch == b
            x_batch = x[mask]
            
            weights = self.attention_pooling(x_batch)
            weights = F.softmax(weights, dim=0)
            pooled = torch.sum(x_batch * weights, dim=0, keepdim=True)
            pooled_outputs.append(pooled)
        
        return torch.cat(pooled_outputs, dim=0)
    
    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node-level embeddings without graph pooling.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Node embeddings (num_nodes, hidden_dim)
        """
        # Input projection
        x = self.input_projection(x)
        x = self.activation(x)
        
        # Apply graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x_residual = x if self.residual else None
            
            # Graph convolution
            x = conv(x, edge_index)
            
            # Batch normalization
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            x = self.dropout(x)
            
            # Residual connection
            if self.residual and x_residual is not None and x.shape == x_residual.shape:
                x = x + x_residual
        
        return x
    
    def compute_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute attention weights for each GAT layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            List of attention weight tensors
        """
        if self.conv_type != 'gat':
            raise ValueError("Attention weights only available for GAT layers")
        
        attention_weights = []
        x = self.input_projection(x)
        x = self.activation(x)
        
        for i, conv in enumerate(self.conv_layers):
            # Get attention weights from GAT layer
            if hasattr(conv, 'attention'):
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        return attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'GraphNeuralNetwork',
            'conv_type': self.conv_type,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'pooling': self.pooling,
            'residual': self.residual,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class HierarchicalGNN(nn.Module):
    """
    Hierarchical Graph Neural Network that processes graphs at multiple scales.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 num_layers_per_level: int = 2,
                 pooling_ratios: List[float] = None,
                 dropout: float = 0.1):
        """
        Initialize Hierarchical GNN.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dims: Hidden dimensions for each hierarchical level
            num_layers_per_level: Number of GNN layers per hierarchical level
            pooling_ratios: Pooling ratios for each level
            dropout: Dropout probability
        """
        super(HierarchicalGNN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        if pooling_ratios is None:
            pooling_ratios = [0.8, 0.6, 0.4]
        
        self.num_levels = len(hidden_dims)
        self.hidden_dims = hidden_dims
        
        # GNN layers for each hierarchical level
        self.gnn_levels = nn.ModuleList()
        
        current_input_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            gnn = GraphNeuralNetwork(
                input_dim=current_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers_per_level,
                dropout=dropout,
                pooling='mean'
            )
            self.gnn_levels.append(gnn)
            current_input_dim = hidden_dim
        
        # Final fusion layer
        self.fusion = nn.Linear(sum(hidden_dims), hidden_dims[-1])
        
        logger.info(f"Initialized HierarchicalGNN with {self.num_levels} levels, "
                   f"hidden_dims={hidden_dims}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Hierarchical GNN.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Hierarchical graph representation
        """
        level_outputs = []
        current_x = x
        current_edge_index = edge_index
        
        for i, gnn in enumerate(self.gnn_levels):
            # Process at current level
            output = gnn(current_x, current_edge_index)
            level_outputs.append(output)
            
            # For next level, we would typically apply graph pooling
            # For simplicity, we'll just use the current output as input to next level
            # In practice, you might implement graph coarsening here
            if i < len(self.gnn_levels) - 1:
                current_x = gnn.get_node_embeddings(current_x, current_edge_index)
        
        # Fuse representations from all levels
        fused = torch.cat(level_outputs, dim=-1)
        output = self.fusion(fused)
        
        return output