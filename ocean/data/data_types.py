"""
Data types and schemas for OCEAN implementation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import pandas as pd
import torch


@dataclass
class MetricsData:
    """Data structure for metrics time series."""
    
    timestamp: List[datetime]
    service_id: List[str]
    cpu_usage: List[float]
    memory_usage: List[float]
    response_time: List[float]
    error_rate: List[float]
    request_count: Optional[List[float]] = None
    disk_usage: Optional[List[float]] = None
    network_io: Optional[List[float]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'timestamp': self.timestamp,
            'service_id': self.service_id,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'response_time': self.response_time,
            'error_rate': self.error_rate,
        }
        
        # Add optional fields if available
        if self.request_count is not None:
            data['request_count'] = self.request_count
        if self.disk_usage is not None:
            data['disk_usage'] = self.disk_usage
        if self.network_io is not None:
            data['network_io'] = self.network_io
            
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MetricsData":
        """Create from pandas DataFrame."""
        required_cols = ['timestamp', 'service_id', 'cpu_usage', 'memory_usage', 
                        'response_time', 'error_rate']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        kwargs = {col: df[col].tolist() for col in required_cols}
        
        # Add optional columns if present
        optional_cols = ['request_count', 'disk_usage', 'network_io']
        for col in optional_cols:
            if col in df.columns:
                kwargs[col] = df[col].tolist()
        
        return cls(**kwargs)


@dataclass
class LogData:
    """Data structure for log entries."""
    
    timestamp: List[datetime]
    service_id: List[str]
    log_level: List[str]
    log_message: List[str]
    log_template: Optional[List[str]] = None
    log_embedding: Optional[List[List[float]]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'timestamp': self.timestamp,
            'service_id': self.service_id,
            'log_level': self.log_level,
            'log_message': self.log_message,
        }
        
        if self.log_template is not None:
            data['log_template'] = self.log_template
            
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "LogData":
        """Create from pandas DataFrame."""
        required_cols = ['timestamp', 'service_id', 'log_level', 'log_message']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        kwargs = {col: df[col].tolist() for col in required_cols}
        
        # Add optional columns if present
        if 'log_template' in df.columns:
            kwargs['log_template'] = df['log_template'].tolist()
        
        return cls(**kwargs)


@dataclass
class TraceData:
    """Data structure for distributed tracing data."""
    
    trace_id: List[str]
    span_id: List[str]
    parent_span_id: List[Optional[str]]
    service_name: List[str]
    operation_name: List[str]
    start_time: List[datetime]
    duration: List[float]
    status_code: Optional[List[str]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'service_name': self.service_name,
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'duration': self.duration,
        }
        
        if self.status_code is not None:
            data['status_code'] = self.status_code
            
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TraceData":
        """Create from pandas DataFrame."""
        required_cols = ['trace_id', 'span_id', 'parent_span_id', 'service_name',
                        'operation_name', 'start_time', 'duration']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        kwargs = {col: df[col].tolist() for col in required_cols}
        
        # Add optional columns if present
        if 'status_code' in df.columns:
            kwargs['status_code'] = df['status_code'].tolist()
        
        return cls(**kwargs)


@dataclass
class ServiceGraph:
    """Service dependency graph representation."""
    
    adjacency_matrix: torch.Tensor  # Shape: (num_services, num_services)
    node_features: torch.Tensor     # Shape: (num_services, feature_dim)
    edge_weights: Optional[torch.Tensor] = None  # Shape: (num_edges,)
    service_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate dimensions after initialization."""
        num_services_adj = self.adjacency_matrix.size(0)
        num_services_feat = self.node_features.size(0)
        
        if num_services_adj != num_services_feat:
            raise ValueError(
                f"Adjacency matrix size ({num_services_adj}) does not match "
                f"node features size ({num_services_feat})"
            )
        
        if self.service_names is not None:
            if len(self.service_names) != num_services_adj:
                raise ValueError(
                    f"Number of service names ({len(self.service_names)}) does not match "
                    f"adjacency matrix size ({num_services_adj})"
                )
    
    @property
    def num_services(self) -> int:
        """Get number of services in the graph."""
        return self.adjacency_matrix.size(0)
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        return self.node_features.size(1)
    
    def to_edge_index(self) -> torch.Tensor:
        """Convert adjacency matrix to edge index format for PyTorch Geometric."""
        return torch.nonzero(self.adjacency_matrix, as_tuple=False).t().contiguous()


@dataclass
class MultiModalFeatures:
    """Multi-modal feature representations."""
    
    temporal_features: torch.Tensor    # From DCNN, shape: (batch_size, hidden_dim)
    spatial_features: torch.Tensor     # From GNN, shape: (batch_size, hidden_dim)
    attention_weights: Optional[torch.Tensor] = None  # Shape: (batch_size, num_heads, seq_len)
    fused_representation: Optional[torch.Tensor] = None  # Shape: (batch_size, hidden_dim)
    
    def __post_init__(self):
        """Validate dimensions after initialization."""
        if self.temporal_features.size(0) != self.spatial_features.size(0):
            raise ValueError(
                "Temporal and spatial features must have the same batch size"
            )


@dataclass
class RootCauseLabels:
    """Root cause labels for training and evaluation."""
    
    timestamp: List[datetime]
    service_id: List[str]
    is_root_cause: List[bool]
    fault_type: Optional[List[str]] = None
    severity: Optional[List[str]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'timestamp': self.timestamp,
            'service_id': self.service_id,
            'is_root_cause': self.is_root_cause,
        }
        
        if self.fault_type is not None:
            data['fault_type'] = self.fault_type
        if self.severity is not None:
            data['severity'] = self.severity
            
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "RootCauseLabels":
        """Create from pandas DataFrame."""
        required_cols = ['timestamp', 'service_id', 'is_root_cause']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        kwargs = {col: df[col].tolist() for col in required_cols}
        
        # Add optional columns if present
        optional_cols = ['fault_type', 'severity']
        for col in optional_cols:
            if col in df.columns:
                kwargs[col] = df[col].tolist()
        
        return cls(**kwargs)


@dataclass
class DatasetSample:
    """A single sample containing all modalities for training."""
    
    metrics: torch.Tensor       # Shape: (seq_len, num_metrics)
    logs: torch.Tensor         # Shape: (num_logs, log_embedding_dim)
    graph: ServiceGraph
    label: torch.Tensor        # Shape: (num_services,) - binary labels
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for batch collation."""
        return {
            'metrics': self.metrics,
            'logs': self.logs,
            'graph': self.graph,
            'label': self.label,
            'timestamp': self.timestamp,
        }


# Type aliases for convenience
BatchData = Dict[str, Union[torch.Tensor, List[ServiceGraph], List[datetime]]]
DatasetDict = Dict[str, Union[MetricsData, LogData, TraceData, RootCauseLabels]]