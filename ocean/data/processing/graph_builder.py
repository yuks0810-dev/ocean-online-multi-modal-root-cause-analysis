"""
Service graph construction from trace data for OCEAN implementation.
Builds adjacency matrices and service dependency graphs from distributed tracing data.
"""

import numpy as np
import pandas as pd
import torch
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Union, Set
from collections import defaultdict, Counter
import logging
from datetime import datetime, timedelta

from ..data_types import TraceData, ServiceGraph


logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds service dependency graphs from distributed tracing data.
    Creates adjacency matrices and extracts graph features for OCEAN model.
    """
    
    def __init__(self,
                 edge_threshold: float = 0.0,
                 max_graph_size: int = 1000,
                 temporal_window: Optional[str] = None,
                 weight_by_frequency: bool = True,
                 include_self_loops: bool = False):
        """
        Initialize graph builder.
        
        Args:
            edge_threshold: Minimum weight threshold for edges
            max_graph_size: Maximum number of nodes in the graph
            temporal_window: Time window for graph aggregation (e.g., '1h', '5min')
            weight_by_frequency: Whether to weight edges by call frequency
            include_self_loops: Whether to include self-loop edges
        """
        self.edge_threshold = edge_threshold
        self.max_graph_size = max_graph_size
        self.temporal_window = temporal_window
        self.weight_by_frequency = weight_by_frequency
        self.include_self_loops = include_self_loops
        
        # Graph data
        self.service_to_index = {}
        self.index_to_service = {}
        self.adjacency_matrix = None
        self.node_features = None
        
        # Processing statistics
        self.processing_stats = {}
    
    def build_service_graph(self, trace_data: TraceData) -> ServiceGraph:
        """
        Build service dependency graph from trace data.
        
        Args:
            trace_data: Distributed tracing data
            
        Returns:
            ServiceGraph object with adjacency matrix and features
        """
        logger.info("Building service dependency graph from traces")
        
        # Convert to DataFrame for easier processing
        df = trace_data.to_dataframe()
        
        # Store original statistics
        self.processing_stats['original_traces'] = len(df)
        self.processing_stats['original_services'] = df['service_name'].nunique()
        
        # Clean and prepare trace data
        df = self._clean_trace_data(df)
        
        # Build service mapping
        self._build_service_mapping(df)
        
        # Extract service dependencies
        dependencies = self._extract_dependencies(df)
        
        # Create adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(dependencies)
        
        # Create node features
        node_features = self._create_node_features(df)
        
        # Create service graph object
        service_graph = ServiceGraph(
            adjacency_matrix=adjacency_matrix,
            node_features=node_features,
            service_names=list(self.service_to_index.keys())
        )
        
        # Store final statistics
        self.processing_stats['final_services'] = len(self.service_to_index)
        self.processing_stats['total_edges'] = (adjacency_matrix > 0).sum().item()
        
        logger.info(f"Built service graph: {len(self.service_to_index)} services, {self.processing_stats['total_edges']} edges")
        
        return service_graph
    
    def _clean_trace_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare trace data."""
        logger.info("Cleaning trace data")
        
        df = df.copy()
        
        # Convert timestamps
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Remove invalid traces
        initial_count = len(df)
        
        # Remove traces with missing essential fields
        df = df.dropna(subset=['trace_id', 'span_id', 'service_name'])
        
        # Remove traces with invalid durations
        df = df[df['duration'] >= 0]
        
        # Remove extremely long durations (likely errors)
        duration_99th = df['duration'].quantile(0.99)
        df = df[df['duration'] <= duration_99th * 10]  # Allow some outliers
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} invalid traces")
        
        self.processing_stats['invalid_traces_removed'] = removed_count
        
        # Sort by trace_id and start_time for easier processing
        df = df.sort_values(['trace_id', 'start_time'])
        
        return df
    
    def _build_service_mapping(self, df: pd.DataFrame) -> None:
        """Build mapping between service names and indices."""
        unique_services = sorted(df['service_name'].unique())
        
        # Limit services if necessary
        if len(unique_services) > self.max_graph_size:
            logger.warning(f"Too many services ({len(unique_services)}), limiting to {self.max_graph_size}")
            
            # Keep most frequent services
            service_counts = df['service_name'].value_counts()
            unique_services = service_counts.head(self.max_graph_size).index.tolist()
        
        # Create mappings
        self.service_to_index = {service: idx for idx, service in enumerate(unique_services)}
        self.index_to_service = {idx: service for service, idx in self.service_to_index.items()}
        
        logger.info(f"Created service mapping for {len(unique_services)} services")
    
    def _extract_dependencies(self, df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Extract service dependencies from trace data."""
        logger.info("Extracting service dependencies")
        
        dependencies = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
            'min_duration': float('inf'),
            'max_duration': 0.0,
            'error_count': 0,
            'traces': set()
        })
        
        # Group by trace to find parent-child relationships
        for trace_id, trace_group in df.groupby('trace_id'):
            trace_group = trace_group.sort_values('start_time')
            
            # Create span lookup
            span_lookup = {row['span_id']: row for _, row in trace_group.iterrows()}
            
            # Find parent-child relationships
            for _, span in trace_group.iterrows():
                parent_span_id = span['parent_span_id']
                
                if pd.isna(parent_span_id) or parent_span_id not in span_lookup:
                    continue
                
                parent_span = span_lookup[parent_span_id]
                parent_service = parent_span['service_name']
                child_service = span['service_name']
                
                # Skip self-loops unless explicitly allowed
                if not self.include_self_loops and parent_service == child_service:
                    continue
                
                # Skip services not in our mapping
                if (parent_service not in self.service_to_index or 
                    child_service not in self.service_to_index):
                    continue
                
                # Record dependency
                edge = (parent_service, child_service)
                dep = dependencies[edge]
                
                dep['count'] += 1
                dep['total_duration'] += span['duration']
                dep['min_duration'] = min(dep['min_duration'], span['duration'])
                dep['max_duration'] = max(dep['max_duration'], span['duration'])
                dep['traces'].add(trace_id)
                
                # Check for errors
                if 'status_code' in span and span['status_code'] and 'error' in str(span['status_code']).lower():
                    dep['error_count'] += 1
        
        # Calculate average durations
        for edge, dep in dependencies.items():
            if dep['count'] > 0:
                dep['avg_duration'] = dep['total_duration'] / dep['count']
            if dep['min_duration'] == float('inf'):
                dep['min_duration'] = 0.0
        
        logger.info(f"Extracted {len(dependencies)} service dependencies")
        self.processing_stats['raw_dependencies'] = len(dependencies)
        
        return dict(dependencies)
    
    def _create_adjacency_matrix(self, dependencies: Dict[Tuple[str, str], Dict[str, Any]]) -> torch.Tensor:
        """Create adjacency matrix from dependencies."""
        logger.info("Creating adjacency matrix")
        
        num_services = len(self.service_to_index)
        adjacency = np.zeros((num_services, num_services), dtype=np.float32)
        
        # Fill adjacency matrix
        for (parent_service, child_service), dep_info in dependencies.items():
            parent_idx = self.service_to_index[parent_service]
            child_idx = self.service_to_index[child_service]
            
            if self.weight_by_frequency:
                weight = dep_info['count']
            else:
                weight = 1.0
            
            adjacency[parent_idx, child_idx] = weight
        
        # Apply edge threshold
        if self.edge_threshold > 0:
            adjacency[adjacency < self.edge_threshold] = 0
        
        # Normalize rows (outgoing edges sum to 1)
        row_sums = adjacency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        adjacency = adjacency / row_sums
        
        self.adjacency_matrix = torch.tensor(adjacency, dtype=torch.float32)
        
        edges_after_threshold = (self.adjacency_matrix > 0).sum().item()
        logger.info(f"Created adjacency matrix: {edges_after_threshold} edges after threshold")
        
        return self.adjacency_matrix
    
    def _create_node_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Create node features for each service."""
        logger.info("Creating node features")
        
        num_services = len(self.service_to_index)
        feature_list = []
        
        for service_name, service_idx in self.service_to_index.items():
            service_data = df[df['service_name'] == service_name]
            
            if len(service_data) == 0:
                # Create zero features for services with no data
                features = np.zeros(self._get_feature_dimension())
            else:
                features = self._extract_service_features(service_data)
            
            feature_list.append(features)
        
        node_features = torch.tensor(np.array(feature_list), dtype=torch.float32)
        self.node_features = node_features
        
        logger.info(f"Created node features: {node_features.shape}")
        return node_features
    
    def _extract_service_features(self, service_data: pd.DataFrame) -> np.ndarray:
        """Extract features for a single service."""
        features = []
        
        # Basic statistics
        features.append(len(service_data))  # Total spans
        features.append(service_data['trace_id'].nunique())  # Unique traces
        
        # Duration statistics
        durations = service_data['duration']
        features.extend([
            durations.mean(),
            durations.std(),
            durations.min(),
            durations.max(),
            durations.median(),
            durations.quantile(0.95),
            durations.quantile(0.99)
        ])
        
        # Time-based features
        if 'start_time' in service_data.columns:
            start_times = pd.to_datetime(service_data['start_time'])
            time_range = (start_times.max() - start_times.min()).total_seconds()
            features.append(time_range)
            
            # Request rate (spans per second)
            if time_range > 0:
                features.append(len(service_data) / time_range)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Error rate
        if 'status_code' in service_data.columns:
            error_count = service_data['status_code'].fillna('').str.contains('error|fail', case=False).sum()
            features.append(error_count / len(service_data) if len(service_data) > 0 else 0.0)
        else:
            features.append(0.0)
        
        # Operation diversity
        if 'operation_name' in service_data.columns:
            unique_operations = service_data['operation_name'].nunique()
            features.append(unique_operations)
        else:
            features.append(1.0)
        
        # Convert to numpy array and handle any NaN values
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _get_feature_dimension(self) -> int:
        """Get the dimension of node features."""
        # Count features in _extract_service_features
        return 12  # Total number of features extracted per service
    
    def create_temporal_graphs(self, 
                             trace_data: TraceData,
                             time_window: str = '1h') -> List[Tuple[datetime, ServiceGraph]]:
        """
        Create temporal sequence of service graphs.
        
        Args:
            trace_data: Distributed tracing data
            time_window: Time window for each graph (e.g., '1h', '30min')
            
        Returns:
            List of (timestamp, ServiceGraph) tuples
        """
        logger.info(f"Creating temporal graphs with window {time_window}")
        
        df = trace_data.to_dataframe()
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Create time windows
        start_time = df['start_time'].min()
        end_time = df['start_time'].max()
        
        # Generate time ranges
        time_ranges = pd.date_range(start=start_time, end=end_time, freq=time_window)
        if len(time_ranges) < 2:
            time_ranges = [start_time, end_time]
        
        temporal_graphs = []
        
        for i in range(len(time_ranges) - 1):
            window_start = time_ranges[i]
            window_end = time_ranges[i + 1]
            
            # Filter data for this time window
            window_data = df[
                (df['start_time'] >= window_start) & 
                (df['start_time'] < window_end)
            ]
            
            if len(window_data) > 0:
                # Create TraceData for this window
                window_trace_data = TraceData(
                    trace_id=window_data['trace_id'].tolist(),
                    span_id=window_data['span_id'].tolist(),
                    parent_span_id=window_data['parent_span_id'].tolist(),
                    service_name=window_data['service_name'].tolist(),
                    operation_name=window_data['operation_name'].tolist(),
                    start_time=window_data['start_time'].tolist(),
                    duration=window_data['duration'].tolist()
                )
                
                # Build graph for this window
                try:
                    graph = self.build_service_graph(window_trace_data)
                    temporal_graphs.append((window_start, graph))
                except Exception as e:
                    logger.warning(f"Failed to build graph for window {window_start}: {e}")
        
        logger.info(f"Created {len(temporal_graphs)} temporal graphs")
        return temporal_graphs
    
    def analyze_graph_properties(self, service_graph: ServiceGraph) -> Dict[str, Any]:
        """Analyze properties of the service graph."""
        adjacency = service_graph.adjacency_matrix.numpy()
        
        # Convert to NetworkX for analysis
        G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        
        # Relabel nodes with service names
        if service_graph.service_names:
            node_mapping = {i: name for i, name in enumerate(service_graph.service_names)}
            G = nx.relabel_nodes(G, node_mapping)
        
        analysis = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_strongly_connected': nx.is_strongly_connected(G),
            'is_weakly_connected': nx.is_weakly_connected(G),
            'num_strongly_connected_components': nx.number_strongly_connected_components(G),
            'num_weakly_connected_components': nx.number_weakly_connected_components(G),
        }
        
        # Calculate centrality measures
        try:
            analysis['in_degree_centrality'] = nx.in_degree_centrality(G)
            analysis['out_degree_centrality'] = nx.out_degree_centrality(G)
            analysis['pagerank'] = nx.pagerank(G)
            analysis['betweenness_centrality'] = nx.betweenness_centrality(G)
        except Exception as e:
            logger.warning(f"Failed to calculate centrality measures: {e}")
        
        # Find critical services (high centrality)
        if 'pagerank' in analysis:
            pagerank_scores = analysis['pagerank']
            sorted_services = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            analysis['top_services_by_pagerank'] = sorted_services[:5]
        
        return analysis
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph building statistics."""
        stats = self.processing_stats.copy()
        
        if self.adjacency_matrix is not None:
            stats['adjacency_matrix_shape'] = self.adjacency_matrix.shape
            stats['adjacency_matrix_sparsity'] = 1.0 - (self.adjacency_matrix > 0).float().mean().item()
        
        if self.node_features is not None:
            stats['node_features_shape'] = self.node_features.shape
        
        return stats
    
    def save_graph(self, service_graph: ServiceGraph, filepath: str) -> None:
        """Save service graph to file."""
        graph_data = {
            'adjacency_matrix': service_graph.adjacency_matrix.numpy(),
            'node_features': service_graph.node_features.numpy(),
            'service_names': service_graph.service_names,
            'service_to_index': self.service_to_index,
            'processing_stats': self.processing_stats
        }
        
        np.savez_compressed(filepath, **graph_data)
        logger.info(f"Service graph saved to {filepath}")
    
    def load_graph(self, filepath: str) -> ServiceGraph:
        """Load service graph from file."""
        data = np.load(filepath, allow_pickle=True)
        
        adjacency_matrix = torch.tensor(data['adjacency_matrix'], dtype=torch.float32)
        node_features = torch.tensor(data['node_features'], dtype=torch.float32)
        service_names = data['service_names'].tolist() if 'service_names' in data else None
        
        if 'service_to_index' in data:
            self.service_to_index = data['service_to_index'].item()
            self.index_to_service = {v: k for k, v in self.service_to_index.items()}
        
        if 'processing_stats' in data:
            self.processing_stats = data['processing_stats'].item()
        
        service_graph = ServiceGraph(
            adjacency_matrix=adjacency_matrix,
            node_features=node_features,
            service_names=service_names
        )
        
        logger.info(f"Service graph loaded from {filepath}")
        return service_graph