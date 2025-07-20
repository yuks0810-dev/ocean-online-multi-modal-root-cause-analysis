"""
Multi-modal dataset and data loaders for OCEAN implementation.
Handles synchronized loading of metrics, logs, and graph data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from ..data_types import DatasetSample, ServiceGraph, BatchData
from ..processing import MetricsProcessor, LogProcessor, GraphBuilder


logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal microservice data.
    Synchronizes metrics, logs, and graph data for OCEAN training.
    """
    
    def __init__(self,
                 metrics_sequences: List[Dict[str, Any]],
                 log_sequences: List[Dict[str, Any]],
                 service_graph: ServiceGraph,
                 labels: Optional[pd.DataFrame] = None,
                 time_alignment_window: str = '1min',
                 require_all_modalities: bool = False):
        """
        Initialize multi-modal dataset.
        
        Args:
            metrics_sequences: List of metrics sequences from MetricsProcessor
            log_sequences: List of log sequences from LogProcessor
            service_graph: Service dependency graph
            labels: DataFrame with root cause labels
            time_alignment_window: Time window for aligning different modalities
            require_all_modalities: Whether to require all modalities for each sample
        """
        self.metrics_sequences = metrics_sequences
        self.log_sequences = log_sequences
        self.service_graph = service_graph
        self.labels = labels
        self.time_alignment_window = time_alignment_window
        self.require_all_modalities = require_all_modalities
        
        # Create synchronized samples
        self.samples = self._create_synchronized_samples()
        
        logger.info(f"Created MultiModalDataset with {len(self.samples)} synchronized samples")
    
    def _create_synchronized_samples(self) -> List[Dict[str, Any]]:
        """Create synchronized samples from different modalities."""
        logger.info("Creating synchronized multi-modal samples")
        
        # Group sequences by service and time
        metrics_by_service = defaultdict(list)
        logs_by_service = defaultdict(list)
        
        # Group metrics sequences
        for seq in self.metrics_sequences:
            service_id = seq['service_id']
            metrics_by_service[service_id].append(seq)
        
        # Group log sequences
        for seq in self.log_sequences:
            service_id = seq['service_id']
            logs_by_service[service_id].append(seq)
        
        # Get all services
        all_services = set(metrics_by_service.keys()) | set(logs_by_service.keys())
        
        # Filter to services in the graph
        if self.service_graph.service_names:
            graph_services = set(self.service_graph.service_names)
            all_services = all_services & graph_services
        
        samples = []
        
        for service_id in all_services:
            service_metrics = metrics_by_service.get(service_id, [])
            service_logs = logs_by_service.get(service_id, [])
            
            # Create samples by time alignment
            service_samples = self._align_sequences_by_time(
                service_id, service_metrics, service_logs
            )
            samples.extend(service_samples)
        
        logger.info(f"Created {len(samples)} synchronized samples from {len(all_services)} services")
        return samples
    
    def _align_sequences_by_time(self,
                                service_id: str,
                                metrics_sequences: List[Dict[str, Any]],
                                log_sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Align metrics and log sequences by time for a single service."""
        samples = []
        
        # If we require all modalities and one is missing, skip this service
        if self.require_all_modalities:
            if not metrics_sequences or not log_sequences:
                return samples
        
        # Use metrics sequences as primary (they typically have more regular timing)
        for metrics_seq in metrics_sequences:
            sample = {
                'service_id': service_id,
                'metrics': metrics_seq,
                'logs': None,
                'timestamp': metrics_seq['start_time']
            }
            
            # Find matching log sequence
            if log_sequences:
                matching_log = self._find_matching_log_sequence(
                    metrics_seq, log_sequences
                )
                if matching_log:
                    sample['logs'] = matching_log
            
            # Skip if all modalities required but logs missing
            if self.require_all_modalities and sample['logs'] is None:
                continue
            
            samples.append(sample)
        
        # Also add log sequences that don't have matching metrics (if not requiring all)
        if not self.require_all_modalities:
            for log_seq in log_sequences:
                # Check if this log sequence is already used
                log_already_used = any(
                    sample['logs'] and sample['logs']['sequence_id'] == log_seq['sequence_id']
                    for sample in samples
                )
                
                if not log_already_used:
                    sample = {
                        'service_id': service_id,
                        'metrics': None,
                        'logs': log_seq,
                        'timestamp': log_seq['start_time']
                    }
                    samples.append(sample)
        
        return samples
    
    def _find_matching_log_sequence(self,
                                  metrics_seq: Dict[str, Any],
                                  log_sequences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find log sequence that best matches the metrics sequence timing."""
        metrics_start = pd.to_datetime(metrics_seq['start_time'])
        metrics_end = pd.to_datetime(metrics_seq['end_time'])
        
        best_match = None
        best_overlap = 0
        
        for log_seq in log_sequences:
            log_start = pd.to_datetime(log_seq['start_time'])
            log_end = pd.to_datetime(log_seq['end_time'])
            
            # Calculate time overlap
            overlap_start = max(metrics_start, log_start)
            overlap_end = min(metrics_end, log_end)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                metrics_duration = (metrics_end - metrics_start).total_seconds()
                log_duration = (log_end - log_start).total_seconds()
                
                # Calculate overlap percentage relative to both sequences
                if metrics_duration > 0 and log_duration > 0:
                    overlap_pct = overlap_duration / min(metrics_duration, log_duration)
                    
                    if overlap_pct > best_overlap:
                        best_overlap = overlap_pct
                        best_match = log_seq
        
        # Return match if overlap is significant (>50%)
        return best_match if best_overlap > 0.5 else None
    
    def _get_service_labels(self, service_id: str, timestamp: datetime) -> torch.Tensor:
        """Get root cause labels for a service at a specific time."""
        if self.labels is None:
            # Return default label (no root cause)
            if self.service_graph.service_names:
                service_idx = self.service_graph.service_names.index(service_id) if service_id in self.service_graph.service_names else 0
                label = torch.zeros(len(self.service_graph.service_names))
                return label
            else:
                return torch.tensor([0.0])
        
        # Find matching labels within time window
        timestamp = pd.to_datetime(timestamp)
        time_window = pd.Timedelta(self.time_alignment_window)
        
        relevant_labels = self.labels[
            (self.labels['service_id'] == service_id) &
            (pd.to_datetime(self.labels['timestamp']) >= timestamp - time_window) &
            (pd.to_datetime(self.labels['timestamp']) <= timestamp + time_window)
        ]
        
        if len(relevant_labels) > 0:
            # Use most recent label
            latest_label = relevant_labels.iloc[-1]
            is_root_cause = latest_label['is_root_cause']
        else:
            is_root_cause = False
        
        # Create label vector for all services
        if self.service_graph.service_names:
            label_vector = torch.zeros(len(self.service_graph.service_names))
            try:
                service_idx = self.service_graph.service_names.index(service_id)
                label_vector[service_idx] = 1.0 if is_root_cause else 0.0
            except ValueError:
                pass  # Service not in graph
            return label_vector
        else:
            return torch.tensor([1.0 if is_root_cause else 0.0])
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DatasetSample:
        """Get a single sample from the dataset."""
        sample_data = self.samples[idx]
        
        service_id = sample_data['service_id']
        timestamp = sample_data['timestamp']
        
        # Process metrics
        if sample_data['metrics'] is not None:
            metrics_tensor = torch.tensor(
                sample_data['metrics']['features'], 
                dtype=torch.float32
            )
        else:
            # Create zero tensor if no metrics
            # Assume standard metrics dimension
            metrics_tensor = torch.zeros((100, 12), dtype=torch.float32)  # seq_len, num_features
        
        # Process logs
        if sample_data['logs'] is not None:
            log_features = sample_data['logs']['features']
            
            # Combine different log feature types
            log_tensors = []
            
            if 'embeddings' in log_features:
                embeddings = torch.tensor(log_features['embeddings'], dtype=torch.float32)
                log_tensors.append(embeddings.mean(dim=0, keepdim=True))  # Average over sequence
            
            if 'numerical' in log_features:
                numerical = torch.tensor(log_features['numerical'], dtype=torch.float32)
                log_tensors.append(numerical.mean(dim=0, keepdim=True))
            
            if 'binary' in log_features:
                binary = torch.tensor(log_features['binary'], dtype=torch.float32)
                log_tensors.append(binary.mean(dim=0, keepdim=True))
            
            if log_tensors:
                logs_tensor = torch.cat(log_tensors, dim=1).squeeze(0)
            else:
                logs_tensor = torch.zeros(768, dtype=torch.float32)  # Default BERT size
        else:
            # Create zero tensor if no logs
            logs_tensor = torch.zeros(768, dtype=torch.float32)
        
        # Get labels
        labels_tensor = self._get_service_labels(service_id, timestamp)
        
        # Create DatasetSample
        dataset_sample = DatasetSample(
            metrics=metrics_tensor,
            logs=logs_tensor,
            graph=self.service_graph,
            label=labels_tensor,
            timestamp=timestamp
        )
        
        return dataset_sample


def collate_multimodal_batch(batch: List[DatasetSample]) -> BatchData:
    """
    Custom collate function for multi-modal batches.
    
    Args:
        batch: List of DatasetSample objects
        
    Returns:
        BatchData dictionary
    """
    if not batch:
        return {}
    
    # Pad sequences to same length
    max_metrics_len = max(sample.metrics.size(0) for sample in batch)
    max_log_len = max(sample.logs.size(0) if sample.logs.dim() > 0 else 1 for sample in batch)
    
    # Collect batch data
    metrics_batch = []
    logs_batch = []
    labels_batch = []
    timestamps_batch = []
    graphs_batch = []
    
    for sample in batch:
        # Pad metrics
        metrics = sample.metrics
        if metrics.size(0) < max_metrics_len:
            padding = torch.zeros((max_metrics_len - metrics.size(0), metrics.size(1)))
            metrics = torch.cat([metrics, padding], dim=0)
        metrics_batch.append(metrics)
        
        # Handle logs
        logs = sample.logs
        if logs.dim() == 0:  # Scalar tensor
            logs = logs.unsqueeze(0)
        logs_batch.append(logs)
        
        labels_batch.append(sample.label)
        timestamps_batch.append(sample.timestamp)
        graphs_batch.append(sample.graph)
    
    # Stack tensors
    batch_data = {
        'metrics': torch.stack(metrics_batch),
        'logs': torch.stack(logs_batch),
        'labels': torch.stack(labels_batch),
        'timestamps': timestamps_batch,
        'graphs': graphs_batch  # Keep as list since graphs might have different sizes
    }
    
    return batch_data


class MultiModalDataLoader:
    """
    Data loader manager for multi-modal OCEAN training.
    Handles train/validation/test splits and batch creation.
    """
    
    def __init__(self,
                 dataset: MultiModalDataset,
                 batch_size: int = 32,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 temporal_split: bool = True):
        """
        Initialize multi-modal data loader.
        
        Args:
            dataset: MultiModalDataset instance
            batch_size: Batch size for training
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            shuffle: Whether to shuffle data (ignored if temporal_split=True)
            num_workers: Number of worker processes
            temporal_split: Whether to split data temporally (chronologically)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle and not temporal_split
        self.num_workers = num_workers
        self.temporal_split = temporal_split
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Create data splits
        self.train_indices, self.val_indices, self.test_indices = self._create_splits()
        
        logger.info(f"Created data splits: train={len(self.train_indices)}, "
                   f"val={len(self.val_indices)}, test={len(self.test_indices)}")
    
    def _create_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """Create train/validation/test splits."""
        n_samples = len(self.dataset)
        indices = list(range(n_samples))
        
        if self.temporal_split:
            # Sort by timestamp for temporal split
            timestamps = [self.dataset.samples[i]['timestamp'] for i in indices]
            sorted_indices = [i for _, i in sorted(zip(timestamps, indices))]
            
            n_train = int(n_samples * self.train_ratio)
            n_val = int(n_samples * self.val_ratio)
            
            train_indices = sorted_indices[:n_train]
            val_indices = sorted_indices[n_train:n_train + n_val]
            test_indices = sorted_indices[n_train + n_val:]
        else:
            # Random split
            if self.shuffle:
                np.random.shuffle(indices)
            
            n_train = int(n_samples * self.train_ratio)
            n_val = int(n_samples * self.val_ratio)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        
        return train_indices, val_indices, test_indices
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_multimodal_batch
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        val_dataset = torch.utils.data.Subset(self.dataset, self.val_indices)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_multimodal_batch
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        test_dataset = torch.utils.data.Subset(self.dataset, self.test_indices)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_multimodal_batch
        )
    
    def get_all_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all data loaders (train, validation, test)."""
        return (
            self.get_train_loader(),
            self.get_val_loader(),
            self.get_test_loader()
        )
    
    def get_split_statistics(self) -> Dict[str, Any]:
        """Get statistics about the data splits."""
        stats = {
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_indices),
            'val_samples': len(self.val_indices),
            'test_samples': len(self.test_indices),
            'train_ratio_actual': len(self.train_indices) / len(self.dataset),
            'val_ratio_actual': len(self.val_indices) / len(self.dataset),
            'test_ratio_actual': len(self.test_indices) / len(self.dataset),
        }
        
        # Add temporal information if temporal split
        if self.temporal_split:
            train_timestamps = [self.dataset.samples[i]['timestamp'] for i in self.train_indices]
            val_timestamps = [self.dataset.samples[i]['timestamp'] for i in self.val_indices]
            test_timestamps = [self.dataset.samples[i]['timestamp'] for i in self.test_indices]
            
            stats['temporal_split_info'] = {
                'train_period': (min(train_timestamps), max(train_timestamps)) if train_timestamps else None,
                'val_period': (min(val_timestamps), max(val_timestamps)) if val_timestamps else None,
                'test_period': (min(test_timestamps), max(test_timestamps)) if test_timestamps else None,
            }
        
        return stats


class OnlineDataLoader:
    """
    Online data loader for streaming multi-modal data.
    Processes data incrementally for online learning.
    """
    
    def __init__(self,
                 dataset: MultiModalDataset,
                 window_size: int = 1000,
                 step_size: int = 100):
        """
        Initialize online data loader.
        
        Args:
            dataset: MultiModalDataset instance
            window_size: Size of sliding window for online learning
            step_size: Step size for sliding window
        """
        self.dataset = dataset
        self.window_size = window_size
        self.step_size = step_size
        self.current_position = 0
        
        # Sort samples by timestamp for online processing
        timestamps = [sample['timestamp'] for sample in self.dataset.samples]
        self.sorted_indices = [i for _, i in sorted(zip(timestamps, range(len(self.dataset))))]
        
        logger.info(f"Initialized OnlineDataLoader with window_size={window_size}, step_size={step_size}")
    
    def get_next_window(self) -> Optional[List[DatasetSample]]:
        """Get next window of data for online learning."""
        if self.current_position >= len(self.sorted_indices):
            return None
        
        # Get window indices
        start_idx = self.current_position
        end_idx = min(start_idx + self.window_size, len(self.sorted_indices))
        
        window_indices = self.sorted_indices[start_idx:end_idx]
        window_samples = [self.dataset[i] for i in window_indices]
        
        # Update position
        self.current_position += self.step_size
        
        return window_samples
    
    def reset(self) -> None:
        """Reset to beginning of dataset."""
        self.current_position = 0
    
    def has_next(self) -> bool:
        """Check if there are more windows available."""
        return self.current_position < len(self.sorted_indices)