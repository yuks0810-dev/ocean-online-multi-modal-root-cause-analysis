"""
Online learning framework for OCEAN model.
Manages sequential data processing and incremental model updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from datetime import datetime, timedelta
import threading
import queue
import time

from ..ocean_model import OCEANModel
from ...data.data_types import ServiceGraph, DatasetSample, BatchData
from ...configs import OCEANConfig


logger = logging.getLogger(__name__)


class MemoryBuffer:
    """
    Memory buffer for storing recent samples in online learning.
    """
    
    def __init__(self, max_size: int = 1000, retention_strategy: str = 'fifo'):
        """
        Initialize memory buffer.
        
        Args:
            max_size: Maximum number of samples to store
            retention_strategy: Strategy for removing old samples ('fifo', 'random', 'importance')
        """
        self.max_size = max_size
        self.retention_strategy = retention_strategy
        self.buffer = deque(maxlen=max_size)
        self.importance_scores = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
    def add(self, sample: DatasetSample, importance_score: float = 1.0):
        """Add sample to buffer."""
        current_time = datetime.now()
        
        if len(self.buffer) >= self.max_size:
            self._remove_sample()
        
        self.buffer.append(sample)
        self.importance_scores.append(importance_score)
        self.timestamps.append(current_time)
    
    def _remove_sample(self):
        """Remove sample based on retention strategy."""
        if self.retention_strategy == 'fifo':
            # Default deque behavior (oldest first)
            self.importance_scores.popleft()
            self.timestamps.popleft()
        elif self.retention_strategy == 'random':
            # Remove random sample
            if len(self.buffer) > 0:
                idx = np.random.randint(0, len(self.buffer))
                # Convert to list for indexed removal
                buffer_list = list(self.buffer)
                importance_list = list(self.importance_scores)
                timestamps_list = list(self.timestamps)
                
                buffer_list.pop(idx)
                importance_list.pop(idx)
                timestamps_list.pop(idx)
                
                self.buffer = deque(buffer_list, maxlen=self.max_size)
                self.importance_scores = deque(importance_list, maxlen=self.max_size)
                self.timestamps = deque(timestamps_list, maxlen=self.max_size)
        elif self.retention_strategy == 'importance':
            # Remove least important sample
            if len(self.importance_scores) > 0:
                min_idx = np.argmin(list(self.importance_scores))
                # Similar to random strategy
                buffer_list = list(self.buffer)
                importance_list = list(self.importance_scores)
                timestamps_list = list(self.timestamps)
                
                buffer_list.pop(min_idx)
                importance_list.pop(min_idx)
                timestamps_list.pop(min_idx)
                
                self.buffer = deque(buffer_list, maxlen=self.max_size)
                self.importance_scores = deque(importance_list, maxlen=self.max_size)
                self.timestamps = deque(timestamps_list, maxlen=self.max_size)
    
    def sample(self, batch_size: int) -> List[DatasetSample]:
        """Sample batch from buffer."""
        if len(self.buffer) == 0:
            return []
        
        if batch_size >= len(self.buffer):
            return list(self.buffer)
        
        # Sample based on importance scores
        if self.retention_strategy == 'importance':
            weights = np.array(list(self.importance_scores))
            weights = weights / weights.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, 
                                     replace=False, p=weights)
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def get_recent(self, time_window: timedelta) -> List[DatasetSample]:
        """Get samples from recent time window."""
        current_time = datetime.now()
        cutoff_time = current_time - time_window
        
        recent_samples = []
        for sample, timestamp in zip(self.buffer, self.timestamps):
            if timestamp >= cutoff_time:
                recent_samples.append(sample)
        
        return recent_samples
    
    def __len__(self):
        return len(self.buffer)


class SlidingWindow:
    """
    Sliding window for temporal dependencies in online learning.
    """
    
    def __init__(self, window_size: int, step_size: int = 1):
        """
        Initialize sliding window.
        
        Args:
            window_size: Size of the sliding window
            step_size: Step size for moving the window
        """
        self.window_size = window_size
        self.step_size = step_size
        self.window = deque(maxlen=window_size)
        self.step_counter = 0
    
    def add(self, sample: DatasetSample) -> bool:
        """
        Add sample to window.
        
        Returns:
            True if window is ready for processing
        """
        self.window.append(sample)
        self.step_counter += 1
        
        # Check if we should process the window
        return (len(self.window) == self.window_size and 
                self.step_counter % self.step_size == 0)
    
    def get_window(self) -> List[DatasetSample]:
        """Get current window contents."""
        return list(self.window)
    
    def reset(self):
        """Reset the window."""
        self.window.clear()
        self.step_counter = 0


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler for online learning.
    """
    
    def __init__(self, 
                 initial_lr: float = 0.001,
                 decay_factor: float = 0.95,
                 adaptation_window: int = 100,
                 min_lr: float = 1e-6):
        """
        Initialize adaptive scheduler.
        
        Args:
            initial_lr: Initial learning rate
            decay_factor: Decay factor for learning rate
            adaptation_window: Window size for performance monitoring
            min_lr: Minimum learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.adaptation_window = adaptation_window
        self.min_lr = min_lr
        
        self.performance_history = deque(maxlen=adaptation_window)
        self.lr_history = []
    
    def step(self, performance_metric: float, optimizer: optim.Optimizer) -> float:
        """
        Update learning rate based on performance.
        
        Args:
            performance_metric: Current performance metric (lower is better)
            optimizer: PyTorch optimizer
            
        Returns:
            New learning rate
        """
        self.performance_history.append(performance_metric)
        
        # Adapt learning rate if we have enough history
        if len(self.performance_history) >= self.adaptation_window:
            recent_avg = np.mean(list(self.performance_history)[-self.adaptation_window//2:])
            older_avg = np.mean(list(self.performance_history)[:self.adaptation_window//2])
            
            # If performance is not improving, decay learning rate
            if recent_avg >= older_avg:
                self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                
                # Update optimizer learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.current_lr
        
        self.lr_history.append(self.current_lr)
        return self.current_lr
    
    def reset(self):
        """Reset to initial learning rate."""
        self.current_lr = self.initial_lr
        self.performance_history.clear()


class OnlineLearner:
    """
    Online learner for OCEAN model with incremental updates and streaming data.
    """
    
    def __init__(self, 
                 model: OCEANModel,
                 config: OCEANConfig,
                 optimizer: Optional[optim.Optimizer] = None):
        """
        Initialize online learner.
        
        Args:
            model: OCEAN model instance
            config: OCEAN configuration
            optimizer: PyTorch optimizer (if None, creates Adam optimizer)
        """
        self.model = model
        self.config = config
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Online learning components
        self.memory_buffer = MemoryBuffer(
            max_size=config.training.sliding_window_size,
            retention_strategy='importance'
        )
        
        self.sliding_window = SlidingWindow(
            window_size=getattr(config.data, 'sequence_length', 100),
            step_size=config.training.update_frequency
        )
        
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=config.training.learning_rate
        )
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.update_count = 0
        self.last_update_time = datetime.now()
        
        # Streaming data queue (for async processing)
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_streaming = False
        self.stream_thread = None
        
        logger.info("Initialized OnlineLearner for streaming data processing")
    
    def process_sample(self, sample: DatasetSample, update_model: bool = True) -> Dict[str, float]:
        """
        Process a single sample in online learning mode.
        
        Args:
            sample: Input sample
            update_model: Whether to update model parameters
            
        Returns:
            Dictionary with processing results
        """
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            sample.metrics.unsqueeze(0),  # Add batch dimension
            sample.graph,
            sample.logs.unsqueeze(0),     # Add batch dimension
        )
        
        # Compute loss
        target = sample.label.unsqueeze(0)  # Add batch dimension
        losses = self.model.compute_loss(outputs, target)
        
        # Calculate importance score for memory buffer
        importance_score = losses['total_loss'].item()
        
        # Add to memory buffer
        self.memory_buffer.add(sample, importance_score)
        
        # Update model if requested
        if update_model:
            # Gradient computation and update
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            self.update_count += 1
            
            # Adapt learning rate
            current_lr = self.lr_scheduler.step(
                losses['total_loss'].item(), 
                self.optimizer
            )
            
            # Log performance
            self.performance_history['loss'].append(losses['total_loss'].item())
            self.performance_history['learning_rate'].append(current_lr)
        
        # Return results
        results = {
            'loss': losses['total_loss'].item(),
            'prediction_loss': losses['prediction_loss'].item(),
            'importance_score': importance_score,
            'memory_buffer_size': len(self.memory_buffer),
            'update_count': self.update_count
        }
        
        if 'contrastive_loss' in losses:
            results['contrastive_loss'] = losses['contrastive_loss'].item()
        
        return results
    
    def process_batch(self, batch_samples: List[DatasetSample]) -> Dict[str, float]:
        """
        Process a batch of samples with experience replay.
        
        Args:
            batch_samples: List of samples to process
            
        Returns:
            Batch processing results
        """
        if not batch_samples:
            return {}
        
        self.model.train()
        
        # Prepare batch tensors
        metrics_batch = []
        logs_batch = []
        labels_batch = []
        
        # Use the same service graph for all samples (assuming consistency)
        service_graph = batch_samples[0].graph
        
        for sample in batch_samples:
            metrics_batch.append(sample.metrics)
            logs_batch.append(sample.logs)
            labels_batch.append(sample.label)
        
        # Stack into tensors
        metrics_tensor = torch.stack(metrics_batch)
        logs_tensor = torch.stack(logs_batch)
        labels_tensor = torch.stack(labels_batch)
        
        # Forward pass
        outputs = self.model(metrics_tensor, service_graph, logs_tensor)
        
        # Compute losses
        losses = self.model.compute_loss(outputs, labels_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        if self.config.training.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.gradient_clip_norm
            )
        
        self.optimizer.step()
        self.update_count += 1
        
        # Adapt learning rate
        current_lr = self.lr_scheduler.step(
            losses['total_loss'].item(), 
            self.optimizer
        )
        
        # Update performance history
        self.performance_history['batch_loss'].append(losses['total_loss'].item())
        self.performance_history['learning_rate'].append(current_lr)
        
        return {
            'batch_loss': losses['total_loss'].item(),
            'batch_prediction_loss': losses['prediction_loss'].item(),
            'batch_size': len(batch_samples),
            'learning_rate': current_lr,
            'update_count': self.update_count
        }
    
    def experience_replay(self, replay_batch_size: int = 32) -> Optional[Dict[str, float]]:
        """
        Perform experience replay on memory buffer.
        
        Args:
            replay_batch_size: Size of replay batch
            
        Returns:
            Replay results or None if insufficient samples
        """
        if len(self.memory_buffer) < replay_batch_size:
            return None
        
        # Sample from memory buffer
        replay_samples = self.memory_buffer.sample(replay_batch_size)
        
        # Process replay batch
        results = self.process_batch(replay_samples)
        results['replay'] = True
        
        return results
    
    def start_streaming(self):
        """Start streaming data processing in separate thread."""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._streaming_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        logger.info("Started streaming data processing")
    
    def stop_streaming(self):
        """Stop streaming data processing."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5.0)
        
        logger.info("Stopped streaming data processing")
    
    def add_streaming_sample(self, sample: DatasetSample):
        """Add sample to streaming queue."""
        try:
            self.data_queue.put(sample, timeout=1.0)
        except queue.Full:
            logger.warning("Streaming queue full, dropping sample")
    
    def _streaming_worker(self):
        """Worker thread for processing streaming data."""
        while self.is_streaming:
            try:
                # Get sample from queue with timeout
                sample = self.data_queue.get(timeout=1.0)
                
                # Process sample
                results = self.process_sample(sample, update_model=True)
                
                # Periodic experience replay
                if self.update_count % self.config.training.update_frequency == 0:
                    replay_results = self.experience_replay()
                    if replay_results:
                        logger.debug(f"Experience replay: {replay_results}")
                
                # Log progress periodically
                if self.update_count % 100 == 0:
                    logger.info(f"Processed {self.update_count} samples, "
                               f"current loss: {results['loss']:.4f}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of online learning performance."""
        summary = {
            'total_updates': self.update_count,
            'memory_buffer_size': len(self.memory_buffer),
            'is_streaming': self.is_streaming,
            'queue_size': self.data_queue.qsize() if hasattr(self.data_queue, 'qsize') else 0,
            'current_lr': self.lr_scheduler.current_lr
        }
        
        # Add performance statistics
        for metric, values in self.performance_history.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values[-100:])  # Last 100 values
                summary[f'{metric}_std'] = np.std(values[-100:])
                summary[f'{metric}_trend'] = np.polyfit(range(len(values[-50:])), values[-50:], 1)[0] if len(values) >= 50 else 0
        
        return summary
    
    def save_checkpoint(self, filepath: str):
        """Save model and learner state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'performance_history': dict(self.performance_history),
            'lr_scheduler_state': {
                'current_lr': self.lr_scheduler.current_lr,
                'performance_history': list(self.lr_scheduler.performance_history),
                'lr_history': self.lr_scheduler.lr_history
            }
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model and learner state."""
        checkpoint = torch.load(filepath, map_location=self.model.device if hasattr(self.model, 'device') else 'cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        
        # Restore performance history
        for metric, values in checkpoint['performance_history'].items():
            self.performance_history[metric] = values
        
        # Restore learning rate scheduler
        if 'lr_scheduler_state' in checkpoint:
            lr_state = checkpoint['lr_scheduler_state']
            self.lr_scheduler.current_lr = lr_state['current_lr']
            self.lr_scheduler.performance_history = deque(
                lr_state['performance_history'], 
                maxlen=self.lr_scheduler.adaptation_window
            )
            self.lr_scheduler.lr_history = lr_state['lr_history']
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def reset(self):
        """Reset online learner state."""
        self.memory_buffer = MemoryBuffer(
            max_size=self.config.training.sliding_window_size,
            retention_strategy='importance'
        )
        self.sliding_window.reset()
        self.lr_scheduler.reset()
        self.performance_history.clear()
        self.update_count = 0
        
        logger.info("Reset online learner state")