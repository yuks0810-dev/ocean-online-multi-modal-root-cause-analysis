"""
Streaming data handler for real-time OCEAN model processing.
Manages continuous data streams and memory-efficient processing.
"""

import torch
import asyncio
import threading
import queue
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Callable, Generator
import numpy as np
import logging
from datetime import datetime, timedelta
import gc
import psutil
import weakref

from ...data.data_types import DatasetSample, ServiceGraph
from .online_learner import OnlineLearner


logger = logging.getLogger(__name__)


class DataStream:
    """
    Base class for data streams.
    """
    
    def __init__(self, stream_id: str, buffer_size: int = 1000):
        """
        Initialize data stream.
        
        Args:
            stream_id: Unique identifier for the stream
            buffer_size: Maximum buffer size
        """
        self.stream_id = stream_id
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.is_active = False
        self.total_samples = 0
        self.dropped_samples = 0
        
    def add_sample(self, sample: Any) -> bool:
        """
        Add sample to stream buffer.
        
        Returns:
            True if sample was added, False if dropped
        """
        if len(self.buffer) >= self.buffer_size:
            self.dropped_samples += 1
            return False
        
        self.buffer.append(sample)
        self.total_samples += 1
        return True
    
    def get_samples(self, max_samples: Optional[int] = None) -> List[Any]:
        """Get samples from buffer."""
        if max_samples is None:
            samples = list(self.buffer)
            self.buffer.clear()
        else:
            samples = []
            for _ in range(min(max_samples, len(self.buffer))):
                if self.buffer:
                    samples.append(self.buffer.popleft())
        
        return samples
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            'stream_id': self.stream_id,
            'buffer_size': len(self.buffer),
            'max_buffer_size': self.buffer_size,
            'total_samples': self.total_samples,
            'dropped_samples': self.dropped_samples,
            'drop_rate': self.dropped_samples / max(1, self.total_samples),
            'is_active': self.is_active
        }


class MemoryManager:
    """
    Memory manager for streaming data processing.
    """
    
    def __init__(self, 
                 max_memory_usage: float = 0.8,
                 cleanup_threshold: float = 0.9,
                 check_interval: float = 10.0):
        """
        Initialize memory manager.
        
        Args:
            max_memory_usage: Maximum memory usage as fraction of total
            cleanup_threshold: Memory usage threshold for cleanup
            check_interval: Interval for memory checks (seconds)
        """
        self.max_memory_usage = max_memory_usage
        self.cleanup_threshold = cleanup_threshold
        self.check_interval = check_interval
        
        self.managed_objects = []  # WeakSet of managed objects
        self.memory_history = deque(maxlen=100)
        self.is_monitoring = False
        self.monitor_thread = None
        
    def register_object(self, obj: Any):
        """Register object for memory management."""
        self.managed_objects.append(weakref.ref(obj))
    
    def get_memory_usage(self) -> float:
        """Get current memory usage fraction."""
        memory_info = psutil.virtual_memory()
        return memory_info.percent / 100.0
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Started memory monitoring")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped memory monitoring")
    
    def _monitoring_worker(self):
        """Worker thread for memory monitoring."""
        while self.is_monitoring:
            try:
                memory_usage = self.get_memory_usage()
                self.memory_history.append(memory_usage)
                
                if memory_usage > self.cleanup_threshold:
                    logger.warning(f"High memory usage: {memory_usage:.2%}")
                    self.cleanup()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
    
    def cleanup(self):
        """Perform memory cleanup."""
        logger.info("Performing memory cleanup")
        
        # Clean up managed objects
        alive_objects = []
        for obj_ref in self.managed_objects:
            obj = obj_ref()
            if obj is not None:
                if hasattr(obj, 'cleanup'):
                    obj.cleanup()
                alive_objects.append(obj_ref)
        
        self.managed_objects = alive_objects
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage after cleanup
        memory_usage = self.get_memory_usage()
        logger.info(f"Memory usage after cleanup: {memory_usage:.2%}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        current_usage = self.get_memory_usage()
        
        stats = {
            'current_usage': current_usage,
            'max_usage': self.max_memory_usage,
            'cleanup_threshold': self.cleanup_threshold,
            'managed_objects_count': len(self.managed_objects),
            'is_monitoring': self.is_monitoring
        }
        
        if self.memory_history:
            stats.update({
                'avg_usage': np.mean(self.memory_history),
                'max_recent_usage': np.max(self.memory_history),
                'usage_trend': np.polyfit(range(len(self.memory_history)), 
                                        list(self.memory_history), 1)[0] if len(self.memory_history) > 1 else 0
            })
        
        return stats


class StreamProcessor:
    """
    Processor for handling multiple data streams concurrently.
    """
    
    def __init__(self,
                 online_learner: OnlineLearner,
                 max_streams: int = 10,
                 processing_interval: float = 1.0,
                 batch_size: int = 32):
        """
        Initialize stream processor.
        
        Args:
            online_learner: Online learner instance
            max_streams: Maximum number of concurrent streams
            processing_interval: Interval between processing cycles (seconds)
            batch_size: Batch size for processing
        """
        self.online_learner = online_learner
        self.max_streams = max_streams
        self.processing_interval = processing_interval
        self.batch_size = batch_size
        
        self.streams = {}  # stream_id -> DataStream
        self.processing_queue = queue.PriorityQueue()
        self.is_processing = False
        self.processor_thread = None
        
        # Statistics
        self.processed_samples = 0
        self.processing_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Memory management
        self.memory_manager = MemoryManager()
        self.memory_manager.register_object(self)
        
    def add_stream(self, stream: DataStream) -> bool:
        """
        Add new data stream.
        
        Returns:
            True if stream was added, False if rejected
        """
        if len(self.streams) >= self.max_streams:
            logger.warning(f"Maximum streams ({self.max_streams}) reached")
            return False
        
        if stream.stream_id in self.streams:
            logger.warning(f"Stream {stream.stream_id} already exists")
            return False
        
        self.streams[stream.stream_id] = stream
        stream.is_active = True
        
        logger.info(f"Added stream {stream.stream_id}")
        return True
    
    def remove_stream(self, stream_id: str) -> bool:
        """Remove data stream."""
        if stream_id not in self.streams:
            return False
        
        stream = self.streams[stream_id]
        stream.is_active = False
        del self.streams[stream_id]
        
        logger.info(f"Removed stream {stream_id}")
        return True
    
    def start_processing(self):
        """Start stream processing."""
        if self.is_processing:
            logger.warning("Stream processing already active")
            return
        
        self.is_processing = True
        self.processor_thread = threading.Thread(target=self._processing_worker)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        # Start memory monitoring
        self.memory_manager.start_monitoring()
        
        logger.info("Started stream processing")
    
    def stop_processing(self):
        """Stop stream processing."""
        self.is_processing = False
        
        if self.processor_thread:
            self.processor_thread.join(timeout=10.0)
        
        # Stop memory monitoring
        self.memory_manager.stop_monitoring()
        
        logger.info("Stopped stream processing")
    
    def _processing_worker(self):
        """Worker thread for processing streams."""
        while self.is_processing:
            try:
                start_time = time.time()
                
                # Collect samples from all active streams
                all_samples = []
                for stream in self.streams.values():
                    if stream.is_active:
                        samples = stream.get_samples(self.batch_size)
                        all_samples.extend(samples)
                
                # Process samples if available
                if all_samples:
                    processed_count = self._process_samples(all_samples)
                    self.processed_samples += processed_count
                    
                    # Update throughput
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    if processing_time > 0:
                        throughput = processed_count / processing_time
                        self.throughput_history.append(throughput)
                
                # Sleep until next processing cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.processing_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in stream processing: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _process_samples(self, samples: List[DatasetSample]) -> int:
        """Process a batch of samples."""
        processed_count = 0
        
        # Process samples in batches
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            
            try:
                # Process batch through online learner
                results = self.online_learner.process_batch(batch)
                processed_count += len(batch)
                
                # Log batch processing results
                if processed_count % 100 == 0:
                    logger.debug(f"Processed batch: {results}")
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
        
        return processed_count
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = {
            'total_processed_samples': self.processed_samples,
            'active_streams': len([s for s in self.streams.values() if s.is_active]),
            'total_streams': len(self.streams),
            'is_processing': self.is_processing,
            'memory_stats': self.memory_manager.get_memory_stats()
        }
        
        # Processing performance
        if self.processing_times:
            stats.update({
                'avg_processing_time': np.mean(self.processing_times),
                'max_processing_time': np.max(self.processing_times),
                'processing_time_std': np.std(self.processing_times)
            })
        
        if self.throughput_history:
            stats.update({
                'avg_throughput': np.mean(self.throughput_history),
                'max_throughput': np.max(self.throughput_history),
                'current_throughput': self.throughput_history[-1] if self.throughput_history else 0
            })
        
        # Stream statistics
        stream_stats = {}
        for stream_id, stream in self.streams.items():
            stream_stats[stream_id] = stream.get_stats()
        stats['streams'] = stream_stats
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        # Clear processing queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear stream buffers
        for stream in self.streams.values():
            stream.buffer.clear()
        
        # Clear processing history
        self.processing_times.clear()
        self.throughput_history.clear()
        
        logger.info("Cleaned up stream processor resources")


class StreamingHandler:
    """
    Main streaming handler that coordinates all streaming components.
    """
    
    def __init__(self, 
                 online_learner: OnlineLearner,
                 max_concurrent_streams: int = 10,
                 processing_interval: float = 1.0,
                 enable_memory_management: bool = True):
        """
        Initialize streaming handler.
        
        Args:
            online_learner: Online learner instance
            max_concurrent_streams: Maximum concurrent streams
            processing_interval: Processing interval in seconds
            enable_memory_management: Enable automatic memory management
        """
        self.online_learner = online_learner
        self.max_concurrent_streams = max_concurrent_streams
        self.processing_interval = processing_interval
        self.enable_memory_management = enable_memory_management
        
        # Initialize components
        self.stream_processor = StreamProcessor(
            online_learner=online_learner,
            max_streams=max_concurrent_streams,
            processing_interval=processing_interval
        )
        
        # Stream management
        self.stream_callbacks = {}  # stream_id -> callback functions
        self.is_active = False
        
        # Performance monitoring
        self.start_time = None
        self.total_processed = 0
        
        logger.info("Initialized StreamingHandler")
    
    def start(self):
        """Start streaming data handling."""
        if self.is_active:
            logger.warning("Streaming handler already active")
            return
        
        self.is_active = True
        self.start_time = datetime.now()
        
        # Start stream processor
        self.stream_processor.start_processing()
        
        logger.info("Started streaming data handling")
    
    def stop(self):
        """Stop streaming data handling."""
        self.is_active = False
        
        # Stop stream processor
        self.stream_processor.stop_processing()
        
        logger.info("Stopped streaming data handling")
    
    def create_stream(self, 
                     stream_id: str, 
                     buffer_size: int = 1000,
                     callback: Optional[Callable] = None) -> bool:
        """
        Create new data stream.
        
        Args:
            stream_id: Unique stream identifier
            buffer_size: Stream buffer size
            callback: Optional callback function for stream events
            
        Returns:
            True if stream was created successfully
        """
        stream = DataStream(stream_id, buffer_size)
        
        if self.stream_processor.add_stream(stream):
            if callback:
                self.stream_callbacks[stream_id] = callback
            return True
        
        return False
    
    def remove_stream(self, stream_id: str) -> bool:
        """Remove data stream."""
        success = self.stream_processor.remove_stream(stream_id)
        
        if success and stream_id in self.stream_callbacks:
            del self.stream_callbacks[stream_id]
        
        return success
    
    def add_sample_to_stream(self, stream_id: str, sample: DatasetSample) -> bool:
        """
        Add sample to specified stream.
        
        Args:
            stream_id: Target stream ID
            sample: Data sample to add
            
        Returns:
            True if sample was added successfully
        """
        if stream_id not in self.stream_processor.streams:
            logger.warning(f"Stream {stream_id} not found")
            return False
        
        stream = self.stream_processor.streams[stream_id]
        success = stream.add_sample(sample)
        
        # Call callback if available
        if stream_id in self.stream_callbacks:
            try:
                self.stream_callbacks[stream_id](stream_id, sample, success)
            except Exception as e:
                logger.error(f"Error in stream callback: {e}")
        
        if success:
            self.total_processed += 1
        
        return success
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the streaming handler."""
        stats = {
            'handler_stats': {
                'is_active': self.is_active,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'total_processed': self.total_processed,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            },
            'processor_stats': self.stream_processor.get_processing_stats(),
            'learner_stats': self.online_learner.get_performance_summary()
        }
        
        # Calculate overall throughput
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
            if uptime > 0:
                stats['handler_stats']['avg_throughput'] = self.total_processed / uptime
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on streaming components."""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': []
        }
        
        # Check if handler is active
        if not self.is_active:
            health['issues'].append('Streaming handler not active')
            health['status'] = 'unhealthy'
        
        # Check memory usage
        memory_stats = self.stream_processor.memory_manager.get_memory_stats()
        if memory_stats['current_usage'] > memory_stats['cleanup_threshold']:
            health['warnings'].append(f"High memory usage: {memory_stats['current_usage']:.2%}")
        
        # Check stream health
        processing_stats = self.stream_processor.get_processing_stats()
        
        # Check for dropped samples
        total_dropped = 0
        for stream_stats in processing_stats.get('streams', {}).values():
            total_dropped += stream_stats.get('dropped_samples', 0)
        
        if total_dropped > 0:
            health['warnings'].append(f"Total dropped samples: {total_dropped}")
        
        # Check processing performance
        if 'avg_throughput' in processing_stats and processing_stats['avg_throughput'] < 1.0:
            health['warnings'].append("Low processing throughput")
        
        return health
    
    def save_state(self, filepath: str):
        """Save streaming handler state."""
        state = {
            'total_processed': self.total_processed,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stream_configs': {
                stream_id: {
                    'buffer_size': stream.buffer_size,
                    'total_samples': stream.total_samples,
                    'dropped_samples': stream.dropped_samples
                }
                for stream_id, stream in self.stream_processor.streams.items()
            },
            'processing_stats': self.stream_processor.get_processing_stats()
        }
        
        # Save online learner state
        learner_path = filepath.replace('.json', '_learner.pt')
        self.online_learner.save_checkpoint(learner_path)
        
        # Save handler state
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved streaming handler state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load streaming handler state."""
        import json
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.total_processed = state.get('total_processed', 0)
        
        if state.get('start_time'):
            self.start_time = datetime.fromisoformat(state['start_time'])
        
        # Load online learner state
        learner_path = filepath.replace('.json', '_learner.pt')
        try:
            self.online_learner.load_checkpoint(learner_path)
        except FileNotFoundError:
            logger.warning(f"Learner checkpoint not found: {learner_path}")
        
        logger.info(f"Loaded streaming handler state from {filepath}")