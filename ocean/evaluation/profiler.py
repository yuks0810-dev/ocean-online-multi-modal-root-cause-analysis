"""
Model profiling utilities for performance analysis and optimization.
Provides detailed profiling of inference time, memory usage, and computational efficiency.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import psutil
import gc
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
import matplotlib.pyplot as plt
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ModelProfiler:
    """
    Comprehensive model profiling for performance analysis.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model profiler.
        
        Args:
            model: PyTorch model to profile
        """
        self.model = model
        self.profiling_results = {}
        self.hooks = []
        self.layer_stats = defaultdict(list)
        
    def profile_inference(self, 
                         data_loader: DataLoader,
                         num_batches: int = 10,
                         warmup_batches: int = 3) -> Dict[str, Any]:
        """
        Profile model inference performance.
        
        Args:
            data_loader: Data loader for profiling
            num_batches: Number of batches to profile
            warmup_batches: Number of warmup batches
            
        Returns:
            Profiling results dictionary
        """
        logger.info(f"Profiling inference performance over {num_batches} batches")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Warmup
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                if i >= warmup_batches:
                    break
                
                batch_data = self._move_batch_to_device(batch_data, device)
                _ = self.model(
                    batch_data['metrics'],
                    batch_data['graphs'][0],
                    batch_data['logs']
                )
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Profile inference
        inference_times = []
        memory_before = []
        memory_after = []
        
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                batch_data = self._move_batch_to_device(batch_data, device)
                
                # Measure memory before inference
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    memory_before.append(torch.cuda.memory_allocated(device) / 1024**2)  # MB
                else:
                    memory_before.append(psutil.virtual_memory().used / 1024**2)
                
                # Time inference
                start_time = time.perf_counter()
                
                outputs = self.model(
                    batch_data['metrics'],
                    batch_data['graphs'][0],
                    batch_data['logs']
                )
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
                
                # Measure memory after inference
                if device.type == 'cuda':
                    memory_after.append(torch.cuda.memory_allocated(device) / 1024**2)
                else:
                    memory_after.append(psutil.virtual_memory().used / 1024**2)
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        memory_usage = np.array(memory_after) - np.array(memory_before)
        
        batch_sizes = [len(batch_data['metrics']) for batch_data in data_loader]
        avg_batch_size = np.mean(batch_sizes[:num_batches])
        
        results = {
            'num_batches_profiled': num_batches,
            'avg_batch_size': avg_batch_size,
            'inference_times': inference_times.tolist(),
            'avg_inference_time': float(np.mean(inference_times)),
            'std_inference_time': float(np.std(inference_times)),
            'min_inference_time': float(np.min(inference_times)),
            'max_inference_time': float(np.max(inference_times)),
            'throughput': float(avg_batch_size / np.mean(inference_times)),  # samples/second
            'memory_usage_per_batch': memory_usage.tolist(),
            'avg_memory_usage': float(np.mean(memory_usage)),
            'peak_memory_usage': float(np.max(memory_usage)),
            'device': str(device)
        }
        
        # Add percentiles
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            results[f'inference_time_p{p}'] = float(np.percentile(inference_times, p))
        
        self.profiling_results['inference'] = results
        logger.info(f"Inference profiling completed. Avg time: {results['avg_inference_time']:.4f}s")
        
        return results
    
    def profile_layers(self, 
                      sample_input: Tuple[torch.Tensor, ...],
                      num_runs: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Profile individual layer performance.
        
        Args:
            sample_input: Sample input for the model
            num_runs: Number of runs for averaging
            
        Returns:
            Layer-wise profiling results
        """
        logger.info("Profiling individual layers")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Move inputs to device
        sample_input = tuple(x.to(device) if torch.is_tensor(x) else x for x in sample_input)
        
        # Register hooks for layer profiling
        self._register_layer_hooks()
        
        layer_results = {}
        
        try:
            with torch.no_grad():
                for run in range(num_runs):
                    self.layer_stats.clear()
                    
                    start_time = time.perf_counter()
                    _ = self.model(*sample_input)
                    end_time = time.perf_counter()
                    
                    # Record layer timings
                    for layer_name, times in self.layer_stats.items():
                        if layer_name not in layer_results:
                            layer_results[layer_name] = []
                        layer_results[layer_name].extend(times)
        
        finally:
            self._remove_hooks()
        
        # Process layer statistics
        processed_results = {}
        for layer_name, times in layer_results.items():
            times_array = np.array(times)
            processed_results[layer_name] = {
                'avg_time': float(np.mean(times_array)),
                'std_time': float(np.std(times_array)),
                'min_time': float(np.min(times_array)),
                'max_time': float(np.max(times_array)),
                'total_calls': len(times_array)
            }
        
        self.profiling_results['layers'] = processed_results
        logger.info(f"Layer profiling completed for {len(processed_results)} layers")
        
        return processed_results
    
    def profile_memory_efficiency(self, 
                                 data_loader: DataLoader,
                                 num_batches: int = 5) -> Dict[str, Any]:
        """
        Profile memory efficiency and identify potential optimizations.
        
        Args:
            data_loader: Data loader for profiling
            num_batches: Number of batches to analyze
            
        Returns:
            Memory efficiency analysis
        """
        logger.info("Profiling memory efficiency")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        if device.type != 'cuda':
            logger.warning("Memory profiling is most useful on CUDA devices")
        
        memory_timeline = []
        peak_memory_per_batch = []
        
        # Register memory tracking hooks
        memory_tracker = MemoryTracker(device)
        self._register_memory_hooks(memory_tracker)
        
        try:
            with torch.no_grad():
                for i, batch_data in enumerate(data_loader):
                    if i >= num_batches:
                        break
                    
                    batch_data = self._move_batch_to_device(batch_data, device)
                    
                    # Clear cache and measure baseline
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        baseline_memory = torch.cuda.memory_allocated(device)
                    
                    memory_tracker.reset()
                    
                    # Forward pass with memory tracking
                    outputs = self.model(
                        batch_data['metrics'],
                        batch_data['graphs'][0],
                        batch_data['logs']
                    )
                    
                    # Record memory usage
                    batch_memory_timeline = memory_tracker.get_timeline()
                    memory_timeline.extend(batch_memory_timeline)
                    
                    if device.type == 'cuda':
                        peak_memory = torch.cuda.max_memory_allocated(device)
                        peak_memory_per_batch.append((peak_memory - baseline_memory) / 1024**2)  # MB
                        torch.cuda.reset_peak_memory_stats(device)
        
        finally:
            self._remove_hooks()
        
        # Analyze memory usage patterns
        results = {
            'memory_timeline': memory_timeline,
            'peak_memory_per_batch': peak_memory_per_batch,
            'avg_peak_memory': float(np.mean(peak_memory_per_batch)) if peak_memory_per_batch else 0,
            'max_peak_memory': float(np.max(peak_memory_per_batch)) if peak_memory_per_batch else 0,
            'memory_efficiency_score': self._calculate_memory_efficiency_score(memory_timeline),
            'device': str(device)
        }
        
        # Add memory optimization suggestions
        results['optimization_suggestions'] = self._generate_memory_optimization_suggestions(results)
        
        self.profiling_results['memory_efficiency'] = results
        logger.info("Memory efficiency profiling completed")
        
        return results
    
    def generate_profiling_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive profiling report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report content as string
        """
        if not self.profiling_results:
            return "No profiling results available. Run profiling methods first."
        
        report_lines = []
        report_lines.append("# Model Profiling Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Inference profiling
        if 'inference' in self.profiling_results:
            inference = self.profiling_results['inference']
            report_lines.append("## Inference Performance")
            report_lines.append(f"- Average inference time: {inference['avg_inference_time']:.4f}s")
            report_lines.append(f"- Standard deviation: {inference['std_inference_time']:.4f}s")
            report_lines.append(f"- Throughput: {inference['throughput']:.2f} samples/second")
            report_lines.append(f"- Peak memory usage: {inference['peak_memory_usage']:.2f} MB")
            report_lines.append("")
            
            # Add percentiles
            report_lines.append("### Latency Percentiles")
            for key, value in inference.items():
                if key.startswith('inference_time_p'):
                    percentile = key.split('_p')[1]
                    report_lines.append(f"- P{percentile}: {value:.4f}s")
            report_lines.append("")
        
        # Layer profiling
        if 'layers' in self.profiling_results:
            layers = self.profiling_results['layers']
            report_lines.append("## Layer Performance")
            
            # Sort layers by average time
            sorted_layers = sorted(layers.items(), key=lambda x: x[1]['avg_time'], reverse=True)
            
            report_lines.append("### Top 10 Slowest Layers")
            for layer_name, stats in sorted_layers[:10]:
                report_lines.append(f"- {layer_name}: {stats['avg_time']:.4f}s (±{stats['std_time']:.4f})")
            report_lines.append("")
        
        # Memory efficiency
        if 'memory_efficiency' in self.profiling_results:
            memory = self.profiling_results['memory_efficiency']
            report_lines.append("## Memory Efficiency")
            report_lines.append(f"- Average peak memory: {memory['avg_peak_memory']:.2f} MB")
            report_lines.append(f"- Maximum peak memory: {memory['max_peak_memory']:.2f} MB")
            report_lines.append(f"- Memory efficiency score: {memory['memory_efficiency_score']:.3f}")
            report_lines.append("")
            
            if 'optimization_suggestions' in memory:
                report_lines.append("### Optimization Suggestions")
                for suggestion in memory['optimization_suggestions']:
                    report_lines.append(f"- {suggestion}")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Saved profiling report to {save_path}")
        
        return report_content
    
    def plot_profiling_results(self, save_path: Optional[str] = None):
        """Plot profiling results."""
        if not self.profiling_results:
            logger.warning("No profiling results to plot")
            return
        
        num_plots = len(self.profiling_results)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Inference timing histogram
        if 'inference' in self.profiling_results:
            inference = self.profiling_results['inference']
            times = inference['inference_times']
            axes[plot_idx].hist(times, bins=20, alpha=0.7, edgecolor='black')
            axes[plot_idx].set_title('Inference Time Distribution')
            axes[plot_idx].set_xlabel('Time (seconds)')
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.4f}s')
            axes[plot_idx].legend()
            plot_idx += 1
        
        # Layer performance bar chart
        if 'layers' in self.profiling_results:
            layers = self.profiling_results['layers']
            layer_names = list(layers.keys())[:10]  # Top 10
            layer_times = [layers[name]['avg_time'] for name in layer_names]
            
            axes[plot_idx].barh(layer_names, layer_times)
            axes[plot_idx].set_title('Top 10 Layer Execution Times')
            axes[plot_idx].set_xlabel('Time (seconds)')
            plot_idx += 1
        
        # Memory usage timeline
        if 'memory_efficiency' in self.profiling_results:
            memory = self.profiling_results['memory_efficiency']
            if 'memory_timeline' in memory and memory['memory_timeline']:
                timeline = memory['memory_timeline']
                axes[plot_idx].plot(timeline)
                axes[plot_idx].set_title('Memory Usage Timeline')
                axes[plot_idx].set_xlabel('Time Step')
                axes[plot_idx].set_ylabel('Memory (MB)')
                plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved profiling plots to {save_path}")
        
        plt.show()
    
    def _move_batch_to_device(self, batch_data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Move batch data to device."""
        device_batch = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            elif isinstance(value, list):
                device_batch[key] = value
            else:
                device_batch[key] = value
        return device_batch
    
    def _register_layer_hooks(self):
        """Register hooks for layer profiling."""
        def create_hook(name):
            def hook(module, input, output):
                start_time = time.perf_counter()
                # The actual computation happens before this hook
                # So we need to track differently
                end_time = time.perf_counter()
                self.layer_stats[name].append(end_time - start_time)
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = create_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
    
    def _register_memory_hooks(self, memory_tracker):
        """Register hooks for memory tracking."""
        def create_memory_hook(name):
            def hook(module, input, output):
                memory_tracker.record_memory(name)
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = create_memory_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _calculate_memory_efficiency_score(self, memory_timeline: List[float]) -> float:
        """Calculate memory efficiency score (0-1, higher is better)."""
        if not memory_timeline:
            return 0.0
        
        # Simple heuristic: lower variance and lower peak usage is better
        timeline = np.array(memory_timeline)
        peak_memory = np.max(timeline)
        avg_memory = np.mean(timeline)
        memory_variance = np.var(timeline)
        
        # Normalize and combine metrics (this is a simplified approach)
        if peak_memory > 0:
            efficiency = 1.0 - (memory_variance / (peak_memory * avg_memory))
            return max(0.0, min(1.0, efficiency))
        return 0.0
    
    def _generate_memory_optimization_suggestions(self, memory_results: Dict[str, Any]) -> List[str]:
        """Generate memory optimization suggestions."""
        suggestions = []
        
        peak_memory = memory_results.get('max_peak_memory', 0)
        avg_memory = memory_results.get('avg_peak_memory', 0)
        
        if peak_memory > 1000:  # > 1GB
            suggestions.append("Consider using gradient checkpointing to reduce memory usage")
            suggestions.append("Try reducing batch size if memory constraints are an issue")
        
        if peak_memory > avg_memory * 2:
            suggestions.append("High memory variance detected - consider memory pooling")
        
        timeline = memory_results.get('memory_timeline', [])
        if len(timeline) > 100:
            # Check for memory leaks (consistently increasing memory)
            recent_avg = np.mean(timeline[-20:]) if len(timeline) >= 20 else 0
            early_avg = np.mean(timeline[:20]) if len(timeline) >= 20 else 0
            
            if recent_avg > early_avg * 1.2:
                suggestions.append("Potential memory leak detected - check for unreleased tensors")
        
        if not suggestions:
            suggestions.append("Memory usage appears optimal")
        
        return suggestions


class MemoryProfiler:
    """
    Specialized memory profiler for detailed memory analysis.
    """
    
    def __init__(self):
        """Initialize memory profiler."""
        self.memory_snapshots = []
        self.memory_timeline = []
        
    @contextmanager
    def profile_context(self):
        """Context manager for memory profiling."""
        # Start profiling
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_memory = self._get_memory_usage()
        
        try:
            yield self
        finally:
            # End profiling
            end_memory = self._get_memory_usage()
            self.memory_snapshots.append({
                'start_memory': start_memory,
                'end_memory': end_memory,
                'memory_increase': end_memory - start_memory
            })
    
    def profile_memory_usage(self, 
                           operation: Callable[[], Any],
                           num_runs: int = 5) -> Dict[str, Any]:
        """
        Profile memory usage of an operation.
        
        Args:
            operation: Function to profile
            num_runs: Number of runs for averaging
            
        Returns:
            Memory profiling results
        """
        memory_usages = []
        peak_memories = []
        
        for run in range(num_runs):
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            gc.collect()
            
            # Measure memory before
            memory_before = self._get_memory_usage()
            
            # Run operation
            with self.profile_context():
                result = operation()
            
            # Measure memory after
            memory_after = self._get_memory_usage()
            
            memory_usages.append(memory_after - memory_before)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                peak_memories.append(peak_memory)
        
        # Calculate statistics
        results = {
            'avg_memory_usage_mb': float(np.mean(memory_usages)),
            'std_memory_usage_mb': float(np.std(memory_usages)),
            'max_memory_usage_mb': float(np.max(memory_usages)),
            'min_memory_usage_mb': float(np.min(memory_usages)),
            'memory_usage_samples': memory_usages
        }
        
        if peak_memories:
            results.update({
                'avg_peak_memory_mb': float(np.mean(peak_memories)),
                'max_peak_memory_mb': float(np.max(peak_memories)),
                'peak_memory_samples': peak_memories
            })
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            return psutil.virtual_memory().used / 1024**2


class MemoryTracker:
    """
    Memory tracker for detailed timeline analysis.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize memory tracker.
        
        Args:
            device: Device to track memory for
        """
        self.device = device
        self.timeline = []
        self.module_memories = {}
        
    def reset(self):
        """Reset tracking state."""
        self.timeline.clear()
        self.module_memories.clear()
    
    def record_memory(self, module_name: str):
        """Record memory usage at a specific point."""
        if self.device.type == 'cuda':
            memory_mb = torch.cuda.memory_allocated(self.device) / 1024**2
        else:
            memory_mb = psutil.virtual_memory().used / 1024**2
        
        self.timeline.append(memory_mb)
        self.module_memories[module_name] = memory_mb
    
    def get_timeline(self) -> List[float]:
        """Get memory usage timeline."""
        return self.timeline.copy()
    
    def get_module_memories(self) -> Dict[str, float]:
        """Get memory usage by module."""
        return self.module_memories.copy()