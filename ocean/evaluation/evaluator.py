"""
Main evaluator class for comprehensive OCEAN model evaluation.
Orchestrates model evaluation across different scenarios and datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

from ..models.ocean_model import OCEANModel, OCEANVariant
from ..models.training.trainer import Trainer
from ..data import MultiModalDataLoader
from ..configs import OCEANConfig
from .metrics import PerformanceMetrics, BaselineComparator, StatisticalSignificance
from .profiler import ModelProfiler, MemoryProfiler

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for OCEAN model performance assessment.
    """
    
    def __init__(self, 
                 model: Union[OCEANModel, OCEANVariant],
                 config: OCEANConfig,
                 data_loader: MultiModalDataLoader,
                 device: Optional[torch.device] = None):
        """
        Initialize evaluator.
        
        Args:
            model: OCEAN model instance
            config: Model configuration
            data_loader: Data loader for evaluation
            device: Computation device
        """
        self.model = model
        self.config = config
        self.data_loader = data_loader
        
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(device)
        
        # Initialize evaluation components
        self.performance_metrics = PerformanceMetrics()
        self.baseline_comparator = BaselineComparator()
        self.statistical_test = StatisticalSignificance()
        self.model_profiler = ModelProfiler(model)
        self.memory_profiler = MemoryProfiler()
        
        # Evaluation results storage
        self.evaluation_results = {}
        self.evaluation_history = []
        
        logger.info(f"Initialized Evaluator with device: {device}")
    
    def evaluate_model(self, 
                      test_loader: Optional[DataLoader] = None,
                      return_predictions: bool = False,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            test_loader: Test data loader (uses default if None)
            return_predictions: Whether to return detailed predictions
            save_results: Whether to save results to file
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation")
        start_time = time.time()
        
        # Get test loader
        if test_loader is None:
            _, _, test_loader = self.data_loader.get_all_loaders()
        
        # Reset metrics
        self.performance_metrics.reset()
        
        # Set model to evaluation mode
        self.model.eval()
        
        all_predictions = []
        all_ground_truth = []
        all_scores = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                # Move data to device
                batch_data = self._move_batch_to_device(batch_data)
                
                # Forward pass
                outputs = self.model(
                    batch_data['metrics'],
                    batch_data['graphs'][0],  # Assume same graph for batch
                    batch_data['logs'],
                    return_intermediate=True
                )
                
                # Extract predictions and scores
                if 'root_cause_probs' in outputs:
                    predictions = (outputs['root_cause_probs'] > 0.5).float()
                    scores = outputs['root_cause_probs']
                else:
                    # Fallback for different output formats
                    predictions = torch.sigmoid(outputs['logits']) > 0.5
                    scores = torch.sigmoid(outputs['logits'])
                
                # Store results
                batch_predictions = predictions.cpu().numpy()
                batch_ground_truth = batch_data['labels'].cpu().numpy()
                batch_scores = scores.cpu().numpy()
                
                all_predictions.extend(batch_predictions.flatten())
                all_ground_truth.extend(batch_ground_truth.flatten())
                all_scores.extend(batch_scores.flatten())
                
                # Add to performance metrics
                self.performance_metrics.add_predictions(
                    predictions=batch_predictions,
                    ground_truth=batch_ground_truth,
                    prediction_scores=batch_scores,
                    service_metadata={'batch_idx': batch_idx}
                )
                
                # Store detailed results if requested
                if return_predictions:
                    all_metadata.append({
                        'batch_idx': batch_idx,
                        'timestamp': datetime.now(),
                        'batch_size': len(batch_predictions)
                    })
                
                if batch_idx % 50 == 0:
                    logger.debug(f"Processed batch {batch_idx}/{len(test_loader)}")
        
        # Compute comprehensive metrics
        classification_metrics = self.performance_metrics.compute_classification_metrics()
        ranking_metrics = self.performance_metrics.compute_ranking_metrics()
        temporal_metrics = self.performance_metrics.compute_temporal_metrics()
        
        # Profile model performance
        profiling_results = self.model_profiler.profile_inference(test_loader, num_batches=10)
        memory_results = self.memory_profiler.profile_memory_usage(
            lambda: self.model(
                torch.randn(1, self.config.model.temporal_dim).to(self.device),
                batch_data['graphs'][0],
                torch.randn(1, self.config.model.log_dim).to(self.device)
            )
        )
        
        # Compile results
        evaluation_time = time.time() - start_time
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_time_seconds': evaluation_time,
            'model_config': self.config.to_dict(),
            'dataset_info': {
                'test_samples': len(all_predictions),
                'positive_ratio': np.mean(all_ground_truth),
                'num_batches': len(test_loader)
            },
            'classification_metrics': classification_metrics,
            'ranking_metrics': ranking_metrics,
            'temporal_metrics': temporal_metrics,
            'profiling_results': profiling_results,
            'memory_results': memory_results
        }
        
        # Add detailed predictions if requested
        if return_predictions:
            results['detailed_predictions'] = {
                'predictions': all_predictions,
                'ground_truth': all_ground_truth,
                'scores': all_scores,
                'metadata': all_metadata
            }
        
        # Store results
        self.evaluation_results = results
        self.evaluation_history.append(results)
        
        # Save results if requested
        if save_results:
            self.save_evaluation_results(results)
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Test Accuracy: {classification_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Test F1-Score: {classification_metrics.get('f1_score', 0):.4f}")
        
        return results
    
    def evaluate_ablation_study(self, 
                               ablation_configs: List[Dict[str, bool]],
                               test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Perform ablation study by disabling different model components.
        
        Args:
            ablation_configs: List of component disable configurations
            test_loader: Test data loader
            
        Returns:
            Ablation study results
        """
        logger.info("Starting ablation study")
        
        if not isinstance(self.model, OCEANVariant):
            logger.warning("Ablation study requires OCEANVariant model")
            return {}
        
        if test_loader is None:
            _, _, test_loader = self.data_loader.get_all_loaders()
        
        ablation_results = {}
        
        # Store original component states
        original_states = {
            'disable_attention': self.model.disable_attention,
            'disable_graph_fusion': self.model.disable_graph_fusion,
            'disable_temporal_cnn': self.model.disable_temporal_cnn,
            'disable_spatial_gnn': self.model.disable_spatial_gnn
        }
        
        for i, config in enumerate(ablation_configs):
            logger.info(f"Running ablation configuration {i+1}/{len(ablation_configs)}: {config}")
            
            # Set ablation configuration
            for component, disable in config.items():
                if hasattr(self.model, component):
                    setattr(self.model, component, disable)
            
            # Evaluate with this configuration
            results = self.evaluate_model(
                test_loader=test_loader,
                return_predictions=False,
                save_results=False
            )
            
            # Store results
            config_name = "_".join([f"{k}={v}" for k, v in config.items()])
            ablation_results[config_name] = {
                'config': config,
                'metrics': results['classification_metrics'],
                'ranking_metrics': results['ranking_metrics']
            }
        
        # Restore original states
        for component, state in original_states.items():
            setattr(self.model, component, state)
        
        # Analyze ablation results
        ablation_analysis = self._analyze_ablation_results(ablation_results)
        
        final_results = {
            'ablation_results': ablation_results,
            'analysis': ablation_analysis,
            'original_states': original_states
        }
        
        logger.info("Ablation study completed")
        return final_results
    
    def compare_with_baselines(self, 
                              baseline_results: Dict[str, Dict[str, float]],
                              statistical_test: bool = True) -> Dict[str, Any]:
        """
        Compare OCEAN model with baseline methods.
        
        Args:
            baseline_results: Dictionary of baseline method results
            statistical_test: Whether to perform statistical significance tests
            
        Returns:
            Comparison results
        """
        logger.info("Comparing with baseline methods")
        
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        # Set OCEAN metrics for comparison
        ocean_metrics = self.evaluation_results['classification_metrics']
        self.baseline_comparator.set_ocean_metrics(ocean_metrics)
        
        # Add baseline results
        for baseline_name, metrics in baseline_results.items():
            self.baseline_comparator.add_baseline(baseline_name, metrics)
        
        # Perform comparisons
        comparisons = self.baseline_comparator.compare_all()
        comparison_table = self.baseline_comparator.generate_comparison_table()
        
        results = {
            'comparisons': comparisons,
            'comparison_table': comparison_table.to_dict(),
            'ocean_metrics': ocean_metrics,
            'baseline_metrics': baseline_results
        }
        
        # Perform statistical tests if requested
        if statistical_test and hasattr(self, '_baseline_scores'):
            statistical_results = {}
            for baseline_name in baseline_results.keys():
                if baseline_name in self._baseline_scores:
                    ocean_scores = self._ocean_scores  # Assumes this is stored
                    baseline_scores = self._baseline_scores[baseline_name]
                    
                    # Paired t-test
                    t_test = self.statistical_test.paired_t_test(ocean_scores, baseline_scores)
                    # Wilcoxon test
                    wilcoxon_test = self.statistical_test.wilcoxon_test(ocean_scores, baseline_scores)
                    
                    statistical_results[baseline_name] = {
                        't_test': t_test,
                        'wilcoxon_test': wilcoxon_test
                    }
            
            results['statistical_tests'] = statistical_results
        
        logger.info("Baseline comparison completed")
        return results
    
    def evaluate_online_performance(self, 
                                   streaming_data: List[Any],
                                   window_size: int = 100) -> Dict[str, Any]:
        """
        Evaluate model performance in online learning scenario.
        
        Args:
            streaming_data: Stream of data samples
            window_size: Window size for performance tracking
            
        Returns:
            Online performance results
        """
        logger.info("Evaluating online performance")
        
        from ..models.training.online_learner import OnlineLearner
        
        # Initialize online learner
        online_learner = OnlineLearner(
            model=self.model,
            config=self.config
        )
        
        # Track performance over time
        performance_windows = []
        window_predictions = []
        window_ground_truth = []
        
        for i, sample in enumerate(streaming_data):
            # Process sample
            results = online_learner.process_sample(sample, update_model=True)
            
            # Collect for window analysis
            prediction = results.get('prediction', 0)
            ground_truth = sample.label.item() if hasattr(sample.label, 'item') else sample.label
            
            window_predictions.append(prediction)
            window_ground_truth.append(ground_truth)
            
            # Analyze window when full
            if len(window_predictions) >= window_size:
                window_metrics = self._compute_window_metrics(
                    window_predictions, window_ground_truth
                )
                performance_windows.append({
                    'window_end': i,
                    'metrics': window_metrics,
                    'learner_stats': online_learner.get_performance_summary()
                })
                
                # Slide window
                window_predictions = window_predictions[window_size//2:]
                window_ground_truth = window_ground_truth[window_size//2:]
        
        # Compile online performance results
        online_results = {
            'total_samples_processed': len(streaming_data),
            'window_size': window_size,
            'num_windows': len(performance_windows),
            'performance_windows': performance_windows,
            'final_learner_stats': online_learner.get_performance_summary()
        }
        
        # Analyze trends
        if performance_windows:
            accuracy_trend = [w['metrics']['accuracy'] for w in performance_windows]
            f1_trend = [w['metrics']['f1_score'] for w in performance_windows]
            
            online_results['trends'] = {
                'accuracy_trend': accuracy_trend,
                'f1_trend': f1_trend,
                'accuracy_slope': np.polyfit(range(len(accuracy_trend)), accuracy_trend, 1)[0],
                'f1_slope': np.polyfit(range(len(f1_trend)), f1_trend, 1)[0]
            }
        
        logger.info("Online performance evaluation completed")
        return online_results
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report content as string
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        report_lines = []
        report_lines.append("# OCEAN Model Evaluation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model configuration
        report_lines.append("## Model Configuration")
        config_dict = self.evaluation_results['model_config']
        for section, params in config_dict.items():
            report_lines.append(f"### {section}")
            if isinstance(params, dict):
                for key, value in params.items():
                    report_lines.append(f"- {key}: {value}")
            else:
                report_lines.append(f"- {params}")
            report_lines.append("")
        
        # Dataset information
        report_lines.append("## Dataset Information")
        dataset_info = self.evaluation_results['dataset_info']
        for key, value in dataset_info.items():
            report_lines.append(f"- {key}: {value}")
        report_lines.append("")
        
        # Performance metrics
        report_lines.append("## Performance Metrics")
        
        # Classification metrics
        report_lines.append("### Classification Metrics")
        class_metrics = self.evaluation_results['classification_metrics']
        for metric, value in class_metrics.items():
            report_lines.append(f"- {metric}: {value:.4f}")
        report_lines.append("")
        
        # Ranking metrics
        if 'ranking_metrics' in self.evaluation_results:
            report_lines.append("### Ranking Metrics")
            ranking_metrics = self.evaluation_results['ranking_metrics']
            for metric, value in ranking_metrics.items():
                report_lines.append(f"- {metric}: {value:.4f}")
            report_lines.append("")
        
        # Performance profiling
        report_lines.append("## Performance Profiling")
        profiling = self.evaluation_results['profiling_results']
        for key, value in profiling.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"- {key}: {value:.4f}")
            else:
                report_lines.append(f"- {key}: {value}")
        report_lines.append("")
        
        # Memory usage
        report_lines.append("## Memory Usage")
        memory = self.evaluation_results['memory_results']
        for key, value in memory.items():
            report_lines.append(f"- {key}: {value}")
        report_lines.append("")
        
        # Confusion matrix
        report_lines.append("## Classification Report")
        report_lines.append("```")
        report_lines.append(self.performance_metrics.get_classification_report())
        report_lines.append("```")
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Saved evaluation report to {save_path}")
        
        return report_content
    
    def plot_comprehensive_results(self, save_dir: Optional[str] = None):
        """
        Generate comprehensive evaluation plots.
        
        Args:
            save_dir: Directory to save plots
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot performance metrics
        metrics_path = save_dir / "performance_metrics.png" if save_dir else None
        self.performance_metrics.plot_metrics(save_path=metrics_path)
        
        # Plot baseline comparisons if available
        if hasattr(self, '_comparison_results'):
            comparison_path = save_dir / "baseline_comparison.png" if save_dir else None
            self.baseline_comparator.plot_comparison(
                metrics_to_plot=['accuracy', 'f1_score', 'precision', 'recall'],
                save_path=comparison_path
            )
        
        # Plot profiling results
        self._plot_profiling_results(save_dir)
        
        logger.info("Comprehensive plots generated")
    
    def _move_batch_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device."""
        device_batch = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                device_batch[key] = value
            else:
                device_batch[key] = value
        return device_batch
    
    def _analyze_ablation_results(self, ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ablation study results."""
        analysis = {}
        
        # Component importance analysis
        component_impacts = {}
        baseline_metrics = None
        
        # Find baseline (all components enabled)
        for config_name, results in ablation_results.items():
            config = results['config']
            if all(not disable for disable in config.values()):
                baseline_metrics = results['metrics']
                break
        
        if baseline_metrics:
            for config_name, results in ablation_results.items():
                config = results['config']
                metrics = results['metrics']
                
                # Calculate impact of disabling each component
                for component, disabled in config.items():
                    if disabled:
                        impact = {}
                        for metric_name, metric_value in metrics.items():
                            baseline_value = baseline_metrics.get(metric_name, 0)
                            if baseline_value != 0:
                                impact[metric_name] = (baseline_value - metric_value) / baseline_value
                        
                        if component not in component_impacts:
                            component_impacts[component] = []
                        component_impacts[component].append(impact)
        
        analysis['component_impacts'] = component_impacts
        analysis['baseline_metrics'] = baseline_metrics
        
        return analysis
    
    def _compute_window_metrics(self, predictions: List[float], 
                               ground_truth: List[float]) -> Dict[str, float]:
        """Compute metrics for a window of predictions."""
        if len(predictions) == 0:
            return {}
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Convert to binary if needed
        if np.max(predictions) <= 1.0 and np.min(predictions) >= 0.0:
            binary_predictions = (predictions > 0.5).astype(int)
        else:
            binary_predictions = predictions
        
        # Compute basic metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metrics = {
            'accuracy': accuracy_score(ground_truth, binary_predictions),
            'f1_score': f1_score(ground_truth, binary_predictions, average='weighted', zero_division=0),
            'precision': precision_score(ground_truth, binary_predictions, average='weighted', zero_division=0),
            'recall': recall_score(ground_truth, binary_predictions, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def _plot_profiling_results(self, save_dir: Optional[Path]):
        """Plot model profiling results."""
        if 'profiling_results' not in self.evaluation_results:
            return
        
        profiling = self.evaluation_results['profiling_results']
        
        # Create profiling plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Profiling Results')
        
        # Inference time distribution
        if 'inference_times' in profiling:
            times = profiling['inference_times']
            axes[0, 0].hist(times, bins=20, alpha=0.7)
            axes[0, 0].set_title('Inference Time Distribution')
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Frequency')
        
        # Memory usage over time
        memory_results = self.evaluation_results.get('memory_results', {})
        if 'memory_timeline' in memory_results:
            timeline = memory_results['memory_timeline']
            axes[0, 1].plot(timeline)
            axes[0, 1].set_title('Memory Usage Timeline')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Memory (MB)')
        
        # Throughput
        if 'throughput' in profiling:
            throughput = profiling['throughput']
            axes[1, 0].bar(['Throughput'], [throughput])
            axes[1, 0].set_title('Model Throughput')
            axes[1, 0].set_ylabel('Samples/Second')
        
        # Resource utilization summary
        resource_metrics = ['avg_inference_time', 'peak_memory_mb', 'throughput']
        values = [profiling.get(metric, 0) for metric in resource_metrics]
        axes[1, 1].bar(resource_metrics, values)
        axes[1, 1].set_title('Resource Utilization Summary')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / "profiling_results.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                               filepath: Optional[str] = None):
        """Save evaluation results to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ocean_evaluation_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {filepath}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj