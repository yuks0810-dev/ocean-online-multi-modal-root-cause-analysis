"""
Performance metrics and statistical analysis for OCEAN model evaluation.
Comprehensive metrics for root cause analysis including ranking metrics and statistical tests.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance metrics for root cause analysis evaluation.
    """
    
    def __init__(self):
        """Initialize performance metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all stored metrics."""
        self.predictions = []
        self.ground_truth = []
        self.prediction_scores = []
        self.timestamps = []
        self.service_metadata = []
    
    def add_predictions(self, 
                       predictions: Union[torch.Tensor, np.ndarray],
                       ground_truth: Union[torch.Tensor, np.ndarray],
                       prediction_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
                       service_metadata: Optional[Dict[str, Any]] = None):
        """
        Add batch of predictions for evaluation.
        
        Args:
            predictions: Binary predictions (0/1)
            ground_truth: Ground truth labels (0/1)
            prediction_scores: Prediction confidence scores [0,1]
            service_metadata: Additional metadata about services
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()
        if prediction_scores is not None and isinstance(prediction_scores, torch.Tensor):
            prediction_scores = prediction_scores.cpu().numpy()
        
        # Store predictions
        self.predictions.extend(predictions.flatten())
        self.ground_truth.extend(ground_truth.flatten())
        
        if prediction_scores is not None:
            self.prediction_scores.extend(prediction_scores.flatten())
        else:
            self.prediction_scores.extend(predictions.flatten())
        
        # Store metadata
        self.timestamps.extend([datetime.now()] * len(predictions.flatten()))
        if service_metadata:
            self.service_metadata.extend([service_metadata] * len(predictions.flatten()))
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """
        Compute standard classification metrics.
        
        Returns:
            Dictionary of classification metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        y_scores = np.array(self.prediction_scores)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Macro and micro averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # ROC AUC (if we have probability scores)
        try:
            if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                metrics['average_precision'] = average_precision_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"Could not compute AUC metrics: {e}")
        
        return metrics
    
    def compute_ranking_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Compute ranking metrics for root cause analysis.
        
        Args:
            k_values: List of k values for top-k metrics
            
        Returns:
            Dictionary of ranking metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        y_true = np.array(self.ground_truth)
        y_scores = np.array(self.prediction_scores)
        
        metrics = {}
        
        # Sort by prediction scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Compute metrics for each k
        for k in k_values:
            if k <= len(sorted_true):
                # Precision@k
                top_k_true = sorted_true[:k]
                precision_at_k = np.sum(top_k_true) / k
                metrics[f'precision@{k}'] = precision_at_k
                
                # Recall@k
                total_relevant = np.sum(y_true)
                if total_relevant > 0:
                    recall_at_k = np.sum(top_k_true) / total_relevant
                    metrics[f'recall@{k}'] = recall_at_k
                
                # Hit rate (whether any relevant item is in top-k)
                hit_rate_at_k = 1.0 if np.sum(top_k_true) > 0 else 0.0
                metrics[f'hit_rate@{k}'] = hit_rate_at_k
        
        # Mean Average Precision (MAP)
        if np.sum(y_true) > 0:
            ap_scores = []
            for i in range(len(sorted_true)):
                if sorted_true[i] == 1:
                    precision_at_i = np.sum(sorted_true[:i+1]) / (i + 1)
                    ap_scores.append(precision_at_i)
            
            if ap_scores:
                metrics['mean_average_precision'] = np.mean(ap_scores)
        
        # Normalized Discounted Cumulative Gain (NDCG)
        def dcg_at_k(scores, k):
            scores = scores[:k]
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
        
        for k in k_values:
            if k <= len(sorted_true):
                dcg_k = dcg_at_k(sorted_true, k)
                ideal_scores = np.sort(y_true)[::-1]
                idcg_k = dcg_at_k(ideal_scores, k)
                
                if idcg_k > 0:
                    ndcg_k = dcg_k / idcg_k
                    metrics[f'ndcg@{k}'] = ndcg_k
        
        return metrics
    
    def compute_temporal_metrics(self, window_size: int = 100) -> Dict[str, List[float]]:
        """
        Compute metrics over time windows for trend analysis.
        
        Args:
            window_size: Size of sliding window
            
        Returns:
            Dictionary of metric time series
        """
        if len(self.predictions) < window_size:
            return {}
        
        metrics_over_time = defaultdict(list)
        
        for i in range(window_size, len(self.predictions) + 1):
            window_pred = self.predictions[i-window_size:i]
            window_true = self.ground_truth[i-window_size:i]
            window_scores = self.prediction_scores[i-window_size:i]
            
            # Compute metrics for this window
            window_acc = accuracy_score(window_true, window_pred)
            window_f1 = f1_score(window_true, window_pred, average='weighted', zero_division=0)
            
            metrics_over_time['accuracy'].append(window_acc)
            metrics_over_time['f1_score'].append(window_f1)
            
            # Add AUC if possible
            try:
                if len(np.unique(window_true)) > 1:
                    window_auc = roc_auc_score(window_true, window_scores)
                    metrics_over_time['roc_auc'].append(window_auc)
            except ValueError:
                pass
        
        return dict(metrics_over_time)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if len(self.predictions) == 0:
            return np.array([])
        
        return confusion_matrix(self.ground_truth, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if len(self.predictions) == 0:
            return "No predictions available"
        
        return classification_report(self.ground_truth, self.predictions)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot comprehensive evaluation metrics.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OCEAN Model Performance Metrics')
        
        # 1. Confusion Matrix
        cm = self.get_confusion_matrix()
        if cm.size > 0:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve (if binary classification)
        from sklearn.metrics import roc_curve
        if len(np.unique(self.ground_truth)) == 2:
            fpr, tpr, _ = roc_curve(self.ground_truth, self.prediction_scores)
            auc = roc_auc_score(self.ground_truth, self.prediction_scores)
            axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        
        # 3. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        if len(np.unique(self.ground_truth)) == 2:
            precision, recall, _ = precision_recall_curve(self.ground_truth, self.prediction_scores)
            ap = average_precision_score(self.ground_truth, self.prediction_scores)
            axes[1, 0].plot(recall, precision, label=f'PR (AP = {ap:.3f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].legend()
        
        # 4. Metrics over time
        temporal_metrics = self.compute_temporal_metrics()
        if temporal_metrics:
            for metric_name, values in temporal_metrics.items():
                axes[1, 1].plot(values, label=metric_name)
            axes[1, 1].set_xlabel('Time Window')
            axes[1, 1].set_ylabel('Metric Value')
            axes[1, 1].set_title('Metrics Over Time')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics plot to {save_path}")
        
        plt.show()


class BaselineComparator:
    """
    Compare OCEAN model performance against baseline methods.
    """
    
    def __init__(self):
        """Initialize baseline comparator."""
        self.baselines = {}
        self.ocean_metrics = None
    
    def add_baseline(self, name: str, metrics: Dict[str, float]):
        """
        Add baseline model results.
        
        Args:
            name: Name of baseline method
            metrics: Performance metrics dictionary
        """
        self.baselines[name] = metrics
        logger.info(f"Added baseline '{name}' with {len(metrics)} metrics")
    
    def set_ocean_metrics(self, metrics: Dict[str, float]):
        """
        Set OCEAN model metrics.
        
        Args:
            metrics: OCEAN performance metrics
        """
        self.ocean_metrics = metrics
        logger.info(f"Set OCEAN metrics with {len(metrics)} values")
    
    def compare_all(self) -> Dict[str, Dict[str, float]]:
        """
        Compare OCEAN against all baselines.
        
        Returns:
            Dictionary of comparison results
        """
        if self.ocean_metrics is None:
            raise ValueError("OCEAN metrics not set")
        
        comparisons = {}
        
        for baseline_name, baseline_metrics in self.baselines.items():
            comparison = self._compare_metrics(self.ocean_metrics, baseline_metrics)
            comparisons[baseline_name] = comparison
        
        return comparisons
    
    def _compare_metrics(self, ocean_metrics: Dict[str, float], 
                        baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare two metric dictionaries."""
        comparison = {}
        
        # Find common metrics
        common_metrics = set(ocean_metrics.keys()) & set(baseline_metrics.keys())
        
        for metric in common_metrics:
            ocean_value = ocean_metrics[metric]
            baseline_value = baseline_metrics[metric]
            
            # Relative improvement
            if baseline_value != 0:
                improvement = (ocean_value - baseline_value) / baseline_value
                comparison[f'{metric}_improvement'] = improvement
            
            # Absolute difference
            comparison[f'{metric}_diff'] = ocean_value - baseline_value
            
            # Ratio
            if baseline_value != 0:
                comparison[f'{metric}_ratio'] = ocean_value / baseline_value
        
        return comparison
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table as DataFrame.
        
        Returns:
            Comparison table
        """
        if self.ocean_metrics is None:
            raise ValueError("OCEAN metrics not set")
        
        # Prepare data for DataFrame
        data = {'OCEAN': self.ocean_metrics}
        data.update(self.baselines)
        
        df = pd.DataFrame(data)
        
        # Add improvement columns
        for baseline_name in self.baselines.keys():
            for metric in df.index:
                if metric in self.ocean_metrics and metric in self.baselines[baseline_name]:
                    ocean_val = self.ocean_metrics[metric]
                    baseline_val = self.baselines[baseline_name][metric]
                    if baseline_val != 0:
                        improvement = (ocean_val - baseline_val) / baseline_val * 100
                        df.loc[metric, f'vs_{baseline_name}_improve_%'] = improvement
        
        return df
    
    def plot_comparison(self, metrics_to_plot: List[str], save_path: Optional[str] = None):
        """
        Plot performance comparison.
        
        Args:
            metrics_to_plot: List of metrics to include in plot
            save_path: Path to save plot
        """
        if self.ocean_metrics is None:
            raise ValueError("OCEAN metrics not set")
        
        # Prepare data
        methods = ['OCEAN'] + list(self.baselines.keys())
        metric_values = defaultdict(list)
        
        for metric in metrics_to_plot:
            # OCEAN value
            metric_values[metric].append(self.ocean_metrics.get(metric, 0))
            
            # Baseline values
            for baseline_name in self.baselines.keys():
                metric_values[metric].append(self.baselines[baseline_name].get(metric, 0))
        
        # Create plot
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(methods)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, method in enumerate(methods):
            values = [metric_values[metric][i] for metric in metrics_to_plot]
            ax.bar(x + i * width, values, width, label=method)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Comparison: OCEAN vs Baselines')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        plt.show()


class StatisticalSignificance:
    """
    Statistical significance testing for model comparisons.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical testing.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def paired_t_test(self, ocean_scores: List[float], 
                     baseline_scores: List[float]) -> Dict[str, Any]:
        """
        Perform paired t-test between OCEAN and baseline.
        
        Args:
            ocean_scores: OCEAN performance scores
            baseline_scores: Baseline performance scores
            
        Returns:
            Test results dictionary
        """
        if len(ocean_scores) != len(baseline_scores):
            raise ValueError("Score lists must have same length")
        
        ocean_array = np.array(ocean_scores)
        baseline_array = np.array(baseline_scores)
        
        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(ocean_array, baseline_array)
        
        # Calculate effect size (Cohen's d)
        differences = ocean_array - baseline_array
        cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        results = {
            'statistic': statistic,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'ocean_mean': np.mean(ocean_array),
            'baseline_mean': np.mean(baseline_array),
            'difference_mean': np.mean(differences),
            'confidence_interval': stats.t.interval(
                1 - self.alpha, 
                len(differences) - 1,
                loc=np.mean(differences),
                scale=stats.sem(differences)
            )
        }
        
        return results
    
    def wilcoxon_test(self, ocean_scores: List[float], 
                     baseline_scores: List[float]) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
        
        Args:
            ocean_scores: OCEAN performance scores
            baseline_scores: Baseline performance scores
            
        Returns:
            Test results dictionary
        """
        if len(ocean_scores) != len(baseline_scores):
            raise ValueError("Score lists must have same length")
        
        ocean_array = np.array(ocean_scores)
        baseline_array = np.array(baseline_scores)
        
        # Perform Wilcoxon test
        statistic, p_value = stats.wilcoxon(ocean_array, baseline_array)
        
        is_significant = p_value < self.alpha
        
        results = {
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'ocean_median': np.median(ocean_array),
            'baseline_median': np.median(baseline_array)
        }
        
        return results
    
    def bootstrap_test(self, ocean_scores: List[float], 
                      baseline_scores: List[float],
                      n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Perform bootstrap test for comparing means.
        
        Args:
            ocean_scores: OCEAN performance scores
            baseline_scores: Baseline performance scores
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Test results dictionary
        """
        ocean_array = np.array(ocean_scores)
        baseline_array = np.array(baseline_scores)
        
        # Observed difference
        observed_diff = np.mean(ocean_array) - np.mean(baseline_array)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        combined_scores = np.concatenate([ocean_array, baseline_array])
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(combined_scores, size=len(combined_scores), replace=True)
            
            # Split into two groups
            group1 = bootstrap_sample[:len(ocean_array)]
            group2 = bootstrap_sample[len(ocean_array):]
            
            bootstrap_diff = np.mean(group1) - np.mean(group2)
            bootstrap_diffs.append(bootstrap_diff)
        
        # Calculate p-value (two-tailed)
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        is_significant = p_value < self.alpha
        
        results = {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'bootstrap_mean': np.mean(bootstrap_diffs),
            'bootstrap_std': np.std(bootstrap_diffs),
            'confidence_interval': np.percentile(bootstrap_diffs, [2.5, 97.5])
        }
        
        return results
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple comparison correction.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Corrected results
        """
        p_array = np.array(p_values)
        
        if method == 'bonferroni':
            corrected_alpha = self.alpha / len(p_values)
            corrected_p = p_array * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1.0
            
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_array)
            corrected_p = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - i
                corrected_p[idx] = min(1.0, p_array[idx] * correction_factor)
                
                # Ensure monotonicity
                if i > 0:
                    corrected_p[idx] = max(corrected_p[idx], corrected_p[sorted_indices[i-1]])
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_array)
            corrected_p = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) / (i + 1)
                corrected_p[idx] = min(1.0, p_array[idx] * correction_factor)
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        significant_tests = corrected_p < self.alpha
        
        results = {
            'original_p_values': p_values,
            'corrected_p_values': corrected_p.tolist(),
            'corrected_alpha': self.alpha,
            'significant_tests': significant_tests.tolist(),
            'method': method,
            'num_significant': np.sum(significant_tests)
        }
        
        return results