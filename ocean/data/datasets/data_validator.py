"""
Data validation utilities for OCEAN implementation.
Ensures data integrity and completeness for multi-modal datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

from ..data_types import MetricsData, LogData, TraceData, RootCauseLabels, DatasetDict


logger = logging.getLogger(__name__)


class DataValidator:
    """Validates multi-modal datasets for OCEAN training."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize data validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation errors.
                        If False, logs warnings and continues.
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
    
    def validate_dataset(self, dataset: DatasetDict) -> Dict[str, Any]:
        """
        Validate complete dataset with all modalities.
        
        Args:
            dataset: Dictionary containing all data modalities
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
            "summary": {}
        }
        
        # Validate each modality
        if dataset.get('metrics'):
            metrics_results = self.validate_metrics_data(dataset['metrics'])
            results["metrics"]["metrics"] = metrics_results
            if not metrics_results["valid"]:
                results["valid"] = False
                results["errors"].extend(metrics_results["errors"])
            results["warnings"].extend(metrics_results["warnings"])
        
        if dataset.get('logs'):
            logs_results = self.validate_log_data(dataset['logs'])
            results["metrics"]["logs"] = logs_results
            if not logs_results["valid"]:
                results["valid"] = False
                results["errors"].extend(logs_results["errors"])
            results["warnings"].extend(logs_results["warnings"])
        
        if dataset.get('traces'):
            traces_results = self.validate_trace_data(dataset['traces'])
            results["metrics"]["traces"] = traces_results
            if not traces_results["valid"]:
                results["valid"] = False
                results["errors"].extend(traces_results["errors"])
            results["warnings"].extend(traces_results["warnings"])
        
        if dataset.get('labels'):
            labels_results = self.validate_labels_data(dataset['labels'])
            results["metrics"]["labels"] = labels_results
            if not labels_results["valid"]:
                results["valid"] = False
                results["errors"].extend(labels_results["errors"])
            results["warnings"].extend(labels_results["warnings"])
        
        # Cross-modality validation
        cross_results = self.validate_cross_modality_consistency(dataset)
        results["metrics"]["cross_modality"] = cross_results
        if not cross_results["valid"]:
            results["valid"] = False
            results["errors"].extend(cross_results["errors"])
        results["warnings"].extend(cross_results["warnings"])
        
        # Generate summary
        results["summary"] = self._generate_summary(dataset, results)
        
        self.validation_results = results
        return results
    
    def validate_metrics_data(self, metrics: MetricsData) -> Dict[str, Any]:
        """Validate metrics time series data."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check data completeness
            if not metrics.timestamp:
                results["errors"].append("No timestamp data found")
                results["valid"] = False
            
            if not metrics.service_id:
                results["errors"].append("No service_id data found")
                results["valid"] = False
            
            # Check data lengths consistency
            lengths = [
                len(metrics.timestamp),
                len(metrics.service_id),
                len(metrics.cpu_usage),
                len(metrics.memory_usage),
                len(metrics.response_time),
                len(metrics.error_rate)
            ]
            
            if len(set(lengths)) > 1:
                results["errors"].append(f"Inconsistent data lengths: {lengths}")
                results["valid"] = False
            
            num_records = lengths[0] if lengths else 0
            results["stats"]["num_records"] = num_records
            
            if num_records == 0:
                results["errors"].append("No metrics records found")
                results["valid"] = False
                return results
            
            # Check for missing values
            metrics_df = metrics.to_dataframe()
            missing_counts = metrics_df.isnull().sum()
            if missing_counts.sum() > 0:
                results["warnings"].append(f"Missing values found: {missing_counts.to_dict()}")
            
            # Check timestamp ordering
            timestamps = pd.to_datetime(metrics.timestamp)
            if not timestamps.is_monotonic_increasing:
                results["warnings"].append("Timestamps are not in chronological order")
            
            # Check value ranges
            if any(x < 0 or x > 100 for x in metrics.cpu_usage if x is not None):
                results["warnings"].append("CPU usage values outside expected range [0, 100]")
            
            if any(x < 0 or x > 100 for x in metrics.memory_usage if x is not None):
                results["warnings"].append("Memory usage values outside expected range [0, 100]")
            
            if any(x < 0 for x in metrics.response_time if x is not None):
                results["warnings"].append("Negative response time values found")
            
            if any(x < 0 or x > 1 for x in metrics.error_rate if x is not None):
                results["warnings"].append("Error rate values outside expected range [0, 1]")
            
            # Statistics
            results["stats"]["unique_services"] = len(set(metrics.service_id))
            results["stats"]["time_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
            }
            
        except Exception as e:
            results["errors"].append(f"Validation error: {str(e)}")
            results["valid"] = False
        
        return results
    
    def validate_log_data(self, logs: LogData) -> Dict[str, Any]:
        """Validate log data."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check data completeness
            if not logs.timestamp:
                results["errors"].append("No timestamp data found in logs")
                results["valid"] = False
            
            if not logs.service_id:
                results["errors"].append("No service_id data found in logs")
                results["valid"] = False
            
            if not logs.log_message:
                results["errors"].append("No log messages found")
                results["valid"] = False
            
            # Check data lengths consistency
            lengths = [
                len(logs.timestamp),
                len(logs.service_id),
                len(logs.log_level),
                len(logs.log_message)
            ]
            
            if len(set(lengths)) > 1:
                results["errors"].append(f"Inconsistent log data lengths: {lengths}")
                results["valid"] = False
            
            num_records = lengths[0] if lengths else 0
            results["stats"]["num_records"] = num_records
            
            if num_records == 0:
                results["errors"].append("No log records found")
                results["valid"] = False
                return results
            
            # Check for empty log messages
            empty_messages = sum(1 for msg in logs.log_message if not msg or msg.strip() == '')
            if empty_messages > 0:
                results["warnings"].append(f"Found {empty_messages} empty log messages")
            
            # Check log levels
            valid_levels = {'DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL'}
            invalid_levels = set(logs.log_level) - valid_levels
            if invalid_levels:
                results["warnings"].append(f"Invalid log levels found: {invalid_levels}")
            
            # Statistics
            results["stats"]["unique_services"] = len(set(logs.service_id))
            results["stats"]["log_levels"] = {level: logs.log_level.count(level) for level in set(logs.log_level)}
            
            timestamps = pd.to_datetime(logs.timestamp)
            results["stats"]["time_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
            }
            
        except Exception as e:
            results["errors"].append(f"Log validation error: {str(e)}")
            results["valid"] = False
        
        return results
    
    def validate_trace_data(self, traces: TraceData) -> Dict[str, Any]:
        """Validate distributed tracing data."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check data completeness
            if not traces.trace_id:
                results["errors"].append("No trace_id data found")
                results["valid"] = False
            
            if not traces.span_id:
                results["errors"].append("No span_id data found")
                results["valid"] = False
            
            if not traces.service_name:
                results["errors"].append("No service_name data found")
                results["valid"] = False
            
            # Check data lengths consistency
            lengths = [
                len(traces.trace_id),
                len(traces.span_id),
                len(traces.parent_span_id),
                len(traces.service_name),
                len(traces.operation_name),
                len(traces.start_time),
                len(traces.duration)
            ]
            
            if len(set(lengths)) > 1:
                results["errors"].append(f"Inconsistent trace data lengths: {lengths}")
                results["valid"] = False
            
            num_records = lengths[0] if lengths else 0
            results["stats"]["num_records"] = num_records
            
            if num_records == 0:
                results["errors"].append("No trace records found")
                results["valid"] = False
                return results
            
            # Check for negative durations
            negative_durations = sum(1 for d in traces.duration if d < 0)
            if negative_durations > 0:
                results["warnings"].append(f"Found {negative_durations} negative duration spans")
            
            # Check parent-child relationships
            span_ids = set(traces.span_id)
            invalid_parents = [pid for pid in traces.parent_span_id 
                             if pid is not None and pid not in span_ids]
            if invalid_parents:
                results["warnings"].append(f"Found {len(invalid_parents)} spans with invalid parent references")
            
            # Statistics
            results["stats"]["unique_traces"] = len(set(traces.trace_id))
            results["stats"]["unique_services"] = len(set(traces.service_name))
            results["stats"]["unique_operations"] = len(set(traces.operation_name))
            
            timestamps = pd.to_datetime(traces.start_time)
            results["stats"]["time_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
            }
            
        except Exception as e:
            results["errors"].append(f"Trace validation error: {str(e)}")
            results["valid"] = False
        
        return results
    
    def validate_labels_data(self, labels: RootCauseLabels) -> Dict[str, Any]:
        """Validate root cause labels."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check data completeness
            if not labels.timestamp:
                results["errors"].append("No timestamp data found in labels")
                results["valid"] = False
            
            if not labels.service_id:
                results["errors"].append("No service_id data found in labels")
                results["valid"] = False
            
            if not labels.is_root_cause:
                results["errors"].append("No root cause labels found")
                results["valid"] = False
            
            # Check data lengths consistency
            lengths = [
                len(labels.timestamp),
                len(labels.service_id),
                len(labels.is_root_cause)
            ]
            
            if len(set(lengths)) > 1:
                results["errors"].append(f"Inconsistent label data lengths: {lengths}")
                results["valid"] = False
            
            num_records = lengths[0] if lengths else 0
            results["stats"]["num_records"] = num_records
            
            if num_records == 0:
                results["errors"].append("No label records found")
                results["valid"] = False
                return results
            
            # Check label distribution
            positive_labels = sum(labels.is_root_cause)
            negative_labels = len(labels.is_root_cause) - positive_labels
            
            results["stats"]["label_distribution"] = {
                "positive": positive_labels,
                "negative": negative_labels,
                "positive_ratio": positive_labels / len(labels.is_root_cause)
            }
            
            # Warn about severe class imbalance
            if positive_labels == 0:
                results["errors"].append("No positive labels found")
                results["valid"] = False
            elif positive_labels / len(labels.is_root_cause) < 0.01:
                results["warnings"].append("Severe class imbalance: less than 1% positive labels")
            
            # Statistics
            results["stats"]["unique_services"] = len(set(labels.service_id))
            
            timestamps = pd.to_datetime(labels.timestamp)
            results["stats"]["time_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
            }
            
        except Exception as e:
            results["errors"].append(f"Labels validation error: {str(e)}")
            results["valid"] = False
        
        return results
    
    def validate_cross_modality_consistency(self, dataset: DatasetDict) -> Dict[str, Any]:
        """Validate consistency across different data modalities."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Collect all service IDs from different modalities
            all_services = {}
            
            if dataset.get('metrics'):
                all_services['metrics'] = set(dataset['metrics'].service_id)
            
            if dataset.get('logs'):
                all_services['logs'] = set(dataset['logs'].service_id)
            
            if dataset.get('traces'):
                all_services['traces'] = set(dataset['traces'].service_name)
            
            if dataset.get('labels'):
                all_services['labels'] = set(dataset['labels'].service_id)
            
            if len(all_services) < 2:
                results["warnings"].append("Insufficient modalities for cross-validation")
                return results
            
            # Check service consistency
            service_lists = list(all_services.values())
            common_services = service_lists[0]
            for services in service_lists[1:]:
                common_services = common_services.intersection(services)
            
            results["stats"]["service_consistency"] = {
                "total_services_per_modality": {k: len(v) for k, v in all_services.items()},
                "common_services": len(common_services),
                "common_service_list": list(common_services)[:10]  # Show first 10
            }
            
            if len(common_services) == 0:
                results["errors"].append("No common services found across modalities")
                results["valid"] = False
            elif len(common_services) < min(len(s) for s in service_lists) * 0.5:
                results["warnings"].append("Less than 50% service overlap between modalities")
            
            # Check time range consistency
            time_ranges = {}
            
            if dataset.get('metrics'):
                timestamps = pd.to_datetime(dataset['metrics'].timestamp)
                time_ranges['metrics'] = (min(timestamps), max(timestamps))
            
            if dataset.get('logs'):
                timestamps = pd.to_datetime(dataset['logs'].timestamp)
                time_ranges['logs'] = (min(timestamps), max(timestamps))
            
            if dataset.get('traces'):
                timestamps = pd.to_datetime(dataset['traces'].start_time)
                time_ranges['traces'] = (min(timestamps), max(timestamps))
            
            if dataset.get('labels'):
                timestamps = pd.to_datetime(dataset['labels'].timestamp)
                time_ranges['labels'] = (min(timestamps), max(timestamps))
            
            if len(time_ranges) > 1:
                all_starts = [start for start, _ in time_ranges.values()]
                all_ends = [end for _, end in time_ranges.values()]
                
                overall_start = min(all_starts)
                overall_end = max(all_ends)
                
                # Check for significant time gaps
                for modality, (start, end) in time_ranges.items():
                    if (start - overall_start).total_seconds() > 3600:  # More than 1 hour gap
                        results["warnings"].append(f"{modality} data starts {start - overall_start} after earliest data")
                    
                    if (overall_end - end).total_seconds() > 3600:  # More than 1 hour gap
                        results["warnings"].append(f"{modality} data ends {overall_end - end} before latest data")
                
                results["stats"]["time_consistency"] = {
                    "time_ranges": {k: {"start": v[0].isoformat(), "end": v[1].isoformat()} 
                                   for k, v in time_ranges.items()},
                    "overall_range": {
                        "start": overall_start.isoformat(),
                        "end": overall_end.isoformat(),
                        "duration_hours": (overall_end - overall_start).total_seconds() / 3600
                    }
                }
            
        except Exception as e:
            results["errors"].append(f"Cross-modality validation error: {str(e)}")
            results["valid"] = False
        
        return results
    
    def _generate_summary(self, dataset: DatasetDict, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        summary = {
            "overall_valid": results["valid"],
            "total_errors": len(results["errors"]),
            "total_warnings": len(results["warnings"]),
            "modalities_validated": list(results["metrics"].keys()),
            "dataset_name": dataset.get("dataset_name", "unknown")
        }
        
        # Aggregate statistics
        total_records = 0
        for modality, modality_results in results["metrics"].items():
            if modality != "cross_modality" and "stats" in modality_results:
                if "num_records" in modality_results["stats"]:
                    total_records += modality_results["stats"]["num_records"]
        
        summary["total_records"] = total_records
        
        # Recommendation
        if results["valid"]:
            if len(results["warnings"]) == 0:
                summary["recommendation"] = "Dataset is valid and ready for training"
            else:
                summary["recommendation"] = "Dataset is valid but has warnings - review before training"
        else:
            summary["recommendation"] = "Dataset has errors - fix before training"
        
        return summary
    
    def print_validation_report(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Print a formatted validation report."""
        if results is None:
            results = self.validation_results
        
        if not results:
            print("No validation results available")
            return
        
        print("="*60)
        print("OCEAN Dataset Validation Report")
        print("="*60)
        
        summary = results.get("summary", {})
        print(f"Dataset: {summary.get('dataset_name', 'Unknown')}")
        print(f"Overall Status: {'✓ VALID' if summary.get('overall_valid') else '✗ INVALID'}")
        print(f"Total Records: {summary.get('total_records', 0):,}")
        print(f"Errors: {summary.get('total_errors', 0)}")
        print(f"Warnings: {summary.get('total_warnings', 0)}")
        print()
        
        # Print errors
        if results.get("errors"):
            print("ERRORS:")
            for i, error in enumerate(results["errors"], 1):
                print(f"  {i}. {error}")
            print()
        
        # Print warnings
        if results.get("warnings"):
            print("WARNINGS:")
            for i, warning in enumerate(results["warnings"], 1):
                print(f"  {i}. {warning}")
            print()
        
        # Print modality details
        print("MODALITY DETAILS:")
        for modality, modality_results in results.get("metrics", {}).items():
            if modality == "cross_modality":
                continue
                
            status = "✓" if modality_results.get("valid") else "✗"
            num_records = modality_results.get("stats", {}).get("num_records", 0)
            print(f"  {modality.capitalize()}: {status} ({num_records:,} records)")
        print()
        
        print(f"Recommendation: {summary.get('recommendation', 'No recommendation available')}")
        print("="*60)