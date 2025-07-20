"""
Dataset loading utilities for OCEAN implementation.
Handles RCAEval and LEMMA-RCA dataset formats.
"""

import os
import json
import pickle
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from datetime import datetime

from ..data_types import MetricsData, LogData, TraceData, RootCauseLabels, DatasetDict


logger = logging.getLogger(__name__)


class DatasetLoader:
    """Main dataset loader for RCAEval and LEMMA-RCA datasets."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.supported_formats = ['.csv', '.json', '.pkl', '.parquet']
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    
    def load_dataset(self, dataset_name: str) -> DatasetDict:
        """
        Load a complete dataset by name.
        
        Args:
            dataset_name: Name of the dataset ('rcaeval' or 'lemma-rca')
            
        Returns:
            Dictionary containing all data modalities
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name == 'rcaeval':
            return self._load_rcaeval()
        elif dataset_name == 'lemma-rca':
            return self._load_lemma_rca()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def _load_rcaeval(self) -> DatasetDict:
        """Load RCAEval dataset."""
        rcaeval_path = self.dataset_path / 'rcaeval'
        
        if not rcaeval_path.exists():
            raise FileNotFoundError(f"RCAEval dataset not found at {rcaeval_path}")
        
        logger.info("Loading RCAEval dataset...")
        
        # Load different data modalities
        metrics_data = self._load_metrics_data(rcaeval_path / 'metrics')
        log_data = self._load_log_data(rcaeval_path / 'logs')
        trace_data = self._load_trace_data(rcaeval_path / 'traces')
        labels = self._load_labels(rcaeval_path / 'labels')
        
        return {
            'metrics': metrics_data,
            'logs': log_data,
            'traces': trace_data,
            'labels': labels,
            'dataset_name': 'rcaeval'
        }
    
    def _load_lemma_rca(self) -> DatasetDict:
        """Load LEMMA-RCA dataset."""
        lemma_path = self.dataset_path / 'lemma-rca'
        
        if not lemma_path.exists():
            raise FileNotFoundError(f"LEMMA-RCA dataset not found at {lemma_path}")
        
        logger.info("Loading LEMMA-RCA dataset...")
        
        # Load different data modalities
        metrics_data = self._load_metrics_data(lemma_path / 'metrics')
        log_data = self._load_log_data(lemma_path / 'logs')
        trace_data = self._load_trace_data(lemma_path / 'traces')
        labels = self._load_labels(lemma_path / 'ground_truth')
        
        return {
            'metrics': metrics_data,
            'logs': log_data,
            'traces': trace_data,
            'labels': labels,
            'dataset_name': 'lemma-rca'
        }
    
    def _load_metrics_data(self, metrics_path: Path) -> MetricsData:
        """Load metrics time series data."""
        if not metrics_path.exists():
            logger.warning(f"Metrics path {metrics_path} does not exist")
            return None
        
        # Look for metrics files
        metrics_files = list(metrics_path.glob('*.csv'))
        if not metrics_files:
            metrics_files = list(metrics_path.glob('*.json'))
        if not metrics_files:
            metrics_files = list(metrics_path.glob('*.pkl'))
        
        if not metrics_files:
            logger.warning(f"No metrics files found in {metrics_path}")
            return None
        
        logger.info(f"Loading metrics from {len(metrics_files)} files")
        
        # Combine all metrics files
        all_metrics = []
        for file_path in metrics_files:
            df = self._load_file(file_path)
            if df is not None:
                all_metrics.append(df)
        
        if not all_metrics:
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # Standardize column names
        combined_df = self._standardize_metrics_columns(combined_df)
        
        # Convert timestamp column
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        return MetricsData.from_dataframe(combined_df)
    
    def _load_log_data(self, logs_path: Path) -> LogData:
        """Load log data."""
        if not logs_path.exists():
            logger.warning(f"Logs path {logs_path} does not exist")
            return None
        
        # Look for log files
        log_files = list(logs_path.glob('*.csv'))
        if not log_files:
            log_files = list(logs_path.glob('*.json'))
        if not log_files:
            log_files = list(logs_path.glob('*.pkl'))
        
        if not log_files:
            logger.warning(f"No log files found in {logs_path}")
            return None
        
        logger.info(f"Loading logs from {len(log_files)} files")
        
        # Combine all log files
        all_logs = []
        for file_path in log_files:
            df = self._load_file(file_path)
            if df is not None:
                all_logs.append(df)
        
        if not all_logs:
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_logs, ignore_index=True)
        
        # Standardize column names
        combined_df = self._standardize_log_columns(combined_df)
        
        # Convert timestamp column
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        return LogData.from_dataframe(combined_df)
    
    def _load_trace_data(self, traces_path: Path) -> TraceData:
        """Load distributed tracing data."""
        if not traces_path.exists():
            logger.warning(f"Traces path {traces_path} does not exist")
            return None
        
        # Look for trace files
        trace_files = list(traces_path.glob('*.csv'))
        if not trace_files:
            trace_files = list(traces_path.glob('*.json'))
        if not trace_files:
            trace_files = list(traces_path.glob('*.pkl'))
        
        if not trace_files:
            logger.warning(f"No trace files found in {traces_path}")
            return None
        
        logger.info(f"Loading traces from {len(trace_files)} files")
        
        # Combine all trace files
        all_traces = []
        for file_path in trace_files:
            df = self._load_file(file_path)
            if df is not None:
                all_traces.append(df)
        
        if not all_traces:
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_traces, ignore_index=True)
        
        # Standardize column names
        combined_df = self._standardize_trace_columns(combined_df)
        
        # Convert timestamp column
        if 'start_time' in combined_df.columns:
            combined_df['start_time'] = pd.to_datetime(combined_df['start_time'])
        
        return TraceData.from_dataframe(combined_df)
    
    def _load_labels(self, labels_path: Path) -> RootCauseLabels:
        """Load root cause labels."""
        if not labels_path.exists():
            logger.warning(f"Labels path {labels_path} does not exist")
            return None
        
        # Look for label files
        label_files = list(labels_path.glob('*.csv'))
        if not label_files:
            label_files = list(labels_path.glob('*.json'))
        if not label_files:
            label_files = list(labels_path.glob('*.pkl'))
        
        if not label_files:
            logger.warning(f"No label files found in {labels_path}")
            return None
        
        logger.info(f"Loading labels from {len(label_files)} files")
        
        # Combine all label files
        all_labels = []
        for file_path in label_files:
            df = self._load_file(file_path)
            if df is not None:
                all_labels.append(df)
        
        if not all_labels:
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_labels, ignore_index=True)
        
        # Standardize column names
        combined_df = self._standardize_label_columns(combined_df)
        
        # Convert timestamp column
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        return RootCauseLabels.from_dataframe(combined_df)
    
    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single file based on its extension."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.csv':
                return pd.read_csv(file_path)
            elif suffix == '.json':
                return pd.read_json(file_path)
            elif suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, pd.DataFrame):
                        return data
                    elif isinstance(data, dict):
                        return pd.DataFrame(data)
                    else:
                        logger.warning(f"Unsupported pickle format in {file_path}")
                        return None
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def _standardize_metrics_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize metrics column names to match MetricsData schema."""
        column_mapping = {
            # Common variations for timestamps
            'time': 'timestamp',
            'datetime': 'timestamp',
            'ts': 'timestamp',
            
            # Service identification
            'service': 'service_id',
            'service_name': 'service_id',
            'node': 'service_id',
            'host': 'service_id',
            
            # CPU metrics
            'cpu': 'cpu_usage',
            'cpu_percent': 'cpu_usage',
            'cpu_utilization': 'cpu_usage',
            
            # Memory metrics
            'memory': 'memory_usage',
            'mem': 'memory_usage',
            'memory_percent': 'memory_usage',
            'memory_utilization': 'memory_usage',
            
            # Response time metrics
            'latency': 'response_time',
            'response_latency': 'response_time',
            'avg_response_time': 'response_time',
            
            # Error rate metrics
            'error_count': 'error_rate',
            'errors': 'error_rate',
            'failure_rate': 'error_rate',
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist with default values if missing
        required_columns = ['timestamp', 'service_id', 'cpu_usage', 'memory_usage', 
                           'response_time', 'error_rate']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                elif col == 'service_id':
                    df[col] = 'unknown'
                else:
                    df[col] = 0.0
                logger.warning(f"Missing column '{col}' filled with default values")
        
        return df
    
    def _standardize_log_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize log column names to match LogData schema."""
        column_mapping = {
            'time': 'timestamp',
            'datetime': 'timestamp',
            'ts': 'timestamp',
            'service': 'service_id',
            'service_name': 'service_id',
            'node': 'service_id',
            'level': 'log_level',
            'severity': 'log_level',
            'message': 'log_message',
            'content': 'log_message',
            'text': 'log_message',
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'service_id', 'log_level', 'log_message']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                elif col == 'service_id':
                    df[col] = 'unknown'
                elif col == 'log_level':
                    df[col] = 'INFO'
                elif col == 'log_message':
                    df[col] = 'empty log message'
                logger.warning(f"Missing column '{col}' filled with default values")
        
        return df
    
    def _standardize_trace_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize trace column names to match TraceData schema."""
        column_mapping = {
            'traceId': 'trace_id',
            'spanId': 'span_id',
            'parentSpanId': 'parent_span_id',
            'service': 'service_name',
            'serviceName': 'service_name',
            'operation': 'operation_name',
            'operationName': 'operation_name',
            'startTime': 'start_time',
            'timestamp': 'start_time',
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['trace_id', 'span_id', 'parent_span_id', 'service_name',
                           'operation_name', 'start_time', 'duration']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'trace_id':
                    df[col] = 'unknown_trace'
                elif col == 'span_id':
                    df[col] = 'unknown_span'
                elif col == 'parent_span_id':
                    df[col] = None
                elif col == 'service_name':
                    df[col] = 'unknown_service'
                elif col == 'operation_name':
                    df[col] = 'unknown_operation'
                elif col == 'start_time':
                    df[col] = pd.Timestamp.now()
                elif col == 'duration':
                    df[col] = 0.0
                logger.warning(f"Missing column '{col}' filled with default values")
        
        return df
    
    def _standardize_label_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize label column names to match RootCauseLabels schema."""
        column_mapping = {
            'time': 'timestamp',
            'datetime': 'timestamp',
            'ts': 'timestamp',
            'service': 'service_id',
            'service_name': 'service_id',
            'node': 'service_id',
            'label': 'is_root_cause',
            'root_cause': 'is_root_cause',
            'is_anomaly': 'is_root_cause',
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'service_id', 'is_root_cause']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                elif col == 'service_id':
                    df[col] = 'unknown'
                elif col == 'is_root_cause':
                    df[col] = False
                logger.warning(f"Missing column '{col}' filled with default values")
        
        # Convert boolean labels if needed
        if df['is_root_cause'].dtype == 'object':
            df['is_root_cause'] = df['is_root_cause'].map({
                'true': True, 'True': True, '1': True, 1: True,
                'false': False, 'False': False, '0': False, 0: False
            }).fillna(False)
        
        return df
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        dataset_name = dataset_name.lower()
        dataset_path = self.dataset_path / dataset_name
        
        if not dataset_path.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        info = {
            "name": dataset_name,
            "path": str(dataset_path),
            "modalities": {},
        }
        
        # Check for each modality
        modalities = ["metrics", "logs", "traces", "labels", "ground_truth"]
        for modality in modalities:
            modality_path = dataset_path / modality
            if modality_path.exists():
                files = list(modality_path.glob("*"))
                info["modalities"][modality] = {
                    "exists": True,
                    "num_files": len(files),
                    "files": [f.name for f in files[:5]]  # Show first 5 files
                }
            else:
                info["modalities"][modality] = {"exists": False}
        
        return info