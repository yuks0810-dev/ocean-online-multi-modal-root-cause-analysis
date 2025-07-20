"""
Main data processing pipeline for OCEAN implementation.
Coordinates loading, validation, and preprocessing of multi-modal data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd

from ..data_types import DatasetDict, MetricsData, LogData, TraceData, RootCauseLabels
from ..datasets import DatasetLoader, DataValidator
from ...configs import OCEANConfig


logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main data processing pipeline for OCEAN.
    Handles multi-modal data preprocessing and synchronization.
    """
    
    def __init__(self, config: OCEANConfig):
        """
        Initialize data processor.
        
        Args:
            config: OCEAN configuration object
        """
        self.config = config
        self.dataset_loader = DatasetLoader(config.data.dataset_path)
        self.data_validator = DataValidator(strict_mode=False)
        
        self.raw_data = {}
        self.processed_data = {}
        self.validation_results = {}
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.system.log_level.upper()))
    
    def load_datasets(self, dataset_names: List[str]) -> Dict[str, DatasetDict]:
        """
        Load multiple datasets by name.
        
        Args:
            dataset_names: List of dataset names to load ('rcaeval', 'lemma-rca')
            
        Returns:
            Dictionary mapping dataset names to loaded data
        """
        loaded_datasets = {}
        
        for dataset_name in dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")
            
            try:
                dataset = self.dataset_loader.load_dataset(dataset_name)
                loaded_datasets[dataset_name] = dataset
                self.raw_data[dataset_name] = dataset
                
                logger.info(f"Successfully loaded {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                if self.config.system.strict_mode:
                    raise
                else:
                    continue
        
        return loaded_datasets
    
    def validate_datasets(self, datasets: Optional[Dict[str, DatasetDict]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Validate loaded datasets.
        
        Args:
            datasets: Optional dictionary of datasets to validate.
                     If None, validates all loaded datasets.
            
        Returns:
            Dictionary mapping dataset names to validation results
        """
        if datasets is None:
            datasets = self.raw_data
        
        validation_results = {}
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"Validating dataset: {dataset_name}")
            
            try:
                results = self.data_validator.validate_dataset(dataset)
                validation_results[dataset_name] = results
                self.validation_results[dataset_name] = results
                
                if results["valid"]:
                    logger.info(f"Dataset {dataset_name} validation passed")
                else:
                    logger.warning(f"Dataset {dataset_name} validation failed with {len(results['errors'])} errors")
                    for error in results["errors"]:
                        logger.warning(f"  - {error}")
                
                if results["warnings"]:
                    logger.info(f"Dataset {dataset_name} has {len(results['warnings'])} warnings")
                    for warning in results["warnings"]:
                        logger.info(f"  - {warning}")
                        
            except Exception as e:
                logger.error(f"Failed to validate dataset {dataset_name}: {e}")
                validation_results[dataset_name] = {
                    "valid": False,
                    "errors": [f"Validation exception: {str(e)}"],
                    "warnings": []
                }
        
        return validation_results
    
    def get_dataset_info(self, dataset_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get information about datasets.
        
        Args:
            dataset_names: Optional list of dataset names. If None, returns info for all available datasets.
            
        Returns:
            Dictionary mapping dataset names to their information
        """
        if dataset_names is None:
            # Try to discover available datasets
            dataset_names = []
            dataset_path = Path(self.config.data.dataset_path)
            
            for potential_dataset in ['rcaeval', 'lemma-rca']:
                if (dataset_path / potential_dataset).exists():
                    dataset_names.append(potential_dataset)
        
        info = {}
        for dataset_name in dataset_names:
            try:
                info[dataset_name] = self.dataset_loader.get_dataset_info(dataset_name)
            except Exception as e:
                logger.error(f"Failed to get info for dataset {dataset_name}: {e}")
                info[dataset_name] = {"error": str(e)}
        
        return info
    
    def preprocess_metrics(self, metrics_data: MetricsData, dataset_name: str = "default") -> pd.DataFrame:
        """
        Preprocess metrics time series data.
        
        Args:
            metrics_data: Raw metrics data
            dataset_name: Name of the dataset for logging
            
        Returns:
            Preprocessed metrics DataFrame
        """
        logger.info(f"Preprocessing metrics data for {dataset_name}")
        
        # Convert to DataFrame for easier processing
        df = metrics_data.to_dataframe()
        
        # Sort by timestamp and service
        df = df.sort_values(['service_id', 'timestamp'])
        
        # Handle missing values based on configuration
        if self.config.data.normalization_method == 'interpolation':
            df = df.groupby('service_id').apply(
                lambda group: group.interpolate(method='time')
            ).reset_index(drop=True)
        else:
            df = df.fillna(0)  # Simple fill for now
        
        # Normalize metrics based on configuration
        metrics_columns = ['cpu_usage', 'memory_usage', 'response_time', 'error_rate']
        
        if self.config.data.normalization_method == 'minmax':
            for col in metrics_columns:
                if col in df.columns:
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        elif self.config.data.normalization_method == 'standard':
            for col in metrics_columns:
                if col in df.columns:
                    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        elif self.config.data.normalization_method == 'robust':
            for col in metrics_columns:
                if col in df.columns:
                    q25 = df[col].quantile(0.25)
                    q75 = df[col].quantile(0.75)
                    df[col] = (df[col] - df[col].median()) / (q75 - q25 + 1e-8)
        
        # Store processed data
        if dataset_name not in self.processed_data:
            self.processed_data[dataset_name] = {}
        self.processed_data[dataset_name]['metrics'] = df
        
        logger.info(f"Processed {len(df)} metrics records for {dataset_name}")
        return df
    
    def preprocess_logs(self, log_data: LogData, dataset_name: str = "default") -> pd.DataFrame:
        """
        Preprocess log data and extract templates.
        
        Args:
            log_data: Raw log data
            dataset_name: Name of the dataset for logging
            
        Returns:
            Preprocessed log DataFrame
        """
        logger.info(f"Preprocessing log data for {dataset_name}")
        
        # Convert to DataFrame
        df = log_data.to_dataframe()
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Clean log messages
        df['log_message'] = df['log_message'].fillna('')
        df['log_message'] = df['log_message'].str.strip()
        
        # Standardize log levels
        level_mapping = {
            'DEBUG': 'DEBUG',
            'INFO': 'INFO',
            'WARN': 'WARNING',
            'WARNING': 'WARNING',
            'ERROR': 'ERROR',
            'FATAL': 'ERROR',
            'CRITICAL': 'ERROR'
        }
        df['log_level'] = df['log_level'].str.upper().map(level_mapping).fillna('INFO')
        
        # Add log level encoding
        level_encoding = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        df['log_level_encoded'] = df['log_level'].map(level_encoding)
        
        # Store processed data
        if dataset_name not in self.processed_data:
            self.processed_data[dataset_name] = {}
        self.processed_data[dataset_name]['logs'] = df
        
        logger.info(f"Processed {len(df)} log records for {dataset_name}")
        return df
    
    def build_service_graph(self, trace_data: TraceData, dataset_name: str = "default") -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Build service dependency graph from trace data.
        
        Args:
            trace_data: Raw trace data
            dataset_name: Name of the dataset for logging
            
        Returns:
            Tuple of (adjacency DataFrame, service_to_index mapping)
        """
        logger.info(f"Building service graph for {dataset_name}")
        
        # Convert to DataFrame
        df = trace_data.to_dataframe()
        
        # Get unique services
        unique_services = sorted(df['service_name'].unique())
        service_to_index = {service: idx for idx, service in enumerate(unique_services)}
        num_services = len(unique_services)
        
        # Initialize adjacency matrix
        adjacency = pd.DataFrame(
            0, 
            index=unique_services, 
            columns=unique_services,
            dtype=float
        )
        
        # Build service relationships from parent-child span relationships
        for _, row in df.iterrows():
            if pd.notna(row['parent_span_id']):
                # Find parent span
                parent_spans = df[df['span_id'] == row['parent_span_id']]
                if not parent_spans.empty:
                    parent_service = parent_spans.iloc[0]['service_name']
                    child_service = row['service_name']
                    
                    if parent_service != child_service:
                        # Add edge from parent to child
                        adjacency.loc[parent_service, child_service] += 1
        
        # Normalize adjacency matrix
        for service in unique_services:
            row_sum = adjacency.loc[service].sum()
            if row_sum > 0:
                adjacency.loc[service] = adjacency.loc[service] / row_sum
        
        # Store processed data
        if dataset_name not in self.processed_data:
            self.processed_data[dataset_name] = {}
        self.processed_data[dataset_name]['graph'] = {
            'adjacency': adjacency,
            'service_mapping': service_to_index
        }
        
        logger.info(f"Built service graph with {num_services} services and {(adjacency > 0).sum().sum()} edges for {dataset_name}")
        return adjacency, service_to_index
    
    def create_temporal_sequences(self, df: pd.DataFrame, sequence_length: Optional[int] = None) -> List[pd.DataFrame]:
        """
        Create temporal sequences from time series data.
        
        Args:
            df: DataFrame with timestamp and service_id columns
            sequence_length: Length of sequences to create
            
        Returns:
            List of sequence DataFrames
        """
        if sequence_length is None:
            sequence_length = self.config.data.sequence_length
        
        sequences = []
        
        # Group by service to create sequences
        for service_id, group in df.groupby('service_id'):
            group = group.sort_values('timestamp')
            
            # Create sliding window sequences
            for i in range(0, len(group) - sequence_length + 1, self.config.data.sliding_window_step):
                sequence = group.iloc[i:i + sequence_length].copy()
                sequence['sequence_id'] = f"{service_id}_{i}"
                sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} temporal sequences of length {sequence_length}")
        return sequences
    
    def synchronize_data(self, dataset_name: str, time_window: str = '1min') -> Dict[str, pd.DataFrame]:
        """
        Synchronize different data modalities by time windows.
        
        Args:
            dataset_name: Name of the dataset to synchronize
            time_window: Time window for synchronization (e.g., '1min', '5min')
            
        Returns:
            Dictionary of synchronized DataFrames
        """
        logger.info(f"Synchronizing data for {dataset_name} with window {time_window}")
        
        if dataset_name not in self.processed_data:
            raise ValueError(f"No processed data found for dataset {dataset_name}")
        
        processed = self.processed_data[dataset_name]
        synchronized = {}
        
        # Synchronize each modality
        for modality, data in processed.items():
            if modality == 'graph':
                synchronized[modality] = data  # Graph doesn't need time synchronization
                continue
            
            if isinstance(data, pd.DataFrame) and 'timestamp' in data.columns:
                # Resample by time window
                data_copy = data.copy()
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
                data_copy = data_copy.set_index('timestamp')
                
                # Group by service and resample
                resampled_groups = []
                for service_id, group in data_copy.groupby('service_id'):
                    resampled = group.resample(time_window).mean()  # Use mean for numerical columns
                    resampled['service_id'] = service_id
                    resampled_groups.append(resampled.reset_index())
                
                if resampled_groups:
                    synchronized[modality] = pd.concat(resampled_groups, ignore_index=True)
                else:
                    synchronized[modality] = data
            else:
                synchronized[modality] = data
        
        # Store synchronized data
        if f"{dataset_name}_synchronized" not in self.processed_data:
            self.processed_data[f"{dataset_name}_synchronized"] = {}
        self.processed_data[f"{dataset_name}_synchronized"] = synchronized
        
        logger.info(f"Synchronized data for {dataset_name}")
        return synchronized
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of data processing status."""
        summary = {
            "raw_datasets": list(self.raw_data.keys()),
            "processed_datasets": list(self.processed_data.keys()),
            "validation_results": {},
            "processing_stats": {}
        }
        
        # Add validation summaries
        for dataset_name, validation in self.validation_results.items():
            summary["validation_results"][dataset_name] = {
                "valid": validation.get("valid", False),
                "errors": len(validation.get("errors", [])),
                "warnings": len(validation.get("warnings", []))
            }
        
        # Add processing statistics
        for dataset_name, processed in self.processed_data.items():
            stats = {}
            for modality, data in processed.items():
                if isinstance(data, pd.DataFrame):
                    stats[modality] = {
                        "records": len(data),
                        "columns": list(data.columns)
                    }
                elif isinstance(data, dict) and 'adjacency' in data:
                    stats[modality] = {
                        "services": len(data['service_mapping']),
                        "edges": (data['adjacency'] > 0).sum().sum()
                    }
            summary["processing_stats"][dataset_name] = stats
        
        return summary
    
    def save_processed_data(self, output_path: str) -> None:
        """Save processed data to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, processed in self.processed_data.items():
            dataset_dir = output_path / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            for modality, data in processed.items():
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(dataset_dir / f"{modality}.parquet")
                elif isinstance(data, dict):
                    if 'adjacency' in data:
                        data['adjacency'].to_csv(dataset_dir / f"{modality}_adjacency.csv")
                        pd.DataFrame(list(data['service_mapping'].items()), 
                                   columns=['service', 'index']).to_csv(
                                       dataset_dir / f"{modality}_mapping.csv", index=False)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, input_path: str) -> None:
        """Load previously processed data from files."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data path {input_path} does not exist")
        
        for dataset_dir in input_path.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                processed = {}
                
                for file_path in dataset_dir.glob("*.parquet"):
                    modality = file_path.stem
                    processed[modality] = pd.read_parquet(file_path)
                
                # Load graph data
                adjacency_path = dataset_dir / "graph_adjacency.csv"
                mapping_path = dataset_dir / "graph_mapping.csv"
                
                if adjacency_path.exists() and mapping_path.exists():
                    adjacency = pd.read_csv(adjacency_path, index_col=0)
                    mapping_df = pd.read_csv(mapping_path)
                    service_mapping = dict(zip(mapping_df['service'], mapping_df['index']))
                    
                    processed['graph'] = {
                        'adjacency': adjacency,
                        'service_mapping': service_mapping
                    }
                
                self.processed_data[dataset_name] = processed
        
        logger.info(f"Loaded processed data from {input_path}")