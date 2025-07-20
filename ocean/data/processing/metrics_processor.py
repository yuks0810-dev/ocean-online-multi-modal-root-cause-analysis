"""
Metrics data preprocessing for OCEAN implementation.
Handles time-series normalization, sequence generation, and temporal processing.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging

from ..data_types import MetricsData


logger = logging.getLogger(__name__)


class MetricsProcessor:
    """
    Specialized processor for metrics time series data.
    Handles normalization, sequence generation, and temporal processing.
    """
    
    def __init__(self, 
                 normalization_method: str = 'minmax',
                 sequence_length: int = 100,
                 sliding_window_step: int = 1,
                 interpolation_method: str = 'linear',
                 outlier_threshold: float = 3.0):
        """
        Initialize metrics processor.
        
        Args:
            normalization_method: 'minmax', 'standard', or 'robust'
            sequence_length: Length of temporal sequences
            sliding_window_step: Step size for sliding window
            interpolation_method: Method for handling missing values
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.normalization_method = normalization_method
        self.sequence_length = sequence_length
        self.sliding_window_step = sliding_window_step
        self.interpolation_method = interpolation_method
        self.outlier_threshold = outlier_threshold
        
        # Initialize scalers
        self.scalers = {}
        self.feature_columns = ['cpu_usage', 'memory_usage', 'response_time', 'error_rate']
        
        # Processing statistics
        self.processing_stats = {}
    
    def preprocess_metrics(self, 
                          metrics_data: MetricsData,
                          fit_scalers: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for metrics data.
        
        Args:
            metrics_data: Raw metrics data
            fit_scalers: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Preprocessed metrics DataFrame
        """
        logger.info("Starting metrics preprocessing pipeline")
        
        # Convert to DataFrame
        df = metrics_data.to_dataframe()
        
        # Store original statistics
        self.processing_stats['original_shape'] = df.shape
        self.processing_stats['original_services'] = df['service_id'].nunique()
        
        # Sort by service and timestamp
        df = df.sort_values(['service_id', 'timestamp'])
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Detect and handle outliers
        df = self._handle_outliers(df)
        
        # Normalize features
        df = self._normalize_features(df, fit_scalers=fit_scalers)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Store final statistics
        self.processing_stats['processed_shape'] = df.shape
        self.processing_stats['processed_services'] = df['service_id'].nunique()
        
        logger.info(f"Metrics preprocessing completed: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in metrics data."""
        logger.info("Handling missing values")
        
        # Check missing value percentage
        missing_pct = df[self.feature_columns].isnull().sum() / len(df) * 100
        self.processing_stats['missing_percentages'] = missing_pct.to_dict()
        
        # Group by service for interpolation
        processed_groups = []
        
        for service_id, group in df.groupby('service_id'):
            group = group.copy()
            
            # Time-based interpolation for each service
            group['timestamp'] = pd.to_datetime(group['timestamp'])
            group = group.set_index('timestamp').sort_index()
            
            # Interpolate missing values
            if self.interpolation_method == 'linear':
                group[self.feature_columns] = group[self.feature_columns].interpolate(method='time')
            elif self.interpolation_method == 'forward':
                group[self.feature_columns] = group[self.feature_columns].fillna(method='ffill')
            elif self.interpolation_method == 'backward':
                group[self.feature_columns] = group[self.feature_columns].fillna(method='bfill')
            
            # Fill any remaining NaN values with median
            for col in self.feature_columns:
                if group[col].isnull().any():
                    median_val = group[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    group[col] = group[col].fillna(median_val)
            
            group = group.reset_index()
            group['service_id'] = service_id
            processed_groups.append(group)
        
        if processed_groups:
            df = pd.concat(processed_groups, ignore_index=True)
        
        logger.info(f"Missing values handled: {missing_pct.sum():.2f}% total missing")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using Z-score method."""
        logger.info("Detecting and handling outliers")
        
        outliers_detected = {}
        
        for col in self.feature_columns:
            if col in df.columns:
                # Calculate Z-scores
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.outlier_threshold
                outliers_count = outliers.sum()
                
                outliers_detected[col] = outliers_count
                
                if outliers_count > 0:
                    # Replace outliers with median
                    median_val = df[col].median()
                    df.loc[outliers, col] = median_val
                    logger.info(f"Replaced {outliers_count} outliers in {col} with median value {median_val:.3f}")
        
        self.processing_stats['outliers_detected'] = outliers_detected
        return df
    
    def _normalize_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> pd.DataFrame:
        """Normalize feature values."""
        logger.info(f"Normalizing features using {self.normalization_method} method")
        
        df = df.copy()
        
        for col in self.feature_columns:
            if col not in df.columns:
                continue
            
            if fit_scalers:
                # Initialize and fit scaler
                if self.normalization_method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.normalization_method == 'standard':
                    scaler = StandardScaler()
                elif self.normalization_method == 'robust':
                    scaler = RobustScaler()
                else:
                    logger.warning(f"Unknown normalization method: {self.normalization_method}")
                    continue
                
                # Fit scaler on the column
                values = df[col].values.reshape(-1, 1)
                scaler.fit(values)
                self.scalers[col] = scaler
            else:
                # Use existing scaler
                if col not in self.scalers:
                    logger.warning(f"No fitted scaler found for {col}")
                    continue
                scaler = self.scalers[col]
            
            # Transform values
            values = df[col].values.reshape(-1, 1)
            normalized_values = scaler.transform(values)
            df[col] = normalized_values.flatten()
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from existing metrics."""
        logger.info("Adding derived features")
        
        df = df.copy()
        
        # Group by service to calculate derived features
        derived_groups = []
        
        for service_id, group in df.groupby('service_id'):
            group = group.copy()
            group = group.sort_values('timestamp')
            
            # Rolling statistics (window of 5)
            window_size = min(5, len(group))
            if window_size > 1:
                for col in self.feature_columns:
                    if col in group.columns:
                        # Rolling mean
                        group[f'{col}_rolling_mean'] = group[col].rolling(
                            window=window_size, min_periods=1
                        ).mean()
                        
                        # Rolling standard deviation
                        group[f'{col}_rolling_std'] = group[col].rolling(
                            window=window_size, min_periods=1
                        ).std().fillna(0)
            
            # Rate of change
            for col in self.feature_columns:
                if col in group.columns:
                    group[f'{col}_rate_of_change'] = group[col].pct_change().fillna(0)
            
            # Composite features
            if all(col in group.columns for col in ['cpu_usage', 'memory_usage']):
                group['resource_utilization'] = (group['cpu_usage'] + group['memory_usage']) / 2
            
            if all(col in group.columns for col in ['response_time', 'error_rate']):
                group['performance_degradation'] = group['response_time'] * (1 + group['error_rate'])
            
            derived_groups.append(group)
        
        if derived_groups:
            df = pd.concat(derived_groups, ignore_index=True)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create temporal sequences from metrics data.
        
        Args:
            df: Preprocessed metrics DataFrame
            
        Returns:
            List of sequence dictionaries
        """
        logger.info(f"Creating temporal sequences of length {self.sequence_length}")
        
        sequences = []
        
        # Get all feature columns (original + derived)
        feature_columns = [col for col in df.columns 
                          if col not in ['timestamp', 'service_id']]
        
        # Group by service
        for service_id, group in df.groupby('service_id'):
            group = group.sort_values('timestamp')
            
            # Skip if not enough data
            if len(group) < self.sequence_length:
                logger.warning(f"Service {service_id} has insufficient data ({len(group)} < {self.sequence_length})")
                continue
            
            # Create sliding window sequences
            for i in range(0, len(group) - self.sequence_length + 1, self.sliding_window_step):
                sequence_data = group.iloc[i:i + self.sequence_length]
                
                # Extract features as numpy array
                features = sequence_data[feature_columns].values
                timestamps = sequence_data['timestamp'].tolist()
                
                sequence = {
                    'service_id': service_id,
                    'features': features,  # Shape: (sequence_length, num_features)
                    'timestamps': timestamps,
                    'start_time': timestamps[0],
                    'end_time': timestamps[-1],
                    'sequence_id': f"{service_id}_{i}",
                    'feature_names': feature_columns
                }
                
                sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} sequences")
        self.processing_stats['num_sequences'] = len(sequences)
        
        return sequences
    
    def sequences_to_tensors(self, sequences: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Convert sequences to PyTorch tensors.
        
        Args:
            sequences: List of sequence dictionaries
            
        Returns:
            Tuple of (features_tensor, service_ids, sequence_ids)
        """
        if not sequences:
            return torch.empty(0), [], []
        
        # Stack all features
        features_list = [seq['features'] for seq in sequences]
        features_tensor = torch.tensor(np.stack(features_list), dtype=torch.float32)
        
        # Extract service and sequence IDs
        service_ids = [seq['service_id'] for seq in sequences]
        sequence_ids = [seq['sequence_id'] for seq in sequences]
        
        logger.info(f"Converted to tensor shape: {features_tensor.shape}")
        return features_tensor, service_ids, sequence_ids
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance based on variance."""
        feature_columns = [col for col in df.columns 
                          if col not in ['timestamp', 'service_id']]
        
        importance = {}
        for col in feature_columns:
            if col in df.columns:
                # Use coefficient of variation as importance measure
                mean_val = df[col].mean()
                std_val = df[col].std()
                if mean_val != 0:
                    importance[col] = std_val / abs(mean_val)
                else:
                    importance[col] = std_val
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def inverse_transform(self, normalized_data: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            normalized_data: Normalized feature values
            feature_name: Name of the feature to inverse transform
            
        Returns:
            Data in original scale
        """
        if feature_name not in self.scalers:
            logger.warning(f"No scaler found for feature {feature_name}")
            return normalized_data
        
        scaler = self.scalers[feature_name]
        if normalized_data.ndim == 1:
            normalized_data = normalized_data.reshape(-1, 1)
        
        return scaler.inverse_transform(normalized_data).flatten()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def save_scalers(self, filepath: str) -> None:
        """Save fitted scalers to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str) -> None:
        """Load fitted scalers from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)
        logger.info(f"Scalers loaded from {filepath}")
    
    def validate_sequences(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate generated sequences."""
        if not sequences:
            return {"valid": False, "error": "No sequences generated"}
        
        validation_results = {
            "valid": True,
            "num_sequences": len(sequences),
            "services": len(set(seq['service_id'] for seq in sequences)),
            "avg_sequence_length": np.mean([seq['features'].shape[0] for seq in sequences]),
            "feature_dimensions": sequences[0]['features'].shape[1] if sequences else 0,
            "issues": []
        }
        
        # Check for consistency
        expected_length = self.sequence_length
        expected_features = len(sequences[0]['feature_names']) if sequences else 0
        
        for i, seq in enumerate(sequences):
            if seq['features'].shape[0] != expected_length:
                validation_results["issues"].append(
                    f"Sequence {i} has length {seq['features'].shape[0]}, expected {expected_length}"
                )
            
            if seq['features'].shape[1] != expected_features:
                validation_results["issues"].append(
                    f"Sequence {i} has {seq['features'].shape[1]} features, expected {expected_features}"
                )
            
            # Check for NaN values
            if np.isnan(seq['features']).any():
                validation_results["issues"].append(f"Sequence {i} contains NaN values")
        
        if validation_results["issues"]:
            validation_results["valid"] = False
        
        return validation_results