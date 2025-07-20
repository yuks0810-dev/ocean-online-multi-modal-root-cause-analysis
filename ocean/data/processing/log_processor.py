"""
Log data preprocessing and vectorization for OCEAN implementation.
Handles log template extraction using Drain algorithm and BERT-based vectorization.
"""

import re
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import logging
from datetime import datetime

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
except ImportError:
    logger.warning("drain3 not available, using fallback log template extraction")
    TemplateMiner = None
    TemplateMinerConfig = None

try:
    from transformers import BertTokenizer, BertModel
    import torch.nn.functional as F
except ImportError:
    logger.warning("transformers not available, BERT vectorization will not work")
    BertTokenizer = None
    BertModel = None

from ..data_types import LogData


logger = logging.getLogger(__name__)


class LogProcessor:
    """
    Specialized processor for log data.
    Handles template extraction and BERT-based vectorization.
    """
    
    def __init__(self,
                 template_extractor: str = 'drain',
                 bert_model_name: str = 'bert-base-uncased',
                 max_log_length: int = 512,
                 template_similarity_threshold: float = 0.4,
                 cache_embeddings: bool = True):
        """
        Initialize log processor.
        
        Args:
            template_extractor: 'drain' or 'simple' for template extraction
            bert_model_name: BERT model name for embedding
            max_log_length: Maximum log message length for BERT
            template_similarity_threshold: Similarity threshold for Drain algorithm
            cache_embeddings: Whether to cache BERT embeddings
        """
        self.template_extractor = template_extractor
        self.bert_model_name = bert_model_name
        self.max_log_length = max_log_length
        self.template_similarity_threshold = template_similarity_threshold
        self.cache_embeddings = cache_embeddings
        
        # Initialize components
        self.drain_parser = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.embedding_cache = {}
        
        # Template mapping
        self.template_to_id = {}
        self.id_to_template = {}
        self.template_counts = Counter()
        
        # Processing statistics
        self.processing_stats = {}
        
        # Initialize BERT components
        self._initialize_bert()
        
        # Initialize Drain components
        if self.template_extractor == 'drain':
            self._initialize_drain()
    
    def _initialize_bert(self) -> None:
        """Initialize BERT tokenizer and model."""
        if BertTokenizer is None or BertModel is None:
            logger.warning("BERT components not available")
            return
        
        try:
            logger.info(f"Loading BERT model: {self.bert_model_name}")
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = BertModel.from_pretrained(self.bert_model_name)
            self.bert_model.eval()
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
    
    def _initialize_drain(self) -> None:
        """Initialize Drain template miner."""
        if TemplateMiner is None:
            logger.warning("Drain3 not available, using simple template extraction")
            return
        
        try:
            # Configure Drain
            config = TemplateMinerConfig()
            config.load({
                'DRAIN': {
                    'sim_th': self.template_similarity_threshold,
                    'depth': 4,
                    'max_children': 100
                }
            })
            
            self.drain_parser = TemplateMiner(config=config)
            logger.info("Drain template miner initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Drain: {e}")
            self.drain_parser = None
    
    def preprocess_logs(self, 
                       log_data: LogData,
                       extract_templates: bool = True,
                       vectorize_logs: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for log data.
        
        Args:
            log_data: Raw log data
            extract_templates: Whether to extract log templates
            vectorize_logs: Whether to create BERT embeddings
            
        Returns:
            Preprocessed log DataFrame
        """
        logger.info("Starting log preprocessing pipeline")
        
        # Convert to DataFrame
        df = log_data.to_dataframe()
        
        # Store original statistics
        self.processing_stats['original_shape'] = df.shape
        self.processing_stats['original_services'] = df['service_id'].nunique()
        self.processing_stats['original_log_levels'] = df['log_level'].value_counts().to_dict()
        
        # Clean log messages
        df = self._clean_log_messages(df)
        
        # Standardize log levels
        df = self._standardize_log_levels(df)
        
        # Extract templates if requested
        if extract_templates:
            df = self._extract_templates(df)
        
        # Create BERT embeddings if requested
        if vectorize_logs and self.bert_model is not None:
            df = self._create_embeddings(df)
        
        # Add derived features
        df = self._add_log_features(df)
        
        # Store final statistics
        self.processing_stats['processed_shape'] = df.shape
        self.processing_stats['processed_services'] = df['service_id'].nunique()
        self.processing_stats['template_count'] = len(self.template_to_id)
        
        logger.info(f"Log preprocessing completed: {df.shape}")
        return df
    
    def _clean_log_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize log messages."""
        logger.info("Cleaning log messages")
        
        df = df.copy()
        
        # Remove null and empty messages
        df['log_message'] = df['log_message'].fillna('')
        
        # Basic cleaning
        def clean_message(message):
            if not isinstance(message, str):
                return ''
            
            # Remove extra whitespace
            message = re.sub(r'\s+', ' ', message.strip())
            
            # Remove ANSI escape codes
            message = re.sub(r'\x1b\[[0-9;]*m', '', message)
            
            # Normalize common patterns
            # IP addresses
            message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', message)
            
            # Numbers (but keep some context)
            message = re.sub(r'\b\d{4,}\b', '<NUM>', message)
            
            # UUIDs
            message = re.sub(
                r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
                '<UUID>', message, flags=re.IGNORECASE
            )
            
            # Timestamps
            message = re.sub(
                r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?',
                '<TIMESTAMP>', message
            )
            
            # File paths
            message = re.sub(r'(/[\w\-_./]+)', '<PATH>', message)
            
            return message
        
        df['log_message_cleaned'] = df['log_message'].apply(clean_message)
        
        # Remove empty messages after cleaning
        empty_before = len(df)
        df = df[df['log_message_cleaned'].str.len() > 0]
        empty_after = len(df)
        
        if empty_before != empty_after:
            logger.info(f"Removed {empty_before - empty_after} empty log messages")
        
        self.processing_stats['empty_messages_removed'] = empty_before - empty_after
        
        return df
    
    def _standardize_log_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize log level names and add numerical encoding."""
        logger.info("Standardizing log levels")
        
        df = df.copy()
        
        # Standardize level names
        level_mapping = {
            'DEBUG': 'DEBUG',
            'TRACE': 'DEBUG',
            'INFO': 'INFO',
            'INFORMATION': 'INFO',
            'WARN': 'WARNING',
            'WARNING': 'WARNING',
            'ERROR': 'ERROR',
            'ERR': 'ERROR',
            'FATAL': 'ERROR',
            'CRITICAL': 'ERROR',
            'CRIT': 'ERROR'
        }
        
        df['log_level'] = df['log_level'].str.upper().map(level_mapping).fillna('INFO')
        
        # Add numerical encoding (higher number = more severe)
        level_encoding = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3
        }
        
        df['log_level_numeric'] = df['log_level'].map(level_encoding)
        
        # Add severity indicator
        df['is_error'] = df['log_level'].isin(['WARNING', 'ERROR'])
        
        return df
    
    def _extract_templates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract log templates using Drain or simple methods."""
        logger.info(f"Extracting log templates using {self.template_extractor} method")
        
        df = df.copy()
        
        if self.template_extractor == 'drain' and self.drain_parser is not None:
            df = self._extract_templates_drain(df)
        else:
            df = self._extract_templates_simple(df)
        
        return df
    
    def _extract_templates_drain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract templates using Drain algorithm."""
        templates = []
        template_ids = []
        
        for message in df['log_message_cleaned']:
            result = self.drain_parser.add_log_message(message)
            template = result['template_mined']
            
            if template not in self.template_to_id:
                template_id = len(self.template_to_id)
                self.template_to_id[template] = template_id
                self.id_to_template[template_id] = template
            
            template_id = self.template_to_id[template]
            templates.append(template)
            template_ids.append(template_id)
            self.template_counts[template] += 1
        
        df['log_template'] = templates
        df['template_id'] = template_ids
        
        logger.info(f"Extracted {len(self.template_to_id)} unique templates using Drain")
        return df
    
    def _extract_templates_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple template extraction using regex patterns."""
        logger.info("Using simple template extraction")
        
        def extract_simple_template(message):
            # Replace specific patterns with placeholders
            template = message
            
            # Replace numbers with <*>
            template = re.sub(r'\d+', '<*>', template)
            
            # Replace quoted strings
            template = re.sub(r'"[^"]*"', '"<*>"', template)
            template = re.sub(r"'[^']*'", "'<*>'", template)
            
            # Replace common variable patterns
            template = re.sub(r'<[A-Z]+>', '<*>', template)
            template = re.sub(r'\b[a-f0-9]{8,}\b', '<*>', template)
            
            return template
        
        templates = df['log_message_cleaned'].apply(extract_simple_template)
        
        # Create template mapping
        unique_templates = templates.unique()
        for i, template in enumerate(unique_templates):
            if template not in self.template_to_id:
                self.template_to_id[template] = len(self.template_to_id)
                self.id_to_template[len(self.id_to_template)] = template
        
        df['log_template'] = templates
        df['template_id'] = templates.map(self.template_to_id)
        
        # Count templates
        self.template_counts = Counter(templates)
        
        logger.info(f"Extracted {len(self.template_to_id)} unique templates using simple method")
        return df
    
    def _create_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BERT embeddings for log messages."""
        logger.info("Creating BERT embeddings for log messages")
        
        embeddings = []
        
        # Process in batches for efficiency
        batch_size = 32
        messages = df['log_message_cleaned'].tolist()
        
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i:i + batch_size]
            batch_embeddings = self._get_bert_embeddings(batch_messages)
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array and add to dataframe
        embeddings_array = np.array(embeddings)
        
        # Store embeddings as a list of arrays (for easier handling)
        df['log_embedding'] = [emb for emb in embeddings_array]
        df['embedding_dim'] = embeddings_array.shape[1]
        
        self.processing_stats['embedding_dimension'] = embeddings_array.shape[1]
        
        logger.info(f"Created embeddings with dimension {embeddings_array.shape[1]}")
        return df
    
    def _get_bert_embeddings(self, messages: List[str]) -> List[np.ndarray]:
        """Get BERT embeddings for a batch of messages."""
        if not messages:
            return []
        
        embeddings = []
        
        for message in messages:
            # Check cache first
            if self.cache_embeddings and message in self.embedding_cache:
                embeddings.append(self.embedding_cache[message])
                continue
            
            try:
                # Tokenize
                inputs = self.bert_tokenizer(
                    message,
                    return_tensors='pt',
                    max_length=self.max_log_length,
                    padding=True,
                    truncation=True
                )
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                
                embeddings.append(embedding)
                
                # Cache embedding
                if self.cache_embeddings:
                    self.embedding_cache[message] = embedding
                    
            except Exception as e:
                logger.warning(f"Failed to create embedding for message: {e}")
                # Use zero embedding as fallback
                embedding_dim = 768  # BERT base hidden size
                embeddings.append(np.zeros(embedding_dim))
        
        return embeddings
    
    def _add_log_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from log data."""
        logger.info("Adding derived log features")
        
        df = df.copy()
        
        # Message length features
        df['message_length'] = df['log_message_cleaned'].str.len()
        df['message_word_count'] = df['log_message_cleaned'].str.split().str.len()
        
        # Template frequency features
        if 'template_id' in df.columns:
            template_freq = df['template_id'].value_counts().to_dict()
            df['template_frequency'] = df['template_id'].map(template_freq)
            df['template_rarity'] = 1.0 / df['template_frequency']
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Service-based features
        service_log_counts = df['service_id'].value_counts().to_dict()
        df['service_log_volume'] = df['service_id'].map(service_log_counts)
        
        # Error pattern features
        df['contains_exception'] = df['log_message_cleaned'].str.contains(
            r'exception|error|fail|timeout|connection|null', case=False, na=False
        )
        
        return df
    
    def create_log_sequences(self, 
                           df: pd.DataFrame,
                           sequence_length: int = 50,
                           step_size: int = 1) -> List[Dict[str, Any]]:
        """
        Create temporal sequences from log data.
        
        Args:
            df: Preprocessed log DataFrame
            sequence_length: Number of log entries per sequence
            step_size: Step size for sliding window
            
        Returns:
            List of log sequences
        """
        logger.info(f"Creating log sequences of length {sequence_length}")
        
        sequences = []
        
        # Group by service
        for service_id, group in df.groupby('service_id'):
            group = group.sort_values('timestamp')
            
            if len(group) < sequence_length:
                continue
            
            # Create sliding window sequences
            for i in range(0, len(group) - sequence_length + 1, step_size):
                sequence_data = group.iloc[i:i + sequence_length]
                
                # Prepare sequence features
                features = {}
                
                # Template IDs
                if 'template_id' in sequence_data.columns:
                    features['template_ids'] = sequence_data['template_id'].values
                
                # Log level numeric
                if 'log_level_numeric' in sequence_data.columns:
                    features['log_levels'] = sequence_data['log_level_numeric'].values
                
                # Embeddings
                if 'log_embedding' in sequence_data.columns:
                    embeddings = np.stack(sequence_data['log_embedding'].values)
                    features['embeddings'] = embeddings
                
                # Derived features
                numerical_features = []
                for col in ['message_length', 'message_word_count', 'template_frequency', 
                           'template_rarity', 'hour', 'day_of_week']:
                    if col in sequence_data.columns:
                        numerical_features.append(sequence_data[col].values)
                
                if numerical_features:
                    features['numerical'] = np.stack(numerical_features, axis=1)
                
                # Binary features
                binary_features = []
                for col in ['is_error', 'is_weekend', 'contains_exception']:
                    if col in sequence_data.columns:
                        binary_features.append(sequence_data[col].astype(int).values)
                
                if binary_features:
                    features['binary'] = np.stack(binary_features, axis=1)
                
                sequence = {
                    'service_id': service_id,
                    'features': features,
                    'timestamps': sequence_data['timestamp'].tolist(),
                    'start_time': sequence_data['timestamp'].iloc[0],
                    'end_time': sequence_data['timestamp'].iloc[-1],
                    'sequence_id': f"{service_id}_log_{i}",
                    'sequence_length': sequence_length
                }
                
                sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} log sequences")
        self.processing_stats['num_log_sequences'] = len(sequences)
        
        return sequences
    
    def get_template_summary(self) -> Dict[str, Any]:
        """Get summary of extracted templates."""
        if not self.template_to_id:
            return {"error": "No templates extracted"}
        
        # Get top templates by frequency
        top_templates = self.template_counts.most_common(10)
        
        summary = {
            "total_templates": len(self.template_to_id),
            "total_log_instances": sum(self.template_counts.values()),
            "top_templates": [
                {"template": template, "count": count, "percentage": count / sum(self.template_counts.values()) * 100}
                for template, count in top_templates
            ],
            "template_distribution": {
                "rare_templates": sum(1 for count in self.template_counts.values() if count == 1),
                "common_templates": sum(1 for count in self.template_counts.values() if count >= 10),
                "frequent_templates": sum(1 for count in self.template_counts.values() if count >= 100)
            }
        }
        
        return summary
    
    def save_templates(self, filepath: str) -> None:
        """Save extracted templates to file."""
        import json
        
        template_data = {
            "template_to_id": self.template_to_id,
            "id_to_template": self.id_to_template,
            "template_counts": dict(self.template_counts)
        }
        
        with open(filepath, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Templates saved to {filepath}")
    
    def load_templates(self, filepath: str) -> None:
        """Load extracted templates from file."""
        import json
        
        with open(filepath, 'r') as f:
            template_data = json.load(f)
        
        self.template_to_id = template_data["template_to_id"]
        self.id_to_template = {int(k): v for k, v in template_data["id_to_template"].items()}
        self.template_counts = Counter(template_data["template_counts"])
        
        logger.info(f"Templates loaded from {filepath}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()