# Design Document

## Overview

The OCEAN (Online Multi-modal Causal structure lEArNiNG) implementation is designed as a modular deep learning system that processes multi-modal data (metrics and logs) from microservice systems to perform real-time root cause analysis. The system follows an online learning paradigm, processing data sequentially and updating model parameters incrementally.

The architecture consists of four main components: a Dilated Convolutional Neural Network (DCNN) for temporal feature extraction, a Graph Neural Network (GNN) for spatial relationship modeling, a Multi-factor Attention mechanism for feature weighting, and a Graph Fusion Module that uses contrastive learning to combine multi-modal representations.

## Architecture

### High-Level System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Preprocessing  │    │  OCEAN Model    │
│                 │    │                  │    │                 │
│ • Metrics       │───▶│ • Normalization  │───▶│ • DCNN          │
│ • Logs          │    │ • Vectorization  │    │ • GNN           │
│ • Traces        │    │ • Graph Building │    │ • Attention     │
└─────────────────┘    └──────────────────┘    │ • Graph Fusion  │
                                               └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │ Root Cause      │
                                               │ Prediction      │
                                               └─────────────────┘
```

### Component Architecture

The OCEAN model follows a multi-branch architecture where different modalities are processed in parallel before being fused:

```
Metrics Data ──┐
               ├──▶ DCNN ──┐
Time Series ───┘           │
                          ├──▶ Multi-factor ──┐
Service Graph ──┐          │    Attention     │
                ├──▶ GNN ──┘                  ├──▶ Graph Fusion ──▶ Root Cause Scores
Log Data ───────┘                            │    Module
                                             │
Contrastive Learning ────────────────────────┘
```

## Components and Interfaces

### 1. Data Processing Pipeline

**DataProcessor Class**
- **Purpose**: Handles multi-modal data preprocessing and synchronization
- **Key Methods**:
  - `load_datasets()`: Loads RCAEval/LEMMA-RCA datasets
  - `preprocess_metrics()`: Normalizes and creates temporal sequences
  - `preprocess_logs()`: Extracts templates and vectorizes using BERT
  - `build_service_graph()`: Constructs adjacency matrices from trace data
  - `create_dataloaders()`: Provides synchronized batch processing

**Interface**:
```python
class DataProcessor:
    def __init__(self, config: Dict[str, Any])
    def load_datasets(self, dataset_path: str) -> Tuple[pd.DataFrame, ...]
    def preprocess_metrics(self, metrics_df: pd.DataFrame) -> torch.Tensor
    def preprocess_logs(self, logs_df: pd.DataFrame) -> torch.Tensor
    def build_service_graph(self, traces_df: pd.DataFrame) -> torch.Tensor
    def create_dataloaders(self, batch_size: int) -> Tuple[DataLoader, ...]
```

### 2. OCEAN Model Components

**DilatedCNN Class**
- **Purpose**: Extracts temporal features from metrics time series
- **Architecture**: Multiple dilated convolution layers with increasing dilation rates
- **Input**: Normalized metrics sequences (batch_size, seq_len, num_metrics)
- **Output**: Temporal feature representations (batch_size, hidden_dim)

**GraphNeuralNetwork Class**
- **Purpose**: Models spatial relationships between services
- **Architecture**: Graph Attention Network (GAT) layers using PyTorch Geometric
- **Input**: Node features and adjacency matrix
- **Output**: Service embeddings with relationship information

**MultiFactorAttention Class**
- **Purpose**: Dynamically weights different features and modalities
- **Architecture**: Multi-head attention mechanism with learnable query, key, value projections
- **Input**: Multi-modal feature representations
- **Output**: Attention-weighted feature vectors

**GraphFusionModule Class**
- **Purpose**: Fuses multi-modal graph representations using contrastive learning
- **Architecture**: Contrastive mutual information maximization framework
- **Key Components**:
  - Projection heads for each modality
  - InfoNCE loss computation
  - Positive/negative pair generation

### 3. Online Learning Framework

**OnlineLearner Class**
- **Purpose**: Manages sequential data processing and incremental model updates
- **Key Features**:
  - Sliding window management for temporal dependencies
  - Incremental parameter updates
  - Memory-efficient processing for continuous streams

**Interface**:
```python
class OnlineLearner:
    def __init__(self, model: OCEANModel, optimizer: torch.optim.Optimizer)
    def process_timestep(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, float]
    def update_model(self, loss: torch.Tensor) -> None
    def evaluate_current_state(self) -> Dict[str, float]
```

### 4. Evaluation Framework

**ModelEvaluator Class**
- **Purpose**: Comprehensive model evaluation with multiple metrics
- **Metrics**: Precision, Recall, F1-score, Accuracy@K, processing time
- **Features**: Statistical significance testing, baseline comparison

## Data Models

### Input Data Schema

**MetricsData**
```python
@dataclass
class MetricsData:
    timestamp: List[datetime]
    service_id: List[str]
    cpu_usage: List[float]
    memory_usage: List[float]
    response_time: List[float]
    error_rate: List[float]
    # Additional metrics as available in dataset
```

**LogData**
```python
@dataclass
class LogData:
    timestamp: List[datetime]
    service_id: List[str]
    log_template: List[str]
    log_embedding: List[List[float]]  # BERT embeddings
    severity_level: List[str]
```

**TraceData**
```python
@dataclass
class TraceData:
    trace_id: List[str]
    span_id: List[str]
    parent_span_id: List[str]
    service_name: List[str]
    operation_name: List[str]
    start_time: List[datetime]
    duration: List[float]
```

### Model Internal Representations

**ServiceGraph**
```python
@dataclass
class ServiceGraph:
    adjacency_matrix: torch.Tensor  # (num_services, num_services)
    node_features: torch.Tensor     # (num_services, feature_dim)
    edge_weights: torch.Tensor      # (num_edges,)
    service_names: List[str]
```

**MultiModalFeatures**
```python
@dataclass
class MultiModalFeatures:
    temporal_features: torch.Tensor    # From DCNN
    spatial_features: torch.Tensor     # From GNN
    attention_weights: torch.Tensor    # From attention mechanism
    fused_representation: torch.Tensor # From graph fusion
```

## Error Handling

### Data Processing Errors
- **Missing Data**: Implement interpolation strategies for missing metrics/logs
- **Format Inconsistencies**: Robust parsing with fallback mechanisms
- **Memory Constraints**: Implement data streaming and chunking for large datasets

### Model Training Errors
- **Convergence Issues**: Implement learning rate scheduling and early stopping
- **Gradient Explosion**: Apply gradient clipping and normalization
- **GPU Memory**: Implement gradient accumulation and model checkpointing

### Online Learning Errors
- **Data Stream Interruptions**: Implement buffering and recovery mechanisms
- **Model Drift**: Monitor performance degradation and trigger retraining
- **Real-time Constraints**: Implement timeout mechanisms and fallback predictions

### Error Recovery Strategies
```python
class ErrorHandler:
    def handle_data_error(self, error: Exception, data_batch: Dict) -> Dict
    def handle_model_error(self, error: Exception, model_state: Dict) -> bool
    def handle_memory_error(self, error: Exception) -> None
    def log_error(self, error: Exception, context: Dict) -> None
```

## Testing Strategy

### Unit Testing
- **Component Testing**: Individual testing of DCNN, GNN, Attention, and Fusion modules
- **Data Processing Testing**: Validation of preprocessing pipelines with synthetic data
- **Utility Function Testing**: Testing of helper functions and data transformations

### Integration Testing
- **End-to-End Pipeline**: Testing complete data flow from raw input to predictions
- **Multi-Modal Integration**: Ensuring proper synchronization between different data types
- **Online Learning Integration**: Testing incremental learning with sequential data

### Performance Testing
- **Scalability Testing**: Evaluating performance with varying dataset sizes
- **Memory Usage Testing**: Monitoring memory consumption during training and inference
- **Speed Benchmarking**: Measuring processing time against baseline methods

### Validation Testing
- **Cross-Validation**: K-fold validation on available datasets
- **Temporal Validation**: Time-based splits to simulate real-world deployment
- **Ablation Testing**: Systematic removal of components to validate contributions

### Test Data Strategy
```python
class TestDataGenerator:
    def generate_synthetic_metrics(self, num_services: int, time_steps: int) -> MetricsData
    def generate_synthetic_logs(self, num_services: int, num_logs: int) -> LogData
    def generate_synthetic_traces(self, num_services: int, num_traces: int) -> TraceData
    def inject_anomalies(self, data: Any, anomaly_type: str) -> Any
```

### Continuous Integration
- **Automated Testing**: GitHub Actions for running tests on code changes
- **Performance Regression**: Automated detection of performance degradation
- **Model Validation**: Automated validation against known benchmarks
- **Documentation Testing**: Ensuring code examples in documentation work correctly

The testing strategy ensures robustness, reproducibility, and reliability of the OCEAN implementation while maintaining compatibility with the research objectives and paper reproduction goals.