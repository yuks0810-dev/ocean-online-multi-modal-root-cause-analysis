"""
Default configuration for OCEAN model implementation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for OCEAN model components."""
    
    # Dilated CNN configuration
    dcnn_channels: List[int] = None
    dcnn_kernel_size: int = 3
    dcnn_dilation_rates: List[int] = None
    dcnn_dropout: float = 0.1
    
    # GNN configuration
    gnn_hidden_dim: int = 128
    gnn_num_heads: int = 4
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.1
    
    # Multi-factor Attention configuration
    attention_dim: int = 128
    attention_num_heads: int = 8
    attention_dropout: float = 0.1
    
    # Graph Fusion configuration
    fusion_projection_dim: int = 256
    fusion_temperature: float = 0.1
    
    # General model configuration
    hidden_dim: int = 128
    output_dim: int = 1
    num_services: int = 100
    
    # Input dimensions
    temporal_dim: int = 128
    spatial_dim: int = 128
    log_dim: int = 768
    fusion_dim: int = 256
    
    def __post_init__(self):
        if self.dcnn_channels is None:
            self.dcnn_channels = [64, 128, 256, 512]
        if self.dcnn_dilation_rates is None:
            self.dcnn_dilation_rates = [1, 2, 4, 8]


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset paths
    dataset_path: str = "data/datasets"
    rcaeval_path: Optional[str] = None
    lemma_rca_path: Optional[str] = None
    
    # Preprocessing parameters
    sequence_length: int = 100
    sliding_window_step: int = 1
    normalization_method: str = "minmax"  # "minmax", "standard", "robust"
    
    # Log processing
    log_template_extractor: str = "drain"  # "drain", "spell"
    bert_model_name: str = "bert-base-uncased"
    max_log_length: int = 512
    
    # Graph construction
    edge_threshold: float = 0.1
    max_graph_size: int = 1000
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Loss function weights
    prediction_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1
    
    # Online learning
    online_learning: bool = True
    sliding_window_size: int = 1000
    update_frequency: int = 10
    
    # Optimization
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    
    # Checkpointing
    save_checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation metrics
    top_k_values: List[int] = None
    compute_statistical_significance: bool = True
    significance_level: float = 0.05
    
    # Baseline methods
    baseline_methods: List[str] = None
    
    # Ablation study
    ablation_components: List[str] = None
    
    def __post_init__(self):
        if self.top_k_values is None:
            self.top_k_values = [1, 3, 5, 10]
        if self.baseline_methods is None:
            self.baseline_methods = ["random", "pagerank", "clustering"]
        if self.ablation_components is None:
            self.ablation_components = ["dcnn", "gnn", "attention", "fusion"]


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    gpu_memory_fraction: float = 0.8
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    wandb_project: Optional[str] = "ocean-rca"
    wandb_entity: Optional[str] = None
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Performance
    num_threads: int = 4
    mixed_precision: bool = False


@dataclass
class OCEANConfig:
    """Main configuration class combining all configurations."""
    
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.system is None:
            self.system = SystemConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "system": self.system.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OCEANConfig":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            system=SystemConfig(**config_dict.get("system", {})),
        )


def get_device(device_config: str = "auto") -> torch.device:
    """Get the appropriate device for computation."""
    if device_config == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)


# Default configuration function
def default_config() -> OCEANConfig:
    """Create and return default OCEAN configuration."""
    return OCEANConfig()