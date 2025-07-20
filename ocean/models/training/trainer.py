"""
Trainer class for OCEAN model with support for both batch and online learning.
Implements comprehensive training loop with loss functions and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb

from ..ocean_model import OCEANModel, OCEANVariant
from ...data import MultiModalDataLoader
from ...configs import OCEANConfig
from .online_learner import OnlineLearner


logger = logging.getLogger(__name__)


class LossFunction(nn.Module):
    """
    Comprehensive loss function for OCEAN model combining multiple objectives.
    """
    
    def __init__(self, 
                 prediction_weight: float = 1.0,
                 contrastive_weight: float = 0.1,
                 consistency_weight: float = 0.05,
                 diversity_weight: float = 0.02):
        """
        Initialize loss function.
        
        Args:
            prediction_weight: Weight for prediction loss
            contrastive_weight: Weight for contrastive learning loss
            consistency_weight: Weight for temporal consistency loss
            diversity_weight: Weight for feature diversity loss
        """
        super(LossFunction, self).__init__()
        
        self.prediction_weight = prediction_weight
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.diversity_weight = diversity_weight
        
        # Loss components
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None,
                prev_temporal_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth labels
            temporal_features: Current temporal features
            prev_temporal_features: Previous temporal features (for consistency)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # 1. Prediction loss (Binary Cross Entropy)
        pred_loss = self.bce_loss(predictions['root_cause_probs'], targets.float())
        losses['prediction_loss'] = pred_loss
        
        # 2. Contrastive learning losses
        total_contrastive_loss = 0.0
        if 'contrastive_losses' in predictions:
            for loss_name, loss_value in predictions['contrastive_losses'].items():
                losses[f'contrastive_{loss_name}'] = loss_value
                total_contrastive_loss += loss_value
        losses['total_contrastive_loss'] = total_contrastive_loss
        
        # 3. Temporal consistency loss
        consistency_loss = 0.0
        if temporal_features is not None and prev_temporal_features is not None:
            # Encourage smooth temporal transitions
            consistency_loss = self.mse_loss(temporal_features, prev_temporal_features)
        losses['consistency_loss'] = consistency_loss
        
        # 4. Feature diversity loss (encourage diverse representations)
        diversity_loss = 0.0
        if 'intermediate' in predictions and 'fused_representation' in predictions['intermediate']:
            fused_features = predictions['intermediate']['fused_representation']
            # Compute pairwise cosine similarities
            normalized_features = nn.functional.normalize(fused_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # Encourage diversity by penalizing high similarities
            # Exclude diagonal (self-similarity)
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
            diversity_loss = similarity_matrix[mask].abs().mean()
        
        losses['diversity_loss'] = diversity_loss
        
        # 5. Total weighted loss
        total_loss = (self.prediction_weight * pred_loss +
                     self.contrastive_weight * total_contrastive_loss +
                     self.consistency_weight * consistency_loss +
                     self.diversity_weight * diversity_loss)
        
        losses['total_loss'] = total_loss
        
        return losses


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: 'min' or 'max' for optimization direction
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, current_value: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            self.best_weights = model.state_dict().copy()
            return False
        
        if self.monitor_op(current_value - self.min_delta, self.best_value):
            self.best_value = current_value
            self.best_weights = model.state_dict().copy()
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class LearningRateScheduler:
    """Advanced learning rate scheduler with multiple strategies."""
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 scheduler_type: str = 'cosine',
                 **scheduler_kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'exponential')
            **scheduler_kwargs: Additional arguments for scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=scheduler_kwargs.get('T_max', 100),
                eta_min=scheduler_kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_kwargs.get('step_size', 30),
                gamma=scheduler_kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_kwargs.get('mode', 'min'),
                factor=scheduler_kwargs.get('factor', 0.5),
                patience=scheduler_kwargs.get('patience', 10),
                min_lr=scheduler_kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_kwargs.get('gamma', 0.95)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler_type == 'plateau':
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates."""
        return self.scheduler.get_last_lr()


class Trainer:
    """
    Comprehensive trainer for OCEAN model supporting both batch and online learning.
    """
    
    def __init__(self, 
                 model: Union[OCEANModel, OCEANVariant],
                 config: OCEANConfig,
                 data_loader: MultiModalDataLoader,
                 optimizer: Optional[optim.Optimizer] = None,
                 device: Optional[torch.device] = None,
                 use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: OCEAN model instance
            config: OCEAN configuration
            data_loader: Multi-modal data loader
            optimizer: PyTorch optimizer (creates Adam if None)
            device: Training device
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.use_wandb = use_wandb
        
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(device)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Initialize training components
        self.loss_function = LossFunction(
            prediction_weight=config.training.prediction_loss_weight,
            contrastive_weight=config.training.contrastive_loss_weight
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_delta
        )
        
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type=config.training.scheduler
        )
        
        # Training state
        self.current_epoch = 0
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.training_start_time = None
        
        # Online learning components
        self.online_learner = None
        self.prev_temporal_features = None
        
        logger.info(f"Initialized Trainer with device: {device}")
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        self.training_start_time = datetime.now()
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.config.system.wandb_project,
                entity=self.config.system.wandb_entity,
                config=self.config.to_dict()
            )
        
        # Get data loaders
        train_loader, val_loader, _ = self.data_loader.get_all_loaders()
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch(train_loader)
                
                # Validation phase
                val_metrics = self._validate_epoch(val_loader)
                
                # Update learning rate
                self.lr_scheduler.step(val_metrics['val_loss'])
                
                # Log metrics
                self._log_epoch_metrics(train_metrics, val_metrics)
                
                # Check early stopping
                if self.early_stopping(val_metrics['val_loss'], self.model):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Save checkpoint
                if (epoch + 1) % self.config.training.save_checkpoint_every == 0:
                    self._save_checkpoint(epoch, val_metrics['val_loss'])
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Save final model
            self._save_final_model()
            
            if self.use_wandb:
                wandb.finish()
        
        training_time = datetime.now() - self.training_start_time
        logger.info(f"Training completed in {training_time}")
        
        return {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Move data to device
            batch_data = self._move_batch_to_device(batch_data)
            
            # Forward pass
            outputs = self.model(
                batch_data['metrics'],
                batch_data['graphs'][0],  # Assume same graph for batch
                batch_data['logs'],
                return_intermediate=True
            )
            
            # Compute losses
            losses = self.loss_function(
                outputs,
                batch_data['labels'],
                outputs['intermediate']['temporal_features'] if 'intermediate' in outputs else None,
                self.prev_temporal_features
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Store losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name].append(loss_value.item())
            
            # Update previous temporal features for consistency loss
            if 'intermediate' in outputs:
                self.prev_temporal_features = outputs['intermediate']['temporal_features'].detach()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.debug(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {losses['total_loss'].item():.4f}")
        
        # Compute epoch averages
        epoch_metrics = {}
        for loss_name, loss_values in epoch_losses.items():
            epoch_metrics[f'train_{loss_name}'] = np.mean(loss_values)
        
        return epoch_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Move data to device
                batch_data = self._move_batch_to_device(batch_data)
                
                # Forward pass
                outputs = self.model(
                    batch_data['metrics'],
                    batch_data['graphs'][0],
                    batch_data['logs'],
                    return_intermediate=True
                )
                
                # Compute losses
                losses = self.loss_function(
                    outputs,
                    batch_data['labels'],
                    outputs['intermediate']['temporal_features'] if 'intermediate' in outputs else None,
                    self.prev_temporal_features
                )
                
                # Store losses
                for loss_name, loss_value in losses.items():
                    epoch_losses[loss_name].append(loss_value.item())
        
        # Compute epoch averages
        epoch_metrics = {}
        for loss_name, loss_values in epoch_losses.items():
            epoch_metrics[f'val_{loss_name}'] = np.mean(loss_values)
        
        return epoch_metrics
    
    def _move_batch_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to training device."""
        device_batch = {}
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                # Handle list of graphs or other objects
                device_batch[key] = value
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics for current epoch."""
        # Update history
        for metric, value in train_metrics.items():
            self.train_history[metric].append(value)
        
        for metric, value in val_metrics.items():
            self.val_history[metric].append(value)
        
        # Log to console
        logger.info(f"Epoch {self.current_epoch}: "
                   f"Train Loss: {train_metrics.get('train_total_loss', 0):.4f}, "
                   f"Val Loss: {val_metrics.get('val_total_loss', 0):.4f}, "
                   f"LR: {self.lr_scheduler.get_last_lr()[0]:.6f}")
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {**train_metrics, **val_metrics}
            log_dict['epoch'] = self.current_epoch
            log_dict['learning_rate'] = self.lr_scheduler.get_last_lr()[0]
            wandb.log(log_dict)
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.scheduler.state_dict(),
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")
    
    def _save_final_model(self):
        """Save final trained model."""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        final_model_path = model_dir / "ocean_final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }, final_model_path)
        
        logger.info(f"Saved final model to {final_model_path}")
    
    def setup_online_learning(self) -> OnlineLearner:
        """Setup online learning mode."""
        self.online_learner = OnlineLearner(
            model=self.model,
            config=self.config,
            optimizer=self.optimizer
        )
        
        logger.info("Setup online learning mode")
        return self.online_learner
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.train_history:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('OCEAN Training History')
        
        # Plot losses
        epochs = range(len(self.train_history['train_total_loss']))
        
        # Total loss
        axes[0, 0].plot(epochs, self.train_history['train_total_loss'], label='Train')
        axes[0, 0].plot(epochs, self.val_history['val_total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Prediction loss
        axes[0, 1].plot(epochs, self.train_history['train_prediction_loss'], label='Train')
        axes[0, 1].plot(epochs, self.val_history['val_prediction_loss'], label='Validation')
        axes[0, 1].set_title('Prediction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Contrastive loss
        if 'train_total_contrastive_loss' in self.train_history:
            axes[1, 0].plot(epochs, self.train_history['train_total_contrastive_loss'], label='Train')
            axes[1, 0].plot(epochs, self.val_history['val_total_contrastive_loss'], label='Validation')
            axes[1, 0].set_title('Contrastive Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if hasattr(self.lr_scheduler.scheduler, '_last_lr'):
            lr_history = [self.lr_scheduler.scheduler._last_lr[0] for _ in epochs]
            axes[1, 1].plot(epochs, lr_history)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training plots to {save_path}")
        
        plt.show()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.train_history = defaultdict(list, checkpoint['train_history'])
        self.val_history = defaultdict(list, checkpoint['val_history'])
        self.best_val_loss = checkpoint['val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'model_info': self.model.get_model_summary() if hasattr(self.model, 'get_model_summary') else {},
            'training_config': {
                'num_epochs': self.current_epoch,
                'learning_rate': self.config.training.learning_rate,
                'batch_size': self.config.data.batch_size,
                'optimizer': type(self.optimizer).__name__,
                'scheduler': self.lr_scheduler.scheduler_type
            },
            'performance': {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.train_history['train_total_loss'][-1] if self.train_history else None,
                'final_val_loss': self.val_history['val_total_loss'][-1] if self.val_history else None
            },
            'data_info': self.data_loader.get_split_statistics(),
            'device': str(self.device)
        }
        
        if self.training_start_time:
            training_duration = datetime.now() - self.training_start_time
            summary['training_duration'] = str(training_duration)
        
        return summary