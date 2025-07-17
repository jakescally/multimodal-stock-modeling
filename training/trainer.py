"""
Training Framework for Multimodal Stock Prediction
=================================================

Comprehensive training framework including:
- Training loop with validation
- Checkpointing and model saving
- Learning rate scheduling
- Early stopping
- Metrics tracking and logging
- Gradient clipping and optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime
import os

from .loss_functions import MultiTaskLoss, LossConfig
from .metrics import FinancialMetrics, MetricsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every: int = 10
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 10
    eval_every: int = 5
    tensorboard_dir: str = "runs"
    
    # Loss configuration
    loss_config: LossConfig = field(default_factory=LossConfig)
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    mixed_precision: bool = True


class Trainer:
    """Main training class for multimodal stock prediction"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Set device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(config.loss_config)
        
        # Initialize metrics
        self.metrics = FinancialMetrics(MetricsConfig())
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Initialize mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
        # Setup logging and checkpointing
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'metrics': []
        }
        
    def _setup_device(self) -> torch.device:
        """Setup computing device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Metal Performance Shaders (MPS)")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        if self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        logger.info(f"Using optimizer: {self.config.optimizer}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config.scheduler.lower() == "none":
            return None
        elif self.config.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
        
        logger.info(f"Using scheduler: {self.config.scheduler}")
        return scheduler
    
    def setup_logging(self):
        """Setup logging and checkpointing directories"""
        # Create directories
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_dir = Path(self.config.tensorboard_dir) / f"experiment_{timestamp}"
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # Save config
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            config_dict = {
                'training_config': self.config.__dict__,
                'model_config': self.model.config.__dict__ if hasattr(self.model, 'config') else {}
            }
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Logging to: {self.tensorboard_dir}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Move data to device
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss_dict = self.criterion(outputs['predictions'], targets)
                    loss = loss_dict['total']
            else:
                outputs = self.model(features)
                loss_dict = self.criterion(outputs['predictions'], targets)
                loss = loss_dict['total']
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            
            # Calculate metrics
            batch_metrics = self.metrics.calculate_batch_metrics(outputs['predictions'], targets)
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Log batch metrics
            if batch_idx % self.config.log_every == 0:
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), self.global_step)
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}")
            
            self.global_step += 1
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        avg_loss = np.mean(epoch_losses)
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_losses = []
        val_metrics = {}
        
        with torch.no_grad():
            for features, targets in val_loader:
                # Move data to device
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                outputs = self.model(features)
                loss_dict = self.criterion(outputs['predictions'], targets)
                loss = loss_dict['total']
                
                val_losses.append(loss.item())
                
                # Calculate metrics
                batch_metrics = self.metrics.calculate_batch_metrics(outputs['predictions'], targets)
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        avg_loss = np.mean(val_losses)
        
        return {'loss': avg_loss, **avg_metrics}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            if epoch % self.config.eval_every == 0:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {'loss': float('inf')}
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log additional metrics
            for key, value in train_metrics.items():
                if key != 'loss':
                    self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                if key != 'loss':
                    self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['metrics'].append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Print progress
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, LR: {current_lr:.2e}")
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_metrics['loss'])
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                if self.config.save_best:
                    self.save_checkpoint(epoch, val_metrics['loss'], is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.writer.close()
        logger.info("Training completed!")
        
        return self.training_history
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            logger.info(f"Saving best model (val_loss: {val_loss:.4f}) to {path}")
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, path)
        
        # Also save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if load_optimizer and self.scaler and 'scaler_state_dict' in checkpoint:
            if checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def predict(self, data_loader: DataLoader) -> Dict[str, List[torch.Tensor]]:
        """Generate predictions on a dataset"""
        self.model.eval()
        predictions = {}
        
        with torch.no_grad():
            for features, _ in data_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                
                outputs = self.model(features)
                batch_preds = outputs['predictions']
                
                # Collect predictions
                for task, task_preds in batch_preds.items():
                    if task not in predictions:
                        predictions[task] = {}
                    
                    if isinstance(task_preds, dict):
                        for subtask, pred in task_preds.items():
                            if subtask not in predictions[task]:
                                predictions[task][subtask] = []
                            predictions[task][subtask].append(pred.cpu())
                    else:
                        if 'predictions' not in predictions[task]:
                            predictions[task]['predictions'] = []
                        predictions[task]['predictions'].append(task_preds.cpu())
        
        # Concatenate predictions
        for task in predictions:
            for subtask in predictions[task]:
                predictions[task][subtask] = torch.cat(predictions[task][subtask], dim=0)
        
        return predictions