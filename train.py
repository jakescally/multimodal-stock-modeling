#!/usr/bin/env python3
"""
Training Script for Multimodal Stock Prediction Model
====================================================

Main training script that puts everything together:
- Data loading and preprocessing
- Model initialization
- Training loop execution
- Model evaluation and saving
- Experiment tracking
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import project modules
from main import MultiModalStockModel, ModelConfig
from data.unified_dataset import UnifiedDatasetBuilder, MultiModalDataset
from training.trainer import Trainer, TrainingConfig
from training.loss_functions import LossConfig
from training.metrics import MetricsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_multiprocessing_optimization(args):
    """Setup optimal multiprocessing settings for MacBook hardware"""
    import os
    import multiprocessing as mp
    
    # Get system info
    cpu_count = mp.cpu_count()
    
    if args.optimize_for_mac:
        logger.info("Optimizing for MacBook hardware...")
        
        # Set PyTorch threading settings for Apple Silicon/Intel Mac
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon optimization
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(min(4, cpu_count // 2))
            logger.info(f"Apple Silicon detected: set {cpu_count} threads, {min(4, cpu_count // 2)} interop threads")
        else:
            # Intel Mac optimization
            torch.set_num_threads(min(8, cpu_count))
            torch.set_num_interop_threads(min(4, cpu_count // 4))
            logger.info(f"Intel Mac detected: set {min(8, cpu_count)} threads, {min(4, cpu_count // 4)} interop threads")
        
        # Set optimal environment variables for Mac
        os.environ['OMP_NUM_THREADS'] = str(min(8, cpu_count))
        os.environ['MKL_NUM_THREADS'] = str(min(8, cpu_count))
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(min(8, cpu_count))
        
        # Auto-detect optimal num_workers if not specified
        if args.num_workers is None:
            # For Mac, optimal is usually 2-4 workers to avoid overhead
            args.num_workers = min(4, max(2, cpu_count // 4))
            logger.info(f"Auto-detected optimal num_workers: {args.num_workers}")
        
        # Enable optimizations
        args.pin_memory = True
        args.persistent_workers = True
        
        logger.info("Mac optimization settings applied")
    
    else:
        # Default settings
        if args.num_workers is None:
            args.num_workers = min(4, max(2, cpu_count // 2))
            logger.info(f"Auto-detected num_workers: {args.num_workers}")
    
    logger.info(f"Final settings - num_workers: {args.num_workers}, pin_memory: {args.pin_memory}, persistent_workers: {args.persistent_workers}")
    return args


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Multimodal Stock Prediction Model")
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--sequence_length', type=int, default=252, help='Input sequence length')
    parser.add_argument('--fusion_strategy', type=str, default='cross_attention', 
                       choices=['cross_attention', 'gated_fusion', 'hierarchical', 'adaptive'],
                       help='Fusion strategy')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'], help='LR scheduler')
    
    # Data configuration
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], 
                       help='Stock symbols to train on')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date for data')
    parser.add_argument('--end_date', type=str, default=None, help='End date for data')
    parser.add_argument('--use_mock_data', action='store_true', help='Use mock data for testing')
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    # Multi-threading optimization
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loader workers (auto-detected if not specified)')
    parser.add_argument('--optimize_for_mac', action='store_true', help='Optimize multi-threading for MacBook hardware')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster data transfer')
    parser.add_argument('--persistent_workers', action='store_true', help='Keep data loading workers alive between epochs')
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to file')
    
    return parser.parse_args()


def create_mock_dataset(config: ModelConfig, batch_size: int, num_workers: int = 2, 
                       pin_memory: bool = False, persistent_workers: bool = False) -> Dict[str, DataLoader]:
    """Create mock dataset for testing"""
    logger.info("Creating mock dataset for testing")
    
    # Create mock data
    n_samples = 1000
    seq_len = config.sequence_length
    
    # Mock features
    mock_features = {
        'stock': torch.randn(n_samples, seq_len, 15),
        'news': torch.randn(n_samples, seq_len, 10),
        'employment': torch.randn(n_samples, seq_len, 8)
    }
    
    # Mock targets
    mock_targets = {}
    for horizon in config.prediction_horizons:
        # Generate realistic-looking returns (small values)
        mock_targets[f'horizon_{horizon}'] = torch.randn(n_samples) * 0.1
    
    # Create datasets
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    train_features = {k: v[:train_end] for k, v in mock_features.items()}
    train_targets = {k: v[:train_end] for k, v in mock_targets.items()}
    
    val_features = {k: v[train_end:val_end] for k, v in mock_features.items()}
    val_targets = {k: v[train_end:val_end] for k, v in mock_targets.items()}
    
    test_features = {k: v[val_end:] for k, v in mock_features.items()}
    test_targets = {k: v[val_end:] for k, v in mock_targets.items()}
    
    # Create DataLoaders
    train_dataset = MultiModalDataset(train_features, train_targets, config.sequence_length)
    val_dataset = MultiModalDataset(val_features, val_targets, config.sequence_length)
    test_dataset = MultiModalDataset(test_features, test_targets, config.sequence_length)
    
    # DataLoader optimization settings
    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers and num_workers > 0
    }
    
    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    }
    
    logger.info(f"Created mock dataset: Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return data_loaders


def load_real_dataset(symbols: list, start_date: str, end_date: str, 
                     config: ModelConfig, batch_size: int, num_workers: int = 4,
                     pin_memory: bool = False, persistent_workers: bool = False) -> Dict[str, DataLoader]:
    """Load real financial dataset"""
    logger.info(f"Loading real dataset for symbols: {symbols}")
    
    # Initialize dataset builder
    builder = UnifiedDatasetBuilder(sequence_length=config.sequence_length)
    
    # Build dataset
    dataset_splits = builder.build_complete_dataset(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        include_news=True,
        include_employment=True
    )
    
    # Create DataLoaders with optimization settings
    data_loaders = builder.create_data_loaders(
        dataset_splits, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    logger.info("Real dataset loaded successfully")
    return data_loaders


def train_model(args):
    """Main training function"""
    # Setup multiprocessing optimization
    args = setup_multiprocessing_optimization(args)
    
    # Setup experiment directory
    if args.experiment_name:
        experiment_dir = Path(args.checkpoint_dir) / args.experiment_name
    else:
        experiment_dir = Path(args.checkpoint_dir) / f"experiment_{torch.randint(0, 10000, (1,)).item()}"
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(experiment_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create model configuration
    model_config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        sequence_length=args.sequence_length,
        prediction_horizons=[30, 180, 365, 730],
        fusion_strategy=args.fusion_strategy,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = MultiModalStockModel(model_config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {total_params:,} total parameters, "
               f"{trainable_params:,} trainable")
    
    # Load dataset
    if args.use_mock_data:
        data_loaders = create_mock_dataset(
            model_config, args.batch_size, 
            args.num_workers, args.pin_memory, args.persistent_workers
        )
    else:
        data_loaders = load_real_dataset(
            args.symbols, args.start_date, args.end_date, 
            model_config, args.batch_size,
            args.num_workers, args.pin_memory, args.persistent_workers
        )
    
    # Create training configuration
    loss_config = LossConfig(
        return_loss_weight=1.0,
        volatility_loss_weight=0.5,
        direction_loss_weight=0.3,
        economic_loss_weight=0.2
    )
    
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        checkpoint_dir=str(experiment_dir),
        device=args.device,
        mixed_precision=args.mixed_precision,
        loss_config=loss_config
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(model, training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Run training or evaluation
    if args.eval_only:
        logger.info("Running evaluation only...")
        test_metrics = trainer.validate_epoch(data_loaders['test'])
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions = trainer.predict(data_loaders['test'])
            torch.save(predictions, experiment_dir / "test_predictions.pth")
            logger.info("Test predictions saved")
    
    else:
        logger.info("Starting training...")
        
        # Train the model
        training_history = trainer.train(
            data_loaders['train'], 
            data_loaders['val']
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.validate_epoch(data_loaders['test'])
        
        # Save final results
        results = {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__
        }
        
        with open(experiment_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed! Results saved to {experiment_dir}")
        logger.info(f"Final test metrics: {test_metrics}")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions = trainer.predict(data_loaders['test'])
            torch.save(predictions, experiment_dir / "test_predictions.pth")
            logger.info("Test predictions saved")


def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()