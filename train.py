#!/usr/bin/env python3
"""
Multimodal Stock Prediction Training Script
==========================================

Train the multimodal stock prediction model using real financial data.
Combines stock prices, financial news, and employment data for comprehensive prediction.
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
from datetime import datetime

# Import project modules
from main import MultiModalStockModel, ModelConfig
from data.real_data_fetcher import DataConfig
from data.real_dataset_builder import RealDatasetBuilder
from training.trainer import Trainer, TrainingConfig
from training.loss_functions import LossConfig
from training.metrics import MetricsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for production training"""
    parser = argparse.ArgumentParser(description="Train Multimodal Stock Prediction Model")
    
    # Data configuration
    parser.add_argument('--start_date', type=str, default='2021-01-01', 
                       help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for training data (YYYY-MM-DD). Current date if not specified.')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
                       help='Stock symbols to include in training')
    parser.add_argument('--cache_dir', type=str, default='data_cache',
                       help='Directory for caching fetched data')
    parser.add_argument('--cache_expiry_hours', type=int, default=24,
                       help='Hours before cached data expires')
    parser.add_argument('--max_cache_size_mb', type=int, default=100,
                       help='Maximum cache size in MB (default: 100MB)')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear all cache files before training')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--sequence_length', type=int, default=252, help='Input sequence length (trading days)')
    parser.add_argument('--fusion_strategy', type=str, default='cross_attention',
                       choices=['cross_attention', 'gated_fusion', 'hierarchical', 'adaptive'],
                       help='Fusion strategy for multimodal data')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'], help='LR scheduler')
    
    # Loss configuration
    parser.add_argument('--return_loss_weight', type=float, default=1.0,
                       help='Weight for return prediction loss')
    parser.add_argument('--volatility_loss_weight', type=float, default=0.5,
                       help='Weight for volatility prediction loss')
    parser.add_argument('--direction_loss_weight', type=float, default=0.3,
                       help='Weight for direction classification loss')
    parser.add_argument('--economic_loss_weight', type=float, default=0.2,
                       help='Weight for economic indicator loss')
    
    # Optimization flags
    parser.add_argument('--optimize_for_mac', action='store_true',
                       help='Optimize multi-threading for MacBook hardware')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loader workers (auto-detected if not specified)')
    parser.add_argument('--pin_memory', action='store_true',
                       help='Pin memory for faster data transfer')
    parser.add_argument('--persistent_workers', action='store_true',
                       help='Keep data loading workers alive between epochs')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for tracking')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    # Evaluation flags
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation on test set')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save model predictions to file')
    
    return parser.parse_args()


def setup_multiprocessing_optimization(args):
    """Setup optimal multiprocessing settings for hardware"""
    import os
    import multiprocessing as mp
    
    cpu_count = mp.cpu_count()
    
    if args.optimize_for_mac:
        logger.info("🍎 Optimizing for MacBook hardware...")
        
        # Set PyTorch threading settings
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon optimization
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(min(4, cpu_count // 2))
            logger.info(f"   Apple Silicon: {cpu_count} threads, {min(4, cpu_count // 2)} interop threads")
        else:
            # Intel Mac optimization
            torch.set_num_threads(min(8, cpu_count))
            torch.set_num_interop_threads(min(4, cpu_count // 4))
            logger.info(f"   Intel Mac: {min(8, cpu_count)} threads, {min(4, cpu_count // 4)} interop threads")
        
        # Set optimal environment variables
        os.environ['OMP_NUM_THREADS'] = str(min(8, cpu_count))
        os.environ['MKL_NUM_THREADS'] = str(min(8, cpu_count))
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(min(8, cpu_count))
        
        # Auto-detect optimal num_workers
        if args.num_workers is None:
            args.num_workers = min(4, max(2, cpu_count // 4))
        
        # Enable optimizations
        args.pin_memory = True
        args.persistent_workers = True
        
        logger.info(f"   Data loading: {args.num_workers} workers, pin_memory={args.pin_memory}")
    
    else:
        if args.num_workers is None:
            args.num_workers = min(4, max(2, cpu_count // 2))
    
    return args


def create_dataset(args):
    """Create dataset from real financial data"""
    logger.info("📊 Creating dataset from real financial data")
    
    # Create data configuration
    data_config = DataConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
        cache_dir=args.cache_dir,
        cache_expiry_hours=args.cache_expiry_hours,
        max_cache_size_mb=args.max_cache_size_mb
    )
    
    # Handle cache clearing if requested
    if args.clear_cache:
        from data.real_data_fetcher import DataCache
        cache = DataCache(args.cache_dir, args.max_cache_size_mb)
        cache.clear_cache()
        logger.info("🧹 Cache cleared successfully")
    
    # Create dataset builder
    dataset_builder = RealDatasetBuilder(
        config=data_config,
        sequence_length=args.sequence_length
    )
    
    # Build dataset
    logger.info(f"   📈 Symbols: {', '.join(args.symbols)}")
    logger.info(f"   📅 Date range: {args.start_date} to {args.end_date or 'current'}")
    logger.info(f"   📏 Sequence length: {args.sequence_length} days")
    
    dataset_splits = dataset_builder.build_real_dataset()
    
    # Create data loaders
    data_loaders = dataset_builder.create_data_loaders(
        dataset_splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )
    
    # Log dataset statistics
    total_samples = sum(split_info['size'] for split_info in dataset_splits.values())
    logger.info(f"✅ Dataset created:")
    logger.info(f"   📊 Total samples: {total_samples:,}")
    logger.info(f"   📈 Train: {dataset_splits['train']['size']:,}")
    logger.info(f"   📊 Validation: {dataset_splits['val']['size']:,}")
    logger.info(f"   📋 Test: {dataset_splits['test']['size']:,}")
    
    return data_loaders, dataset_splits


def train_model(args):
    """Main training function"""
    logger.info("🚀 Starting training with real financial data")
    
    # Setup hardware optimization
    args = setup_multiprocessing_optimization(args)
    
    # Create experiment directory
    if args.experiment_name:
        experiment_dir = Path(args.checkpoint_dir) / args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(args.checkpoint_dir) / f"experiment_{timestamp}"
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    logger.info(f"💾 Experiment directory: {experiment_dir}")
    
    # Create dataset
    data_loaders, dataset_splits = create_dataset(args)
    
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
    logger.info("🧠 Initializing multimodal model...")
    model = MultiModalStockModel(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create training configuration
    loss_config = LossConfig(
        return_loss_weight=args.return_loss_weight,
        volatility_loss_weight=args.volatility_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        economic_loss_weight=args.economic_loss_weight
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
    logger.info("🏋️ Initializing trainer...")
    trainer = Trainer(model, training_config)
    
    # Log training setup
    logger.info("⚙️  Training Configuration:")
    logger.info(f"   📊 Epochs: {args.epochs}")
    logger.info(f"   📦 Batch size: {args.batch_size}")
    logger.info(f"   📈 Learning rate: {args.learning_rate}")
    logger.info(f"   🔧 Optimizer: {args.optimizer}")
    logger.info(f"   📅 Scheduler: {args.scheduler}")
    logger.info(f"   🖥️  Device: {trainer.device}")
    logger.info(f"   ⚡ Mixed precision: {args.mixed_precision}")
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Run training or evaluation
    if args.eval_only:
        logger.info("📊 Running evaluation only...")
        test_metrics = trainer.validate_epoch(data_loaders['test'])
        logger.info(f"✅ Test metrics: {test_metrics}")
        
        if args.save_predictions:
            predictions = trainer.predict(data_loaders['test'])
            pred_path = experiment_dir / "test_predictions.pth"
            torch.save(predictions, pred_path)
            logger.info(f"💾 Predictions saved to: {pred_path}")
    
    else:
        logger.info("🚀 Starting training...")
        
        # Record training start
        start_time = datetime.now()
        logger.info(f"   🕐 Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Train the model
        try:
            training_history = trainer.train(
                data_loaders['train'],
                data_loaders['val']
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"   ⏱️  Training completed in: {duration}")
            
            # Evaluate on test set
            logger.info("📊 Evaluating on test set...")
            test_metrics = trainer.validate_epoch(data_loaders['test'])
            
            # Save comprehensive results
            results = {
                'experiment_config': vars(args),
                'model_config': model_config.__dict__,
                'training_config': training_config.__dict__,
                'dataset_info': {
                    'total_samples': sum(split_info['size'] for split_info in dataset_splits.values()),
                    'train_samples': dataset_splits['train']['size'],
                    'val_samples': dataset_splits['val']['size'],
                    'test_samples': dataset_splits['test']['size'],
                    'symbols': args.symbols,
                    'date_range': f"{args.start_date} to {args.end_date or 'current'}"
                },
                'training_history': training_history,
                'test_metrics': test_metrics,
                'training_duration': str(duration),
                'training_start': start_time.isoformat(),
                'training_end': end_time.isoformat()
            }
            
            results_path = experiment_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Training completed!")
            logger.info(f"   📊 Final test metrics: {test_metrics}")
            logger.info(f"   💾 Results saved to: {results_path}")
            
            # Save predictions if requested
            if args.save_predictions:
                predictions = trainer.predict(data_loaders['test'])
                pred_path = experiment_dir / "test_predictions.pth"
                torch.save(predictions, pred_path)
                logger.info(f"   💾 Predictions saved to: {pred_path}")
            
        except KeyboardInterrupt:
            logger.info("⚠️  Training interrupted by user")
            return
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise


def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("🏭 MULTIMODAL STOCK PREDICTION - TRAINING")
    logger.info("=" * 80)
    
    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()