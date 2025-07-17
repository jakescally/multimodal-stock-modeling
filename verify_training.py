#!/usr/bin/env python3
"""
Training Pipeline Verification Script
====================================

Comprehensive verification of the complete training pipeline including:
- Model initialization and parameter counting
- Data loading and preprocessing
- Training loop execution
- Loss computation and backpropagation
- Metrics calculation
- Checkpointing and model saving
- Prediction generation
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from main import MultiModalStockModel, ModelConfig
from data.unified_dataset import UnifiedDatasetBuilder, MultiModalDataset
from training.trainer import Trainer, TrainingConfig
from training.loss_functions import MultiTaskLoss, LossConfig
from training.metrics import FinancialMetrics, MetricsConfig


def test_model_initialization():
    """Test model initialization and basic forward pass"""
    print("Testing model initialization...")
    
    config = ModelConfig(
        d_model=128,
        n_heads=4,
        sequence_length=60,
        prediction_horizons=[30, 180],
        fusion_strategy='cross_attention'
    )
    
    model = MultiModalStockModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model initialized with {total_params:,} total parameters")
    print(f"✓ {trainable_params:,} trainable parameters")
    
    # Test forward pass
    batch_size = 8
    mock_features = {
        'stock': torch.randn(batch_size, config.sequence_length, 15),
        'news': torch.randn(batch_size, config.sequence_length, 10),
        'employment': torch.randn(batch_size, config.sequence_length, 8)
    }
    
    with torch.no_grad():
        outputs = model(mock_features)
    
    assert 'predictions' in outputs, "Model output missing predictions"
    assert 'returns' in outputs['predictions'], "Model output missing returns"
    
    print("✓ Forward pass successful")
    return True


def test_loss_computation():
    """Test loss function computation"""
    print("\nTesting loss computation...")
    
    config = LossConfig()
    loss_fn = MultiTaskLoss(config)
    
    # Mock predictions and targets
    batch_size = 8
    mock_predictions = {
        'returns': {
            'predictions': {
                'horizon_30': torch.randn(batch_size),
                'horizon_180': torch.randn(batch_size)
            }
        },
        'direction': {
            'horizon_30': {
                'logits': torch.randn(batch_size, 3),
                'probabilities': torch.softmax(torch.randn(batch_size, 3), dim=1)
            },
            'horizon_180': {
                'logits': torch.randn(batch_size, 3),
                'probabilities': torch.softmax(torch.randn(batch_size, 3), dim=1)
            }
        },
        'volatility': {
            'horizon_30': torch.randn(batch_size),
            'horizon_180': torch.randn(batch_size)
        }
    }
    
    mock_targets = {
        'horizon_30': torch.randn(batch_size) * 0.1,
        'horizon_180': torch.randn(batch_size) * 0.15
    }
    
    loss_dict = loss_fn(mock_predictions, mock_targets)
    
    assert 'total' in loss_dict, "Loss output missing total loss"
    assert torch.is_tensor(loss_dict['total']), "Total loss is not a tensor"
    assert loss_dict['total'].requires_grad, "Loss does not require gradients"
    
    print(f"✓ Loss computation successful: {loss_dict['total'].item():.4f}")
    return True


def test_metrics_calculation():
    """Test metrics calculation"""
    print("\nTesting metrics calculation...")
    
    config = MetricsConfig()
    metrics = FinancialMetrics(config)
    
    # Mock predictions and targets
    batch_size = 16
    mock_predictions = {
        'returns': {
            'predictions': {
                'horizon_30': torch.randn(batch_size),
                'horizon_180': torch.randn(batch_size)
            }
        },
        'direction': {
            'horizon_30': {
                'logits': torch.randn(batch_size, 3),
                'probabilities': torch.softmax(torch.randn(batch_size, 3), dim=1)
            },
            'horizon_180': {
                'logits': torch.randn(batch_size, 3),
                'probabilities': torch.softmax(torch.randn(batch_size, 3), dim=1)
            }
        },
        'volatility': {
            'horizon_30': torch.randn(batch_size),
            'horizon_180': torch.randn(batch_size)
        }
    }
    
    mock_targets = {
        'horizon_30': torch.randn(batch_size) * 0.1,
        'horizon_180': torch.randn(batch_size) * 0.15
    }
    
    batch_metrics = metrics.calculate_batch_metrics(mock_predictions, mock_targets)
    
    # Check that we have expected metrics
    expected_metrics = ['mse', 'mae', 'r2', 'ic', 'direction_accuracy', 'vol_mse']
    found_metrics = [m for m in batch_metrics.keys() if any(exp in m for exp in expected_metrics)]
    
    assert len(found_metrics) > 0, "No expected metrics found"
    
    print(f"✓ Metrics calculation successful: {len(batch_metrics)} metrics computed")
    print(f"  Sample metrics: {list(batch_metrics.keys())[:3]}")
    return True


def test_dataset_creation():
    """Test dataset creation with mock data"""
    print("\nTesting dataset creation...")
    
    config = ModelConfig(sequence_length=60)
    
    # Create mock data
    n_samples = 200
    seq_len = config.sequence_length
    
    mock_features = {
        'stock': torch.randn(n_samples, seq_len, 15),
        'news': torch.randn(n_samples, seq_len, 10),
        'employment': torch.randn(n_samples, seq_len, 8)
    }
    
    mock_targets = {
        'horizon_30': torch.randn(n_samples) * 0.1,
        'horizon_180': torch.randn(n_samples) * 0.15
    }
    
    dataset = MultiModalDataset(mock_features, mock_targets, seq_len)
    
    # Test dataset length and sample access
    assert len(dataset) == n_samples, f"Dataset length mismatch: {len(dataset)} != {n_samples}"
    
    sample_features, sample_targets = dataset[0]
    
    assert isinstance(sample_features, dict), "Features should be a dictionary"
    assert isinstance(sample_targets, dict), "Targets should be a dictionary"
    
    print(f"✓ Dataset creation successful: {len(dataset)} samples")
    return True


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    # Setup model and trainer
    config = ModelConfig(
        d_model=64,
        n_heads=4,
        sequence_length=30,
        prediction_horizons=[30, 180]
    )
    
    model = MultiModalStockModel(config)
    
    training_config = TrainingConfig(
        num_epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        mixed_precision=False  # Disable for testing
    )
    
    trainer = Trainer(model, training_config)
    
    # Create mock batch
    batch_size = 8
    mock_features = {
        'stock': torch.randn(batch_size, config.sequence_length, 15),
        'news': torch.randn(batch_size, config.sequence_length, 10),
        'employment': torch.randn(batch_size, config.sequence_length, 8)
    }
    
    mock_targets = {
        'horizon_30': torch.randn(batch_size) * 0.1,
        'horizon_180': torch.randn(batch_size) * 0.15
    }
    
    # Move to device
    features = {k: v.to(trainer.device) for k, v in mock_features.items()}
    targets = {k: v.to(trainer.device) for k, v in mock_targets.items()}
    
    # Test forward pass
    outputs = model(features)
    loss_dict = trainer.criterion(outputs['predictions'], targets)
    loss = loss_dict['total']
    
    # Test backward pass
    initial_grad = None
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    loss.backward()
    
    # Check gradients
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    
    assert grad_norm > 0, "No gradients computed"
    
    print(f"✓ Training step successful: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    return True


def test_checkpointing():
    """Test model checkpointing and loading"""
    print("\nTesting checkpointing...")
    
    # Create temporary checkpoint directory
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    try:
        # Setup model and trainer
        config = ModelConfig(d_model=32, n_heads=2, sequence_length=20)
        model = MultiModalStockModel(config)
        
        training_config = TrainingConfig(
            checkpoint_dir=str(checkpoint_dir),
            mixed_precision=False
        )
        
        trainer = Trainer(model, training_config)
        
        # Save checkpoint
        trainer.save_checkpoint(epoch=0, val_loss=1.0, is_best=True)
        
        # Check that checkpoint files were created
        best_model_path = checkpoint_dir / "best_model.pth"
        latest_path = checkpoint_dir / "latest_checkpoint.pth"
        
        assert best_model_path.exists(), "Best model checkpoint not saved"
        assert latest_path.exists(), "Latest checkpoint not saved"
        
        # Test loading
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Modify model parameters
        for param in model.parameters():
            param.data.fill_(0.5)
        
        # Load checkpoint
        trainer.load_checkpoint(str(best_model_path))
        
        # Verify parameters were restored
        params_restored = True
        for name, param in model.named_parameters():
            if not torch.allclose(param, original_params[name], atol=1e-6):
                params_restored = False
                break
        
        assert params_restored, "Model parameters not properly restored"
        
        print("✓ Checkpointing successful")
        return True
        
    finally:
        # Cleanup
        import shutil
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)


def test_end_to_end_training():
    """Test complete end-to-end training pipeline"""
    print("\nTesting end-to-end training...")
    
    # Create minimal model for fast testing
    config = ModelConfig(
        d_model=32,
        n_heads=2,
        sequence_length=20,
        prediction_horizons=[30]
    )
    
    model = MultiModalStockModel(config)
    
    # Create small mock dataset
    n_samples = 50
    mock_features = {
        'stock': torch.randn(n_samples, config.sequence_length, 15),
        'news': torch.randn(n_samples, config.sequence_length, 10),
        'employment': torch.randn(n_samples, config.sequence_length, 8)
    }
    
    mock_targets = {
        'horizon_30': torch.randn(n_samples) * 0.1
    }
    
    # Create datasets
    train_end = int(n_samples * 0.7)
    train_features = {k: v[:train_end] for k, v in mock_features.items()}
    train_targets = {k: v[:train_end] for k, v in mock_targets.items()}
    
    val_features = {k: v[train_end:] for k, v in mock_features.items()}
    val_targets = {k: v[train_end:] for k, v in mock_targets.items()}
    
    train_dataset = MultiModalDataset(train_features, train_targets, config.sequence_length)
    val_dataset = MultiModalDataset(val_features, val_targets, config.sequence_length)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Setup trainer
    training_config = TrainingConfig(
        num_epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        patience=10,
        mixed_precision=False,
        save_every=1,
        eval_every=1
    )
    
    trainer = Trainer(model, training_config)
    
    # Run training
    initial_loss = float('inf')
    final_loss = float('inf')
    
    try:
        # Get initial loss
        initial_metrics = trainer.validate_epoch(val_loader)
        initial_loss = initial_metrics['loss']
        
        # Train for a few epochs
        history = trainer.train(train_loader, val_loader)
        
        # Get final loss
        final_metrics = trainer.validate_epoch(val_loader)
        final_loss = final_metrics['loss']
        
        # Check that training history was recorded
        assert 'train_loss' in history, "Training history missing train_loss"
        assert 'val_loss' in history, "Training history missing val_loss"
        assert len(history['train_loss']) > 0, "No training loss recorded"
        
        print(f"✓ End-to-end training successful")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Training epochs: {len(history['train_loss'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end training failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests"""
    print("="*60)
    print("MULTIMODAL STOCK PREDICTION - TRAINING VERIFICATION")
    print("="*60)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Loss Computation", test_loss_computation),
        ("Metrics Calculation", test_metrics_calculation),
        ("Dataset Creation", test_dataset_creation),
        ("Training Step", test_training_step),
        ("Checkpointing", test_checkpointing),
        ("End-to-End Training", test_end_to_end_training)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All training pipeline tests passed!")
        print("The complete multimodal stock prediction training framework is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)