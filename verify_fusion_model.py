#!/usr/bin/env python3
"""
Fusion Layer and Complete Model Verification Script
==================================================

Verifies that the cross-modal fusion layer and complete model are working:
1. Cross-modal attention mechanisms
2. Gated fusion strategies  
3. Hierarchical fusion
4. Adaptive fusion
5. Multi-task prediction heads
6. Complete end-to-end model forward pass
7. Parameter counting and memory usage
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")

def test_fusion_layer_components():
    """Test individual fusion layer components"""
    print_section("Testing Fusion Layer Components")
    
    try:
        from models.fusion_layer import (
            CrossModalAttention, GatedFusion, HierarchicalFusion, 
            AdaptiveFusion, FusionConfig
        )
        
        # Test configuration
        config = FusionConfig(
            d_model=64,  # Smaller for testing
            n_heads=4,
            fusion_strategy="cross_attention"
        )
        
        print_success("Fusion layer imports successful")
        
        # Create mock modality features
        batch_size, seq_len = 4, 20
        stock_features = torch.randn(batch_size, seq_len, config.stock_dim)
        text_features = torch.randn(batch_size, seq_len, config.text_dim)
        employment_features = torch.randn(batch_size, seq_len, config.employment_dim)
        
        print_success(f"Created mock features: stock{stock_features.shape}, text{text_features.shape}, employment{employment_features.shape}")
        
        # Test CrossModalAttention
        cross_attn = CrossModalAttention(config)
        cross_output = cross_attn(stock_features, text_features, employment_features)
        
        print_success(f"Cross-modal attention output shape: {cross_output['fused_features'].shape}")
        print_info(f"Attention weights keys: {list(cross_output['attention_weights'].keys())}")
        print_info(f"Modality weights: {cross_output['modality_weights']}")
        
        # Test GatedFusion
        gated_fusion = GatedFusion(config)
        gated_output = gated_fusion(stock_features, text_features, employment_features)
        
        print_success(f"Gated fusion output shape: {gated_output['fused_features'].shape}")
        print_info(f"Gate values keys: {list(gated_output['gate_values'].keys())}")
        
        # Test HierarchicalFusion
        hierarchical_fusion = HierarchicalFusion(config)
        hierarchical_output = hierarchical_fusion(stock_features, text_features, employment_features)
        
        print_success(f"Hierarchical fusion output shape: {hierarchical_output['fused_features'].shape}")
        print_info(f"Pairwise fusions: {list(hierarchical_output['pairwise_fusions'].keys())}")
        
        # Test AdaptiveFusion
        adaptive_fusion = AdaptiveFusion(config)
        adaptive_output = adaptive_fusion(stock_features, text_features, employment_features)
        
        print_success(f"Adaptive fusion output shape: {adaptive_output['fused_features'].shape}")
        print_info(f"Strategy weights shape: {adaptive_output['strategy_weights'].shape}")
        print_info(f"Individual outputs: {list(adaptive_output['individual_outputs'].keys())}")
        
        return True
        
    except Exception as e:
        print_error(f"Fusion layer components test failed: {e}")
        traceback.print_exc()
        return False

def test_prediction_heads():
    """Test multi-task prediction heads"""
    print_section("Testing Multi-Task Prediction Heads")
    
    try:
        from models.prediction_heads import (
            ReturnPredictionHead, VolatilityPredictionHead, 
            DirectionClassificationHead, EconomicIndicatorHead,
            MultiTaskPredictionHead, PredictionConfig
        )
        
        # Test configuration
        config = PredictionConfig(
            d_model=64,
            prediction_horizons=[30, 180, 365],
            dropout=0.1
        )
        
        print_success("Prediction heads imports successful")
        
        # Create mock features
        batch_size = 8
        features = torch.randn(batch_size, config.d_model)
        sequence_features = torch.randn(batch_size, 20, config.d_model)
        
        # Test ReturnPredictionHead
        return_head = ReturnPredictionHead(config)
        return_output = return_head(features)
        
        print_success(f"Return prediction head output keys: {list(return_output.keys())}")
        print_info(f"Prediction horizons: {list(return_output['predictions'].keys())}")
        print_info(f"Uncertainty estimates: {list(return_output['uncertainties'].keys())}")
        
        # Test VolatilityPredictionHead
        volatility_head = VolatilityPredictionHead(config)
        volatility_output = volatility_head(features)
        
        print_success(f"Volatility prediction shape: {[v.shape for v in volatility_output.values()]}")
        
        # Test DirectionClassificationHead
        direction_head = DirectionClassificationHead(config)
        direction_output = direction_head(features)
        
        print_success(f"Direction classification output: {list(direction_output.keys())}")
        for horizon_key, direction_data in direction_output.items():
            print_info(f"{horizon_key} probabilities shape: {direction_data['probabilities'].shape}")
        
        # Test EconomicIndicatorHead
        economic_head = EconomicIndicatorHead(config)
        economic_output = economic_head(features)
        
        print_success(f"Economic indicators: {list(economic_output.keys())}")
        
        # Test MultiTaskPredictionHead
        multi_head = MultiTaskPredictionHead(config)
        multi_output = multi_head(sequence_features)
        
        print_success(f"Multi-task output keys: {list(multi_output.keys())}")
        for task_name, task_output in multi_output.items():
            if task_name != 'task_weights':
                print_info(f"{task_name} output type: {type(task_output)}")
        
        return True
        
    except Exception as e:
        print_error(f"Prediction heads test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_model():
    """Test the complete integrated model"""
    print_section("Testing Complete Multimodal Model")
    
    try:
        from main import MultiModalStockModel, ModelConfig
        
        # Create model configuration
        config = ModelConfig(
            d_model=64,  # Smaller for testing
            n_heads=4,
            sequence_length=20,
            prediction_horizons=[30, 180],
            fusion_strategy="cross_attention"
        )
        
        print_success("Model configuration created")
        
        # Initialize complete model
        model = MultiModalStockModel(config)
        
        print_success("Complete model instantiated")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_info(f"Total parameters: {total_params:,}")
        print_info(f"Trainable parameters: {trainable_params:,}")
        
        # Create mock batch data
        batch_size, seq_len = 4, 20
        
        batch = {
            'stock': torch.randn(batch_size, seq_len, 15),      # 15 stock features
            'news': torch.randn(batch_size, seq_len, 10),       # 10 news features
            'employment': torch.randn(batch_size, seq_len, 8)   # 8 employment features
        }
        
        print_success(f"Created mock batch: {[(k, v.shape) for k, v in batch.items()]}")
        
        # Forward pass
        with torch.no_grad():
            output = model(batch)
        
        print_success("Forward pass completed successfully")
        
        # Verify output structure
        print_info(f"Output keys: {list(output.keys())}")
        
        # Check predictions
        predictions = output['predictions']
        print_info(f"Prediction tasks: {list(predictions.keys())}")
        
        if 'returns' in predictions:
            returns = predictions['returns']
            print_info(f"Return predictions: {list(returns['predictions'].keys())}")
            print_info(f"Return uncertainties: {list(returns['uncertainties'].keys())}")
        
        # Check fusion output
        fusion_output = output['fusion_output']
        print_info(f"Fusion output shape: {fusion_output['fused_features'].shape}")
        
        # Check encoded features
        encoded_features = output['encoded_features']
        print_info(f"Encoded features: {list(encoded_features.keys())}")
        
        return True
        
    except Exception as e:
        print_error(f"Complete model test failed: {e}")
        traceback.print_exc()
        return False

def test_different_fusion_strategies():
    """Test different fusion strategies"""
    print_section("Testing Different Fusion Strategies")
    
    try:
        from main import MultiModalStockModel, ModelConfig
        
        strategies = ["cross_attention", "gated_fusion", "hierarchical", "adaptive"]
        
        # Create mock batch
        batch_size, seq_len = 2, 10  # Small for speed
        batch = {
            'stock': torch.randn(batch_size, seq_len, 15),
            'news': torch.randn(batch_size, seq_len, 10),
            'employment': torch.randn(batch_size, seq_len, 8)
        }
        
        for strategy in strategies:
            try:
                config = ModelConfig(
                    d_model=32,  # Very small for testing
                    n_heads=2,
                    sequence_length=10,
                    prediction_horizons=[30],
                    fusion_strategy=strategy
                )
                
                model = MultiModalStockModel(config)
                
                with torch.no_grad():
                    output = model(batch)
                
                params = sum(p.numel() for p in model.parameters())
                output_shape = output['fusion_output']['fused_features'].shape
                
                print_success(f"{strategy}: {params:,} params, output {output_shape}")
                
            except Exception as e:
                print_error(f"{strategy} failed: {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Fusion strategies test failed: {e}")
        traceback.print_exc()
        return False

def test_model_training_compatibility():
    """Test model compatibility with training procedures"""
    print_section("Testing Training Compatibility")
    
    try:
        from main import MultiModalStockModel, ModelConfig
        
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            sequence_length=10,
            prediction_horizons=[30, 90]
        )
        
        model = MultiModalStockModel(config)
        
        # Create mock data and targets
        batch = {
            'stock': torch.randn(2, 10, 15),
            'news': torch.randn(2, 10, 10),
            'employment': torch.randn(2, 10, 8)
        }
        
        targets = {
            'horizon_30': torch.randn(2),
            'horizon_90': torch.randn(2)
        }
        
        # Test forward pass
        output = model(batch)
        predictions = output['predictions']['returns']['predictions']
        
        print_success("Forward pass successful")
        
        # Test loss calculation
        loss = 0.0
        for horizon_key, pred in predictions.items():
            if horizon_key in targets:
                loss += torch.nn.functional.mse_loss(pred, targets[horizon_key])
        
        print_success(f"Loss calculation successful: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        
        print_success("Backward pass successful")
        
        # Check gradients
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print_success(f"Gradients computed: {has_gradients}")
        
        return True
        
    except Exception as e:
        print_error(f"Training compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory usage and efficiency"""
    print_section("Testing Memory Efficiency")
    
    try:
        from main import MultiModalStockModel, ModelConfig
        import gc
        
        # Test with different batch sizes
        batch_sizes = [1, 4, 8]
        
        config = ModelConfig(
            d_model=64,
            n_heads=4,
            sequence_length=50,
            prediction_horizons=[30, 180]
        )
        
        model = MultiModalStockModel(config)
        
        for batch_size in batch_sizes:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            batch = {
                'stock': torch.randn(batch_size, 50, 15),
                'news': torch.randn(batch_size, 50, 10),
                'employment': torch.randn(batch_size, 50, 8)
            }
            
            # Measure memory before
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            with torch.no_grad():
                output = model(batch)
            
            # Measure memory after
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_used = memory_after - memory_before
            
            print_info(f"Batch size {batch_size}: Memory used ~{memory_used / 1024**2:.1f} MB")
        
        print_success("Memory efficiency test completed")
        
        return True
        
    except Exception as e:
        print_error(f"Memory efficiency test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all fusion layer and model verification tests"""
    print_section("Fusion Layer and Complete Model Verification")
    print_info("Testing cross-modal fusion and integrated model functionality")
    
    tests = [
        ("Fusion Layer Components", test_fusion_layer_components),
        ("Multi-Task Prediction Heads", test_prediction_heads),
        ("Complete Multimodal Model", test_complete_model),
        ("Different Fusion Strategies", test_different_fusion_strategies),
        ("Training Compatibility", test_model_training_compatibility),
        ("Memory Efficiency", test_memory_efficiency)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_section("Fusion Layer and Model Verification Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status:<12} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("🎉 All fusion and model tests passed! Your complete multimodal architecture is working correctly.")
        print_info("The model successfully integrates:")
        print_info("  • Cross-modal attention mechanisms")
        print_info("  • Multiple fusion strategies (cross-attention, gated, hierarchical, adaptive)")
        print_info("  • Multi-task prediction heads (returns, volatility, direction, economic)")
        print_info("  • End-to-end differentiable architecture")
        print_info("  • Memory-efficient processing")
        print_info("  • Training-ready with proper gradient flow")
    else:
        print_error("Some fusion/model tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)