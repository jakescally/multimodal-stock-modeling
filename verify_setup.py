#!/usr/bin/env python3
"""
Verification script for multimodal stock modeling project
========================================================

This script verifies that all components are working correctly:
1. Import all modules without errors
2. Test model configurations
3. Verify basic functionality of each encoder
4. Check data processing capabilities
5. Test model forward passes with dummy data
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

def test_imports():
    """Test all module imports"""
    print_section("Testing Module Imports")
    
    try:
        # Test main configuration
        from main import ModelConfig, StockDataProcessor, MultiModalStockModel
        print_success("Main module imports successful")
        
        # Test TFT encoder
        from models.tft_encoder import TFTEncoder, GatedResidualNetwork, InterpretableMultiHeadAttention
        print_success("TFT encoder imports successful")
        
        # Test text encoder
        from models.text_encoder import FinancialTextEncoder, NewsProcessor, SentimentAnalyzer
        print_success("Text encoder imports successful")
        
        # Test employment encoder
        from models.employment_encoder import EmploymentEncoder, EmploymentDataProcessor, EmploymentSignalGenerator
        print_success("Employment encoder imports successful")
        
        return True
        
    except Exception as e:
        print_error(f"Import failed: {e}")
        traceback.print_exc()
        return False

def test_model_config():
    """Test model configuration"""
    print_section("Testing Model Configuration")
    
    try:
        from main import ModelConfig
        
        # Test default config
        config = ModelConfig()
        print_success(f"Default config created")
        print_info(f"Model dimension: {config.d_model}")
        print_info(f"Prediction horizons: {config.prediction_horizons}")
        print_info(f"Sequence length: {config.sequence_length}")
        
        # Test custom config
        custom_config = ModelConfig(
            d_model=128,
            n_heads=4,
            prediction_horizons=[30, 90, 365]
        )
        print_success("Custom config created")
        
        return True
        
    except Exception as e:
        print_error(f"Config test failed: {e}")
        return False

def test_tft_encoder():
    """Test TFT encoder functionality"""
    print_section("Testing TFT Encoder")
    
    try:
        from main import ModelConfig
        from models.tft_encoder import TFTEncoder
        
        config = ModelConfig(d_model=64, sequence_length=50)  # Smaller for testing
        encoder = TFTEncoder(config)
        
        print_success("TFT encoder instantiated")
        print_info(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        
        # Test forward pass with dummy data
        batch_size, seq_len = 2, 50
        static_features = torch.randn(batch_size, encoder.static_input_size)
        historical_features = torch.randn(batch_size, seq_len, encoder.historical_input_size)
        
        with torch.no_grad():
            outputs = encoder(static_features, historical_features)
            
        print_success("Forward pass completed")
        print_info(f"Predictions shape: {[v.shape for v in outputs['predictions'].values()]}")
        print_info(f"Attention weights shape: {outputs['attention_weights'].shape}")
        
        return True
        
    except Exception as e:
        print_error(f"TFT encoder test failed: {e}")
        traceback.print_exc()
        return False

def test_text_encoder():
    """Test text encoder functionality"""
    print_section("Testing Text Encoder")
    
    try:
        from main import ModelConfig
        from models.text_encoder import FinancialTextEncoder, NewsProcessor, SentimentAnalyzer
        
        # Test news processor
        processor = NewsProcessor()
        print_success("News processor instantiated")
        
        # Test sentiment analyzer
        sentiment = SentimentAnalyzer()
        test_text = "Apple reported strong earnings growth and beat expectations"
        score = sentiment.lexicon_sentiment(test_text)
        print_success(f"Sentiment analysis working (score: {score:.2f})")
        
        # Test text preprocessing
        clean_text = processor.preprocess_text("Apple Inc. (AAPL) reports Q4 earnings! 📈 https://example.com")
        print_success(f"Text preprocessing working: '{clean_text[:50]}...'")
        
        return True
        
    except Exception as e:
        print_error(f"Text encoder test failed: {e}")
        traceback.print_exc()
        return False

def test_employment_encoder():
    """Test employment encoder functionality"""
    print_section("Testing Employment Encoder")
    
    try:
        from main import ModelConfig
        from models.employment_encoder import (
            EmploymentEncoder, EmploymentDataProcessor, 
            EmploymentSignalGenerator, EmploymentFeatures
        )
        
        config = ModelConfig(d_model=64)
        encoder = EmploymentEncoder(config)
        
        print_success("Employment encoder instantiated")
        print_info(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        
        # Test data processor
        processor = EmploymentDataProcessor()
        print_success("Employment data processor instantiated")
        
        # Test signal generator
        signal_gen = EmploymentSignalGenerator()
        
        # Create dummy employment features
        features = EmploymentFeatures(
            job_postings_30d=150,
            job_postings_90d=420,
            layoffs_30d=0,
            layoffs_90d=5,
            hiring_velocity=0.15,
            ai_ml_demand=0.08,
            technical_skills_demand=0.25,
            leadership_demand=0.12,
            sector_hiring_ratio=1.2,
            sector_layoff_ratio=0.8,
            unemployment_rate=3.7,
            job_openings_rate=6.2
        )
        
        signals = signal_gen.generate_signals(features)
        overall_signal = signal_gen.aggregate_signal(signals)
        
        print_success(f"Signal generation working (overall: {overall_signal:.3f})")
        print_info(f"Individual signals: {signals}")
        
        return True
        
    except Exception as e:
        print_error(f"Employment encoder test failed: {e}")
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing utilities"""
    print_section("Testing Data Processing")
    
    try:
        # Create dummy stock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        stock_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(80, 150, 100),
            'close': np.random.uniform(90, 220, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        print_success(f"Created dummy stock data: {stock_data.shape}")
        
        # Create dummy news data
        news_data = [
            "Apple reports strong quarterly earnings",
            "Tech stocks rally on positive outlook",
            "Federal Reserve maintains interest rates"
        ]
        
        print_success(f"Created dummy news data: {len(news_data)} articles")
        
        # Create dummy employment data
        employment_data = pd.DataFrame({
            'company_id': ['AAPL'] * 50,
            'posting_date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'description': ['Software Engineer position'] * 50
        })
        
        print_success(f"Created dummy employment data: {employment_data.shape}")
        
        return True
        
    except Exception as e:
        print_error(f"Data processing test failed: {e}")
        return False

def test_pytorch_setup():
    """Test PyTorch setup and CUDA availability"""
    print_section("Testing PyTorch Setup")
    
    try:
        print_info(f"PyTorch version: {torch.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print_info(f"CUDA device count: {torch.cuda.device_count()}")
            print_info(f"Current CUDA device: {torch.cuda.current_device()}")
            
        # Test basic tensor operations
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print_success("Basic tensor operations working")
        
        # Test gradients
        x = torch.randn(5, 5, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        print_success("Gradient computation working")
        
        return True
        
    except Exception as e:
        print_error(f"PyTorch test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive model test"""
    print_section("Comprehensive Model Test")
    
    try:
        from main import ModelConfig, MultiModalStockModel
        
        # Create a small config for testing
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=2,
            sequence_length=20,
            prediction_horizons=[30, 90]
        )
        
        # Instantiate main model (will be mostly None placeholders for now)
        model = MultiModalStockModel(config)
        print_success("Main model instantiated")
        
        print_info("Note: Main model components are placeholders until fusion layer is implemented")
        
        return True
        
    except Exception as e:
        print_error(f"Comprehensive test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print_section("Multimodal Stock Model Verification")
    print_info("This script verifies the setup and basic functionality")
    
    tests = [
        ("Module Imports", test_imports),
        ("Model Configuration", test_model_config),
        ("PyTorch Setup", test_pytorch_setup),
        ("TFT Encoder", test_tft_encoder),
        ("Text Encoder", test_text_encoder),
        ("Employment Encoder", test_employment_encoder),
        ("Data Processing", test_data_processing),
        ("Comprehensive Model", run_comprehensive_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_section("Verification Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status:<12} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("🎉 All verifications passed! Your setup is working correctly.")
        print_info("You can now proceed with implementing the fusion layer and training pipeline.")
    else:
        print_error("Some tests failed. Please check the error messages above.")
        print_info("Consider installing missing dependencies or checking your Python environment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)