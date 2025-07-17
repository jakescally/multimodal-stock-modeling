#!/usr/bin/env python3
"""
Data Pipeline Verification Script
================================

Verifies that the data preprocessing pipeline is working correctly:
1. Stock data loading and technical indicators
2. News data collection and processing  
3. Employment data generation and analysis
4. Unified dataset creation
5. Data alignment and normalization
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import traceback
from datetime import datetime, timedelta

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

def test_stock_data_loading():
    """Test stock data loading and technical indicators"""
    print_section("Testing Stock Data Loading")
    
    try:
        from data.stock_data_loader import StockDataLoader, TechnicalIndicators
        
        # Test basic stock data loading
        loader = StockDataLoader(cache_dir="data/test_cache")
        print_success("Stock data loader instantiated")
        
        # Test getting a single stock (using a mock/small dataset approach)
        # We'll create synthetic data instead of downloading to avoid network dependencies
        
        # Create synthetic stock data
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        synthetic_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 300),
            'High': np.random.uniform(150, 250, 300),
            'Low': np.random.uniform(80, 150, 300), 
            'Close': np.random.uniform(90, 220, 300),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
        
        print_success(f"Created synthetic stock data: {synthetic_data.shape}")
        
        # Test technical indicators
        indicators = TechnicalIndicators()
        
        # Test moving averages
        data_with_ma = indicators.add_moving_averages(synthetic_data)
        print_success(f"Added moving averages: {len([c for c in data_with_ma.columns if 'SMA' in c])} indicators")
        
        # Test RSI
        data_with_rsi = indicators.add_rsi(data_with_ma)
        print_success("Added RSI indicator")
        
        # Test MACD
        data_with_macd = indicators.add_macd(data_with_rsi)
        print_success("Added MACD indicator")
        
        # Test Bollinger Bands
        data_with_bb = indicators.add_bollinger_bands(data_with_macd)
        print_success("Added Bollinger Bands")
        
        # Test all indicators
        data_complete = indicators.add_all_indicators(synthetic_data)
        print_success(f"Added all indicators: {data_complete.shape[1]} total features")
        print_info(f"Feature columns: {list(data_complete.columns)[:10]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Stock data loading test failed: {e}")
        traceback.print_exc()
        return False

def test_news_data_processing():
    """Test news data processing"""
    print_section("Testing News Data Processing")
    
    try:
        from data.news_data_loader import NewsProcessor, NewsAggregator
        
        # Create mock news articles
        mock_articles = [
            {
                'title': 'Apple reports strong quarterly earnings with revenue growth',
                'summary': 'Apple Inc. exceeded analyst expectations with 15% revenue growth driven by iPhone sales',
                'source': 'reuters',
                'published_date': datetime.now() - timedelta(days=1),
                'link': 'http://example.com/news1'
            },
            {
                'title': 'Tech stocks decline amid market volatility concerns',
                'summary': 'Major technology companies saw share prices fall as investors worry about economic outlook',
                'source': 'marketwatch', 
                'published_date': datetime.now() - timedelta(days=2),
                'link': 'http://example.com/news2'
            },
            {
                'title': 'Federal Reserve maintains interest rates, signals cautious approach',
                'summary': 'The central bank kept rates unchanged while monitoring inflation and employment data',
                'source': 'yahoo_finance',
                'published_date': datetime.now() - timedelta(days=3),
                'link': 'http://example.com/news3'
            }
        ]
        
        print_success(f"Created {len(mock_articles)} mock news articles")
        
        # Test news processing
        processor = NewsProcessor()
        processed_df = processor.process_news_batch(mock_articles)
        
        print_success(f"Processed news articles: {processed_df.shape}")
        print_info(f"Processed columns: {list(processed_df.columns)}")
        
        # Test sentiment scores
        avg_sentiment = processed_df['sentiment_positive'].mean()
        print_info(f"Average positive sentiment: {avg_sentiment:.3f}")
        
        # Test news aggregation
        aggregator = NewsAggregator()
        daily_news = aggregator.aggregate_daily_news(processed_df)
        
        print_success(f"Aggregated news by day: {daily_news.shape}")
        print_info(f"Daily aggregation columns: {list(daily_news.columns)}")
        
        return True
        
    except Exception as e:
        print_error(f"News data processing test failed: {e}")
        traceback.print_exc()
        return False

def test_employment_data_processing():
    """Test employment data processing"""
    print_section("Testing Employment Data Processing")
    
    try:
        from data.employment_data_loader import JobPostingsScraper, EmploymentProcessor
        
        companies = ['AAPL', 'MSFT', 'GOOGL']
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now()
        
        # Test job postings scraper
        scraper = JobPostingsScraper()
        job_df = scraper.generate_mock_job_data(companies, start_date, end_date)
        
        print_success(f"Generated mock job data: {job_df.shape}")
        print_info(f"Job posting columns: {list(job_df.columns)}")
        
        # Test skill analysis
        skill_df = scraper.analyze_skill_demand(job_df)
        
        print_success(f"Analyzed skill demand: {skill_df.shape}")
        print_info(f"Skill analysis columns: {list(skill_df.columns)}")
        
        # Test employment processor
        processor = EmploymentProcessor()
        
        # Test hiring velocity calculation
        velocity_df = processor.calculate_hiring_velocity(job_df)
        
        print_success(f"Calculated hiring velocity: {velocity_df.shape}")
        
        # Create mock layoff data for testing
        layoff_data = []
        for company in companies:
            if np.random.random() < 0.3:  # 30% chance of layoffs
                layoff_data.append({
                    'company_id': company,
                    'layoff_date': start_date + timedelta(days=np.random.randint(0, 90)),
                    'employees_affected': np.random.randint(10, 100),
                    'reason': 'restructuring'
                })
        
        layoff_df = pd.DataFrame(layoff_data)
        print_success(f"Created mock layoff data: {layoff_df.shape}")
        
        # Test employment signals calculation
        if not layoff_df.empty:
            signals_df = processor.calculate_employment_signals(job_df, layoff_df, skill_df)
            print_success(f"Calculated employment signals: {signals_df.shape}")
            print_info(f"Signal columns: {list(signals_df.columns)}")
            
            # Check signal values
            avg_employment_score = signals_df['employment_score'].mean()
            print_info(f"Average employment score: {avg_employment_score:.3f}")
        
        return True
        
    except Exception as e:
        print_error(f"Employment data processing test failed: {e}")
        traceback.print_exc()
        return False

def test_unified_dataset():
    """Test unified dataset creation"""
    print_section("Testing Unified Dataset Creation")
    
    try:
        from data.unified_dataset import UnifiedDatasetBuilder, MultiModalDataset
        
        # Test with small dataset
        symbols = ['AAPL', 'MSFT']
        
        builder = UnifiedDatasetBuilder(
            cache_dir="data/test_unified_cache",
            sequence_length=20  # Smaller for testing
        )
        
        print_success("Unified dataset builder instantiated")
        
        # Since we're testing without actual data downloads, we'll test the components
        # Test MultiModalDataset directly
        
        # Create mock features and targets
        n_samples = 50
        sequence_length = 20
        
        mock_features = {
            'stock': torch.randn(n_samples, sequence_length, 15),      # 15 stock features
            'news': torch.randn(n_samples, sequence_length, 10),       # 10 news features  
            'employment': torch.randn(n_samples, sequence_length, 8)   # 8 employment features
        }
        
        mock_targets = {
            'horizon_30': torch.randn(n_samples),
            'horizon_180': torch.randn(n_samples),
            'horizon_365': torch.randn(n_samples),
            'horizon_730': torch.randn(n_samples)
        }
        
        # Test dataset creation
        dataset = MultiModalDataset(mock_features, mock_targets, sequence_length)
        
        print_success(f"Created multimodal dataset with {len(dataset)} samples")
        
        # Test data loading
        sample_features, sample_targets = dataset[0]
        
        print_success("Dataset indexing works correctly")
        print_info(f"Feature shapes: {[(k, v.shape) for k, v in sample_features.items()]}")
        print_info(f"Target shapes: {[(k, v.shape) for k, v in sample_targets.items()]}")
        
        # Test data loader creation
        data_loaders = builder.create_data_loaders({
            'train': {'features': mock_features, 'targets': mock_targets},
            'val': {'features': mock_features, 'targets': mock_targets},
            'test': {'features': mock_features, 'targets': mock_targets}
        }, batch_size=8)
        
        print_success("Created data loaders for train/val/test")
        
        # Test batch loading
        train_loader = data_loaders['train']
        batch_features, batch_targets = next(iter(train_loader))
        
        print_success("Batch loading works correctly")
        print_info(f"Batch feature shapes: {[(k, v.shape) for k, v in batch_features.items()]}")
        print_info(f"Batch target shapes: {[(k, v.shape) for k, v in batch_targets.items()]}")
        
        return True
        
    except Exception as e:
        print_error(f"Unified dataset test failed: {e}")
        traceback.print_exc()
        return False

def test_data_normalization():
    """Test data normalization and preprocessing"""
    print_section("Testing Data Normalization")
    
    try:
        from data.stock_data_loader import MarketDataProcessor
        
        processor = MarketDataProcessor(sequence_length=50)
        
        # Create mock time series data
        n_samples = 200
        n_features = 15
        sequence_length = 50
        
        # Create realistic-looking financial data
        mock_data = np.random.randn(n_samples, sequence_length, n_features)
        
        # Add some realistic scaling (stock prices, volumes, etc.)
        mock_data[:, :, 0] *= 100  # Stock prices around 100
        mock_data[:, :, 1] *= 1000000  # Volume in millions
        
        print_success(f"Created mock time series data: {mock_data.shape}")
        
        # Test normalization
        normalized_data, scaler_params = processor.normalize_features(
            mock_data, fit_on_train=True
        )
        
        print_success("Data normalization completed")
        print_info(f"Normalized data shape: {normalized_data.shape}")
        print_info(f"Mean after normalization: {np.mean(normalized_data):.6f}")
        print_info(f"Std after normalization: {np.std(normalized_data):.6f}")
        
        # Test that mean is close to 0 and std is close to 1
        assert abs(np.mean(normalized_data)) < 0.01, "Mean should be close to 0"
        assert abs(np.std(normalized_data) - 1.0) < 0.01, "Std should be close to 1"
        
        print_success("Normalization statistics are correct")
        
        # Test applying normalization to new data
        new_data = np.random.randn(20, sequence_length, n_features)
        new_normalized, _ = processor.normalize_features(
            new_data, fit_on_train=False, scaler_params=scaler_params
        )
        
        print_success("Applied existing normalization to new data")
        
        return True
        
    except Exception as e:
        print_error(f"Data normalization test failed: {e}")
        traceback.print_exc()
        return False

def test_feature_engineering():
    """Test feature engineering capabilities"""
    print_section("Testing Feature Engineering")
    
    try:
        from data.stock_data_loader import TechnicalIndicators
        
        # Create realistic stock price data
        n_days = 500
        dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
        
        # Generate price with realistic patterns
        price_trend = np.linspace(100, 150, n_days)  # Upward trend
        price_noise = np.random.normal(0, 5, n_days)  # Daily volatility
        price_seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n_days))  # Seasonal pattern
        
        close_prices = price_trend + price_noise + price_seasonal
        
        # Generate OHLV data based on close prices
        stock_data = pd.DataFrame({
            'Open': close_prices + np.random.normal(0, 1, n_days),
            'High': close_prices + np.abs(np.random.normal(2, 1, n_days)),
            'Low': close_prices - np.abs(np.random.normal(2, 1, n_days)),
            'Close': close_prices,
            'Volume': np.random.randint(500000, 5000000, n_days)
        }, index=dates)
        
        print_success(f"Generated realistic stock data: {stock_data.shape}")
        
        # Test comprehensive feature engineering
        enriched_data = TechnicalIndicators.add_all_indicators(stock_data)
        
        print_success(f"Added technical indicators: {enriched_data.shape[1]} total features")
        
        # Verify key indicators exist and have reasonable values
        assert 'SMA_20' in enriched_data.columns, "SMA_20 should exist"
        assert 'RSI' in enriched_data.columns, "RSI should exist"
        assert 'MACD' in enriched_data.columns, "MACD should exist"
        
        # Check RSI is in expected range (0-100)
        rsi_values = enriched_data['RSI'].dropna()
        assert rsi_values.min() >= 0 and rsi_values.max() <= 100, "RSI should be between 0-100"
        
        print_success("Technical indicators have valid ranges")
        
        # Test feature completeness
        non_null_features = enriched_data.dropna()
        print_info(f"Complete feature rows: {len(non_null_features)} / {len(enriched_data)}")
        
        feature_categories = {
            'Price': ['Open', 'High', 'Low', 'Close'],
            'Volume': ['Volume', 'OBV', 'VPT'],
            'Trend': [col for col in enriched_data.columns if 'SMA_' in col or 'EMA_' in col],
            'Momentum': ['RSI', 'MACD', 'MACD_Signal'],
            'Volatility': [col for col in enriched_data.columns if 'Volatility_' in col or 'BB_' in col]
        }
        
        for category, features in feature_categories.items():
            existing_features = [f for f in features if f in enriched_data.columns]
            print_info(f"{category} features: {len(existing_features)} ({existing_features[:3]}...)")
        
        return True
        
    except Exception as e:
        print_error(f"Feature engineering test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all data pipeline verification tests"""
    print_section("Data Pipeline Verification")
    print_info("Testing the data preprocessing pipeline components")
    
    tests = [
        ("Stock Data Loading", test_stock_data_loading),
        ("News Data Processing", test_news_data_processing), 
        ("Employment Data Processing", test_employment_data_processing),
        ("Unified Dataset Creation", test_unified_dataset),
        ("Data Normalization", test_data_normalization),
        ("Feature Engineering", test_feature_engineering)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_section("Data Pipeline Verification Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status:<12} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("🎉 All data pipeline tests passed! Your preprocessing pipeline is working correctly.")
        print_info("The data pipeline can handle:")
        print_info("  • Stock data loading and technical indicators")
        print_info("  • News data processing and sentiment analysis")
        print_info("  • Employment data generation and signal calculation")
        print_info("  • Multimodal dataset creation and normalization")
        print_info("  • PyTorch DataLoader integration")
    else:
        print_error("Some data pipeline tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)