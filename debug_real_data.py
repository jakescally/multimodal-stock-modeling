#!/usr/bin/env python3
"""
Debug Real Data Shapes
======================

Debug the shapes of real data to understand dimension mismatches.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data.real_data_fetcher import DataConfig
from data.real_dataset_builder import RealDatasetBuilder

def debug_data_shapes():
    """Debug the shapes of real data"""
    print("🔍 Debugging Real Data Shapes")
    print("=" * 50)
    
    # Create small test dataset
    config = DataConfig(
        start_date="2024-01-01",
        symbols=['AAPL', 'MSFT']
    )
    
    builder = RealDatasetBuilder(config, sequence_length=60)
    dataset_splits = builder.build_real_dataset()
    
    # Create data loaders
    data_loaders = builder.create_data_loaders(dataset_splits, batch_size=16)
    
    # Get a batch
    train_loader = data_loaders['train']
    features, targets = next(iter(train_loader))
    
    print("📊 Feature Shapes:")
    for key, tensor in features.items():
        print(f"  {key}: {tensor.shape}")
    
    print("\n🎯 Target Shapes:")
    for key, tensor in targets.items():
        print(f"  {key}: {tensor.shape}")
    
    print("\n🔍 Expected vs Actual:")
    print("Expected by model:")
    print("  stock: [batch, seq_len, stock_features] = [16, 60, 15]")
    print("  news: [batch, seq_len, news_features] = [16, 60, 10]")
    print("  employment: [batch, seq_len, employment_features] = [16, 60, 8]")
    
    print("\nActual from real data:")
    print(f"  stock: {features['stock'].shape}")
    print(f"  news: {features['news'].shape}")
    print(f"  employment: {features['employment'].shape}")
    
    # Check if we need to expand news and employment
    if len(features['news'].shape) == 2:
        print("\n⚠️  NEWS DATA ISSUE: Missing sequence dimension")
        print("   News data needs to be expanded to [batch, seq_len, features]")
    
    if len(features['employment'].shape) == 2:
        print("\n⚠️  EMPLOYMENT DATA ISSUE: Missing sequence dimension")
        print("   Employment data needs to be expanded to [batch, seq_len, features]")

if __name__ == "__main__":
    debug_data_shapes()