#!/usr/bin/env python3
"""
Real Dataset Builder
===================

Build training datasets from real financial data.
Integrates with RealDataFetcher to create production-ready training data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from real_data_fetcher import RealDataFetcher, DataConfig

# For now, we'll create a simple text encoder placeholder
class SimpleTextEncoder:
    def __init__(self, model_name='distilbert-base-uncased', max_length=128, device='auto'):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
    
    def encode(self, texts):
        # Simple placeholder - in production this would use BERT
        return np.zeros((len(texts), 10))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealMultiModalDataset(Dataset):
    """Dataset for real multimodal financial data"""
    
    def __init__(self, features: Dict[str, np.ndarray], targets: Dict[str, np.ndarray], 
                 sequence_length: int = 252):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Determine dataset size from any feature
        self.size = len(next(iter(features.values())))
        
        # Convert to tensors
        self.features_tensor = {
            key: torch.FloatTensor(value) for key, value in features.items()
        }
        self.targets_tensor = {
            key: torch.FloatTensor(value) for key, value in targets.items()
        }
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        features = {key: tensor[idx] for key, tensor in self.features_tensor.items()}
        targets = {key: tensor[idx] for key, tensor in self.targets_tensor.items()}
        return features, targets


class RealDatasetBuilder:
    """Build training datasets from real financial data"""
    
    def __init__(self, config: DataConfig, sequence_length: int = 252):
        self.config = config
        self.sequence_length = sequence_length
        self.data_fetcher = RealDataFetcher(config)
        
        # Initialize text encoder for news processing
        self.text_encoder = SimpleTextEncoder(
            model_name='distilbert-base-uncased',
            max_length=128,
            device='auto'
        )
        
    def build_real_dataset(self) -> Dict[str, Dict]:
        """Build complete dataset from real data sources"""
        logger.info("🏗️  Building real dataset from multiple sources")
        
        # Fetch all real data
        stock_data, news_data, targets, employment_data = self.data_fetcher.fetch_all_data()
        
        # Process and align data
        processed_data = self._process_and_align_data(stock_data, news_data, employment_data, targets)
        
        # Create train/validation/test splits
        dataset_splits = self._create_time_based_splits(processed_data)
        
        # Convert to tensors and create final datasets
        final_datasets = self._create_tensor_datasets(dataset_splits)
        
        logger.info("✅ Real dataset built successfully")
        return final_datasets
        
    def _process_and_align_data(self, stock_data: Dict[str, pd.DataFrame], 
                               news_data: Dict[str, pd.DataFrame],
                               employment_data: pd.DataFrame,
                               targets: Dict[str, Dict[str, pd.Series]]) -> Dict:
        """Process and temporally align all data sources"""
        logger.info("🔄 Processing and aligning multimodal data")
        
        # Get common date range across all stocks
        all_dates = set()
        for symbol, df in stock_data.items():
            all_dates.update(df.index.date)
        
        # Sort dates
        sorted_dates = sorted(all_dates)
        
        # Process each symbol separately
        processed_samples = []
        
        for symbol in stock_data.keys():
            stock_df = stock_data[symbol]
            symbol_targets = targets[symbol]
            
            logger.info(f"  📈 Processing {symbol} ({len(stock_df)} trading days)")
            
            # Create sequences
            for i in range(self.sequence_length, len(stock_df)):
                current_date = stock_df.index[i].date()
                
                # Stock sequence (technical indicators)
                stock_sequence = self._extract_stock_features(stock_df, i)
                
                # News sequence
                news_sequence = self._extract_news_features(news_data, current_date)
                
                # Employment sequence
                employment_sequence = self._extract_employment_features(employment_data, current_date)
                
                # Targets
                sample_targets = {}
                for horizon_key, target_series in symbol_targets.items():
                    if i - self.sequence_length < len(target_series):
                        sample_targets[horizon_key] = target_series.iloc[i - self.sequence_length]
                
                # Only add if we have targets
                if sample_targets:
                    processed_samples.append({
                        'symbol': symbol,
                        'date': current_date,
                        'stock_features': stock_sequence,
                        'news_features': news_sequence,
                        'employment_features': employment_sequence,
                        'targets': sample_targets
                    })
        
        logger.info(f"  ✅ Created {len(processed_samples)} training samples")
        return processed_samples
        
    def _extract_stock_features(self, stock_df: pd.DataFrame, end_idx: int) -> np.ndarray:
        """Extract stock features for a sequence"""
        start_idx = max(0, end_idx - self.sequence_length)
        sequence_df = stock_df.iloc[start_idx:end_idx]
        
        # Select relevant features (excluding basic OHLC, focus on technical indicators)
        feature_columns = [
            'SMA_5', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'MACD_histogram', 'RSI',
            'BB_width', 'BB_position', 'Volume_ratio',
            'Returns_1d', 'Returns_5d', 'Returns_20d'
        ]
        
        # Fill any missing values
        features = sequence_df[feature_columns].fillna(method='ffill').fillna(0)
        
        # Pad if sequence is shorter than required
        if len(features) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(features), len(feature_columns)))
            features = np.vstack([padding, features.values])
        else:
            features = features.values
            
        return features.astype(np.float32)
        
    def _extract_news_features(self, news_data: Dict[str, pd.DataFrame], 
                              current_date: datetime.date) -> np.ndarray:
        """Extract news features for a given date"""
        # Look for news in the past 7 days
        start_date = current_date - timedelta(days=7)
        
        all_news_texts = []
        
        for source, news_df in news_data.items():
            if news_df.empty:
                continue
                
            # Filter news by date
            news_df['date'] = pd.to_datetime(news_df['date']).dt.date
            recent_news = news_df[
                (news_df['date'] >= start_date) & 
                (news_df['date'] <= current_date)
            ]
            
            # Collect titles and summaries
            for _, row in recent_news.iterrows():
                if pd.notna(row['title']):
                    all_news_texts.append(row['title'])
                if pd.notna(row['summary']):
                    all_news_texts.append(row['summary'])
        
        # If no news found, create neutral embedding
        if not all_news_texts:
            return np.zeros(10, dtype=np.float32)  # 10-dimensional news features
        
        # Encode news text (simplified for now)
        # In production, you'd use the actual text encoder
        news_embedding = self._simple_news_embedding(all_news_texts)
        
        return news_embedding
        
    def _simple_news_embedding(self, texts: List[str]) -> np.ndarray:
        """Simple news embedding (placeholder for actual BERT encoding)"""
        # For now, create a simple sentiment-like embedding
        combined_text = ' '.join(texts).lower()
        
        # Simple sentiment keywords
        positive_words = ['growth', 'profit', 'gain', 'up', 'rise', 'increase', 'strong', 'beat', 'exceed']
        negative_words = ['loss', 'down', 'fall', 'decline', 'weak', 'miss', 'cut', 'reduce', 'concern']
        
        # Count occurrences
        positive_count = sum(word in combined_text for word in positive_words)
        negative_count = sum(word in combined_text for word in negative_words)
        
        # Create feature vector
        features = np.array([
            positive_count,
            negative_count,
            positive_count - negative_count,  # Net sentiment
            len(texts),  # Number of articles
            len(combined_text.split()),  # Total words
            combined_text.count('stock'),
            combined_text.count('earnings'),
            combined_text.count('revenue'),
            combined_text.count('market'),
            min(1.0, positive_count / max(1, positive_count + negative_count))  # Positive ratio
        ], dtype=np.float32)
        
        return features
        
    def _extract_employment_features(self, employment_data: pd.DataFrame, 
                                   current_date: datetime.date) -> np.ndarray:
        """Extract employment features for a given date"""
        # Find the most recent employment data (monthly data)
        employment_data['date'] = pd.to_datetime(employment_data['date']).dt.date
        
        # Get data up to current date
        valid_data = employment_data[employment_data['date'] <= current_date]
        
        if valid_data.empty:
            # Return zero features if no data
            return np.zeros(8, dtype=np.float32)
        
        # Get most recent record
        latest_record = valid_data.iloc[-1]
        
        # Extract features
        features = np.array([
            latest_record['unemployment_rate'],
            latest_record['job_openings'],
            latest_record['labor_participation'],
            latest_record['jobless_claims'],
            latest_record['tech_employment_index'],
            latest_record['hiring_rate'],
            latest_record['layoff_rate'],
            latest_record['consumer_sentiment']
        ], dtype=np.float32)
        
        return features
        
    def _create_time_based_splits(self, processed_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Create time-based train/validation/test splits"""
        logger.info("📊 Creating time-based data splits")
        
        # Sort by date
        processed_data.sort(key=lambda x: x['date'])
        
        # Calculate split indices
        total_samples = len(processed_data)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        splits = {
            'train': processed_data[:train_end],
            'val': processed_data[train_end:val_end],
            'test': processed_data[val_end:]
        }
        
        logger.info(f"  📈 Train: {len(splits['train'])} samples")
        logger.info(f"  📊 Validation: {len(splits['val'])} samples")
        logger.info(f"  📋 Test: {len(splits['test'])} samples")
        
        return splits
        
    def _create_tensor_datasets(self, dataset_splits: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Convert processed data to tensor datasets"""
        logger.info("🔢 Converting to tensor datasets")
        
        final_datasets = {}
        
        for split_name, split_data in dataset_splits.items():
            if not split_data:
                continue
                
            # Collect all features and targets
            all_stock_features = []
            all_news_features = []
            all_employment_features = []
            all_targets = {key: [] for key in ['horizon_30', 'horizon_180', 'horizon_365', 'horizon_730']}
            
            for sample in split_data:
                all_stock_features.append(sample['stock_features'])
                all_news_features.append(sample['news_features'])
                all_employment_features.append(sample['employment_features'])
                
                for horizon_key in all_targets.keys():
                    if horizon_key in sample['targets']:
                        all_targets[horizon_key].append(sample['targets'][horizon_key])
                    else:
                        all_targets[horizon_key].append(0.0)  # Default value
            
            # Convert to numpy arrays
            features = {
                'stock': np.array(all_stock_features),
                'news': np.array(all_news_features),
                'employment': np.array(all_employment_features)
            }
            
            targets = {key: np.array(values) for key, values in all_targets.items()}
            
            # Create dataset
            dataset = RealMultiModalDataset(features, targets, self.sequence_length)
            
            final_datasets[split_name] = {
                'dataset': dataset,
                'features': features,
                'targets': targets,
                'size': len(dataset)
            }
            
            logger.info(f"  ✅ {split_name}: {len(dataset)} samples")
        
        return final_datasets
        
    def create_data_loaders(self, dataset_splits: Dict[str, Dict], 
                           batch_size: int = 32, num_workers: int = 4,
                           pin_memory: bool = False, persistent_workers: bool = False) -> Dict[str, DataLoader]:
        """Create DataLoaders for training"""
        logger.info("🔄 Creating data loaders")
        
        data_loaders = {}
        
        for split_name, split_info in dataset_splits.items():
            dataset = split_info['dataset']
            shuffle = (split_name == 'train')
            
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers > 0
            )
            
            data_loaders[split_name] = data_loader
            logger.info(f"  ✅ {split_name}: {len(dataset)} samples, {len(data_loader)} batches")
        
        return data_loaders


def main():
    """Test the real dataset builder"""
    config = DataConfig(
        start_date="2023-01-01",  # Shorter period for testing
        symbols=['AAPL', 'MSFT']
    )
    
    builder = RealDatasetBuilder(config, sequence_length=60)  # Shorter sequence for testing
    
    # Build dataset
    dataset_splits = builder.build_real_dataset()
    
    # Create data loaders
    data_loaders = builder.create_data_loaders(dataset_splits, batch_size=16)
    
    # Test data loading
    train_loader = data_loaders['train']
    features, targets = next(iter(train_loader))
    
    print("\n📊 Dataset Test Results:")
    print(f"Batch size: {features['stock'].shape[0]}")
    print(f"Stock features shape: {features['stock'].shape}")
    print(f"News features shape: {features['news'].shape}")
    print(f"Employment features shape: {features['employment'].shape}")
    print(f"Target horizons: {list(targets.keys())}")
    print(f"Target sample shape: {targets['horizon_30'].shape}")


if __name__ == "__main__":
    main()