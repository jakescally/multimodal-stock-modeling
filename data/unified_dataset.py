"""
Unified Dataset Builder
======================

Combines stock, news, and employment data into a unified dataset
for multimodal training.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle

from .stock_data_loader import StockDataLoader, TechnicalIndicators, MarketDataProcessor
from .news_data_loader import NewsDataLoader, NewsProcessor, NewsAggregator
from .employment_data_loader import EmploymentDataLoader, JobPostingsScraper, EmploymentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """PyTorch Dataset for multimodal stock prediction"""
    
    def __init__(self, features: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor],
                 sequence_length: int = 252):
        """
        Args:
            features: Dictionary containing different modality features
            targets: Dictionary containing target values for different horizons
            sequence_length: Length of input sequences
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Validate that all features have the same number of samples
        sample_counts = [len(v) for v in features.values()]
        assert len(set(sample_counts)) == 1, "All features must have same number of samples"
        
        self.n_samples = sample_counts[0]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        batch_features = {}
        batch_targets = {}
        
        for key, data in self.features.items():
            batch_features[key] = data[idx]
            
        for key, data in self.targets.items():
            batch_targets[key] = data[idx]
            
        return batch_features, batch_targets


class UnifiedDatasetBuilder:
    """Main class for building unified multimodal datasets"""
    
    def __init__(self, cache_dir: str = "data/unified_cache", 
                 sequence_length: int = 252):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = sequence_length
        
        # Initialize data loaders
        self.stock_loader = StockDataLoader()
        self.news_loader = NewsDataLoader()
        self.employment_loader = EmploymentDataLoader()
        
        # Initialize processors
        self.market_processor = MarketDataProcessor(sequence_length)
        self.news_processor = NewsProcessor()
        self.news_aggregator = NewsAggregator()
        self.employment_processor = EmploymentProcessor()
        
    def build_complete_dataset(self, symbols: List[str], 
                             start_date: str = "2020-01-01",
                             end_date: Optional[str] = None,
                             include_news: bool = True,
                             include_employment: bool = True) -> Dict:
        """
        Build a complete multimodal dataset
        
        Args:
            symbols: List of stock symbols to include
            start_date: Start date for data collection
            end_date: End date for data collection (default: today)
            include_news: Whether to include news data
            include_employment: Whether to include employment data
            
        Returns:
            Dictionary containing train/val/test splits and metadata
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"Building unified dataset for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Check cache
        cache_key = f"unified_{'-'.join(symbols)}_{start_date}_{end_date}_{include_news}_{include_employment}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=6):  # Cache for 6 hours
                logger.info("Loading dataset from cache")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # Step 1: Load stock data
        logger.info("Loading stock data...")
        stock_data = self._load_stock_data(symbols, start_date, end_date)
        
        # Step 2: Load and align news data
        news_data = {}
        if include_news:
            logger.info("Loading news data...")
            news_data = self._load_news_data(symbols, start_date, end_date)
        
        # Step 3: Load and align employment data
        employment_data = {}
        if include_employment:
            logger.info("Loading employment data...")
            employment_data = self._load_employment_data(symbols, start_date, end_date)
        
        # Step 4: Align all data sources
        logger.info("Aligning data sources...")
        aligned_data = self._align_all_data(stock_data, news_data, employment_data)
        
        # Step 5: Create sequences and targets
        logger.info("Creating sequences...")
        dataset = self._create_sequences(aligned_data, symbols)
        
        # Step 6: Split data
        logger.info("Splitting data...")
        splits = self._split_data(dataset)
        
        # Step 7: Normalize features
        logger.info("Normalizing features...")
        normalized_splits = self._normalize_features(splits)
        
        # Cache the result
        with open(cache_file, 'wb') as f:
            pickle.dump(normalized_splits, f)
        
        logger.info("Dataset building complete!")
        return normalized_splits
    
    def _load_stock_data(self, symbols: List[str], start_date: str, 
                        end_date: str) -> Dict[str, pd.DataFrame]:
        """Load and process stock data for all symbols"""
        stock_data = {}
        
        for symbol in symbols:
            logger.info(f"Processing stock data for {symbol}")
            
            # Get stock data
            df = self.stock_loader.get_stock_data(symbol, period="max")
            
            if df.empty:
                logger.warning(f"No stock data found for {symbol}")
                continue
            
            # Filter date range
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            df = df.loc[mask]
            
            if len(df) < self.sequence_length + 730:  # Need enough data
                logger.warning(f"Insufficient stock data for {symbol}")
                continue
            
            # Add technical indicators
            df_processed = TechnicalIndicators.add_all_indicators(df)
            df_processed = df_processed.dropna()
            
            stock_data[symbol] = df_processed
            
        return stock_data
    
    def _load_news_data(self, symbols: List[str], start_date: str, 
                       end_date: str) -> Dict[str, pd.DataFrame]:
        """Load and process news data for all symbols"""
        news_data = {}
        
        # Get general financial news
        all_news = self.news_loader.get_all_rss_news()
        
        if not all_news:
            logger.warning("No news data available")
            return news_data
        
        # Process news articles
        news_df = self.news_processor.process_news_batch(all_news)
        
        # Filter date range
        news_df['published_date'] = pd.to_datetime(news_df['published_date'])
        mask = (news_df['published_date'] >= start_date) & (news_df['published_date'] <= end_date)
        news_df = news_df.loc[mask]
        
        # Aggregate by day
        daily_news = self.news_aggregator.aggregate_daily_news(news_df)
        
        # For each symbol, create company-specific news features
        for symbol in symbols:
            # In a real implementation, this would filter news specific to each company
            # For now, we'll use general market news for all symbols
            symbol_news = daily_news.copy()
            symbol_news['symbol'] = symbol
            news_data[symbol] = symbol_news
            
        return news_data
    
    def _load_employment_data(self, symbols: List[str], start_date: str,
                            end_date: str) -> Dict[str, pd.DataFrame]:
        """Load and process employment data for all symbols"""
        employment_data = {}
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate mock employment data (in production, use real data sources)
        job_scraper = JobPostingsScraper()
        
        # Generate job postings data
        job_df = job_scraper.generate_mock_job_data(symbols, start_dt, end_dt)
        
        # Analyze skill demand
        skill_df = job_scraper.analyze_skill_demand(job_df)
        
        # Generate layoff data
        layoff_loader = self.employment_loader
        layoff_df = layoff_loader.generate_mock_layoff_data(symbols, start_dt, end_dt)
        
        # Calculate employment signals
        employment_signals = self.employment_processor.calculate_employment_signals(
            job_df, layoff_df, skill_df
        )
        
        # Split by symbol
        for symbol in symbols:
            symbol_employment = employment_signals[
                employment_signals['company_id'] == symbol
            ].copy()
            employment_data[symbol] = symbol_employment
            
        return employment_data
    
    def _align_all_data(self, stock_data: Dict[str, pd.DataFrame],
                       news_data: Dict[str, pd.DataFrame],
                       employment_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all data sources by date and symbol"""
        aligned_data = {}
        
        for symbol in stock_data.keys():
            logger.info(f"Aligning data for {symbol}")
            
            # Start with stock data
            df = stock_data[symbol].copy()
            df['symbol'] = symbol
            df['date'] = df['Date'].dt.date
            
            # Merge news data
            if symbol in news_data:
                news_df = news_data[symbol]
                news_df['date'] = pd.to_datetime(news_df['date']).dt.date
                
                df = df.merge(news_df, on='date', how='left')
                
                # Fill missing news data
                news_columns = [col for col in news_df.columns 
                              if col not in ['date', 'symbol']]
                df[news_columns] = df[news_columns].fillna(0)
            
            # Merge employment data
            if symbol in employment_data:
                emp_df = employment_data[symbol]
                emp_df['date'] = pd.to_datetime(emp_df['date']).dt.date
                
                df = df.merge(emp_df, on='date', how='left')
                
                # Fill missing employment data
                emp_columns = [col for col in emp_df.columns 
                             if col not in ['company_id', 'date']]
                df[emp_columns] = df[emp_columns].fillna(0)
            
            # Remove rows with insufficient data
            df = df.dropna(subset=['Close', 'Volume'])
            
            aligned_data[symbol] = df
            
        return aligned_data
    
    def _create_sequences(self, aligned_data: Dict[str, pd.DataFrame],
                         symbols: List[str]) -> Dict:
        """Create sequences for multimodal training"""
        
        # Define feature groups
        stock_features = ['Open', 'High', 'Low', 'Close', 'Volume'] + \
                        [col for col in aligned_data[symbols[0]].columns 
                         if any(x in col for x in ['SMA_', 'EMA_', 'RSI', 'MACD', 'BB_', 'Volatility_', 'OBV'])]
        
        news_features = [col for col in aligned_data[symbols[0]].columns 
                        if any(x in col for x in ['article_', 'sentiment_', 'keyword_', 'news_'])]
        
        employment_features = [col for col in aligned_data[symbols[0]].columns 
                             if any(x in col for x in ['jobs_', 'layoffs_', 'hiring_', 'demand', 'employment_'])]
        
        all_features = []
        all_targets = {f'horizon_{h}': [] for h in [30, 180, 365, 730]}
        
        for symbol in symbols:
            df = aligned_data[symbol]
            
            if len(df) < self.sequence_length + 730:
                continue
            
            # Separate feature types
            stock_data = df[stock_features].values
            news_data = df[news_features].values if news_features else np.zeros((len(df), 10))
            employment_data = df[employment_features].values if employment_features else np.zeros((len(df), 10))
            
            # Create targets (future returns)
            close_prices = df['Close'].values
            targets = {}
            for horizon in [30, 180, 365, 730]:
                future_returns = []
                for i in range(len(close_prices) - horizon):
                    current_price = close_prices[i]
                    future_price = close_prices[i + horizon]
                    ret = (future_price / current_price) - 1
                    future_returns.append(ret)
                targets[f'horizon_{horizon}'] = np.array(future_returns)
            
            # Create sequences
            for i in range(self.sequence_length, len(df) - 730):  # Leave room for longest horizon
                # Feature sequences
                stock_seq = stock_data[i-self.sequence_length:i]
                news_seq = news_data[i-self.sequence_length:i]
                employment_seq = employment_data[i-self.sequence_length:i]
                
                features = {
                    'stock': stock_seq,
                    'news': news_seq,
                    'employment': employment_seq
                }
                
                all_features.append(features)
                
                # Target values
                for horizon in [30, 180, 365, 730]:
                    target_idx = i - self.sequence_length
                    if target_idx < len(targets[f'horizon_{horizon}']):
                        all_targets[f'horizon_{horizon}'].append(targets[f'horizon_{horizon}'][target_idx])
        
        return {
            'features': all_features,
            'targets': all_targets,
            'feature_names': {
                'stock': stock_features,
                'news': news_features,
                'employment': employment_features
            }
        }
    
    def _split_data(self, dataset: Dict) -> Dict:
        """Split data into train/validation/test sets"""
        n_samples = len(dataset['features'])
        
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        splits = {}
        
        for split_name, (start, end) in [
            ('train', (0, train_end)),
            ('val', (train_end, val_end)),
            ('test', (val_end, n_samples))
        ]:
            # Features
            split_features = {}
            for modality in ['stock', 'news', 'employment']:
                modality_data = [dataset['features'][i][modality] for i in range(start, end)]
                split_features[modality] = np.array(modality_data)
            
            # Targets
            split_targets = {}
            for horizon in [30, 180, 365, 730]:
                horizon_key = f'horizon_{horizon}'
                split_targets[horizon_key] = np.array(dataset['targets'][horizon_key][start:end])
            
            splits[split_name] = {
                'features': split_features,
                'targets': split_targets
            }
        
        splits['feature_names'] = dataset['feature_names']
        
        return splits
    
    def _normalize_features(self, splits: Dict) -> Dict:
        """Normalize features using training set statistics"""
        
        # Calculate normalization parameters from training set
        normalization_params = {}
        
        for modality in ['stock', 'news', 'employment']:
            train_data = splits['train']['features'][modality]
            
            # Calculate mean and std across samples and time
            mean = np.mean(train_data, axis=(0, 1), keepdims=True)
            std = np.std(train_data, axis=(0, 1), keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            
            normalization_params[modality] = {'mean': mean, 'std': std}
        
        # Apply normalization to all splits
        normalized_splits = {}
        
        for split_name in ['train', 'val', 'test']:
            normalized_features = {}
            
            for modality in ['stock', 'news', 'employment']:
                data = splits[split_name]['features'][modality]
                mean = normalization_params[modality]['mean']
                std = normalization_params[modality]['std']
                
                normalized_data = (data - mean) / std
                normalized_features[modality] = torch.FloatTensor(normalized_data)
            
            # Convert targets to tensors
            normalized_targets = {}
            for horizon_key, target_data in splits[split_name]['targets'].items():
                normalized_targets[horizon_key] = torch.FloatTensor(target_data)
            
            normalized_splits[split_name] = {
                'features': normalized_features,
                'targets': normalized_targets
            }
        
        # Add metadata
        normalized_splits['normalization_params'] = normalization_params
        normalized_splits['feature_names'] = splits['feature_names']
        
        return normalized_splits
    
    def create_data_loaders(self, dataset_splits: Dict, batch_size: int = 32,
                           num_workers: int = 4) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders for training"""
        
        data_loaders = {}
        
        for split_name in ['train', 'val', 'test']:
            dataset = MultiModalDataset(
                features=dataset_splits[split_name]['features'],
                targets=dataset_splits[split_name]['targets'],
                sequence_length=self.sequence_length
            )
            
            shuffle = (split_name == 'train')
            
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
            
            data_loaders[split_name] = data_loader
        
        return data_loaders