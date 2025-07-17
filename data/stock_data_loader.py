"""
Stock Data Loader
================

Handles downloading and preprocessing of stock market data from various sources.
Supports yfinance, Alpha Vantage, and other financial data providers.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import requests
import time
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """Main class for loading and caching stock data"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_stock_data(self, symbol: str, period: str = "2y", 
                      interval: str = "1d") -> pd.DataFrame:
        """
        Get stock data for a single symbol
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        """
        cache_key = f"{symbol}_{period}_{interval}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache first
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=1):  # Cache for 1 hour
                logger.info(f"Loading {symbol} from cache")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        logger.info(f"Downloading {symbol} data from Yahoo Finance")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
                
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "2y", 
                           interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get data for multiple stock symbols"""
        results = {}
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}")
            data = self.get_stock_data(symbol, period, interval)
            if not data.empty:
                results[symbol] = data
            time.sleep(0.1)  # Rate limiting
            
        return results
    
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 stock symbols from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            return sp500_table['Symbol'].tolist()
        except Exception as e:
            logger.error(f"Error getting S&P 500 symbols: {e}")
            # Fallback to common symbols
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']


class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Add simple moving averages"""
        df = df.copy()
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        return df
    
    @staticmethod
    def add_exponential_moving_averages(df: pd.DataFrame, windows: List[int] = [12, 26]) -> pd.DataFrame:
        """Add exponential moving averages"""
        df = df.copy()
        for window in windows:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        df = df.copy()
        
        # Calculate price changes
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        df = df.copy()
        
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df = df.copy()
        
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        
        df['BB_Middle'] = rolling_mean
        df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        return df
    
    @staticmethod
    def add_volatility(df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
        """Add volatility measures"""
        df = df.copy()
        
        # Daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Rolling volatility
        for window in windows:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
            
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        df = df.copy()
        
        # Volume moving averages
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume Price Trend
        df['VPT'] = (df['Volume'] * (df['Close'].diff() / df['Close'].shift(1))).fillna(0).cumsum()
        
        return df
    
    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators"""
        df = cls.add_moving_averages(df)
        df = cls.add_exponential_moving_averages(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_bollinger_bands(df)
        df = cls.add_volatility(df)
        df = cls.add_volume_indicators(df)
        
        return df


class MarketDataProcessor:
    """Process and prepare market data for ML models"""
    
    def __init__(self, sequence_length: int = 252):
        self.sequence_length = sequence_length
        
    def create_sequences(self, df: pd.DataFrame, target_column: str = 'Close',
                        horizon_days: List[int] = [30, 180, 365, 730]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences for time series modeling
        
        Args:
            df: DataFrame with stock data and technical indicators
            target_column: Column to predict
            horizon_days: Prediction horizons in days
            
        Returns:
            features: Array of shape (n_samples, sequence_length, n_features)
            targets: Dict of arrays for each horizon
        """
        # Select feature columns (exclude target and date columns)
        feature_cols = [col for col in df.columns if col not in ['Date'] and 
                       not col.startswith('target_')]
        
        # Create target columns for each horizon
        targets = {}
        for horizon in horizon_days:
            # Future return calculation
            future_price = df[target_column].shift(-horizon)
            current_price = df[target_column]
            targets[f'horizon_{horizon}'] = (future_price / current_price - 1).values
        
        # Prepare features
        feature_data = df[feature_cols].values
        
        # Create sequences
        sequences = []
        target_sequences = {f'horizon_{h}': [] for h in horizon_days}
        
        for i in range(self.sequence_length, len(df) - max(horizon_days)):
            # Features sequence
            seq = feature_data[i-self.sequence_length:i]
            sequences.append(seq)
            
            # Target values
            for horizon in horizon_days:
                target_sequences[f'horizon_{horizon}'].append(targets[f'horizon_{horizon}'][i])
        
        sequences = np.array(sequences)
        target_arrays = {k: np.array(v) for k, v in target_sequences.items()}
        
        return sequences, target_arrays
    
    def normalize_features(self, data: np.ndarray, fit_on_train: bool = True,
                          scaler_params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Normalize features using z-score normalization
        
        Args:
            data: Input data of shape (n_samples, sequence_length, n_features)
            fit_on_train: Whether to fit scaler on this data
            scaler_params: Pre-fitted scaler parameters (mean, std)
            
        Returns:
            normalized_data: Normalized data
            scaler_params: Scaler parameters for inverse transform
        """
        if fit_on_train:
            # Calculate mean and std across samples and time
            mean = np.mean(data, axis=(0, 1), keepdims=True)
            std = np.std(data, axis=(0, 1), keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            
            scaler_params = {'mean': mean, 'std': std}
        else:
            mean = scaler_params['mean']
            std = scaler_params['std']
        
        normalized_data = (data - mean) / std
        
        return normalized_data, scaler_params
    
    def split_data(self, features: np.ndarray, targets: Dict[str, np.ndarray],
                  train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """
        Split data into train/validation/test sets with temporal ordering
        """
        n_samples = len(features)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        return {
            'train': {
                'features': features[:train_end],
                'targets': {k: v[:train_end] for k, v in targets.items()}
            },
            'val': {
                'features': features[train_end:val_end],
                'targets': {k: v[train_end:val_end] for k, v in targets.items()}
            },
            'test': {
                'features': features[val_end:],
                'targets': {k: v[val_end:] for k, v in targets.items()}
            }
        }


class DatasetBuilder:
    """High-level interface for building complete datasets"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.loader = StockDataLoader(cache_dir)
        self.processor = MarketDataProcessor()
        
    def build_stock_dataset(self, symbols: List[str], period: str = "5y",
                           sequence_length: int = 252) -> Dict:
        """
        Build a complete dataset for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            sequence_length: Sequence length for model
            
        Returns:
            Complete dataset ready for training
        """
        logger.info(f"Building dataset for {len(symbols)} symbols")
        
        # Download stock data
        stock_data = self.loader.get_multiple_stocks(symbols, period)
        
        # Process each stock
        all_features = []
        all_targets = {f'horizon_{h}': [] for h in [30, 180, 365, 730]}
        
        for symbol, df in stock_data.items():
            if len(df) < sequence_length + 730:  # Need enough data
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                continue
                
            logger.info(f"Processing {symbol}: {len(df)} rows")
            
            # Add technical indicators
            df_processed = TechnicalIndicators.add_all_indicators(df)
            
            # Remove rows with NaN values
            df_processed = df_processed.dropna()
            
            if len(df_processed) < sequence_length + 730:
                logger.warning(f"Insufficient data after processing for {symbol}")
                continue
            
            # Create sequences
            features, targets = self.processor.create_sequences(df_processed)
            
            if len(features) > 0:
                all_features.append(features)
                for horizon, target_data in targets.items():
                    all_targets[horizon].extend(target_data)
        
        if not all_features:
            raise ValueError("No valid data found for any symbols")
        
        # Combine all data
        combined_features = np.concatenate(all_features, axis=0)
        combined_targets = {k: np.array(v) for k, v in all_targets.items()}
        
        logger.info(f"Combined dataset shape: {combined_features.shape}")
        
        # Split data
        splits = self.processor.split_data(combined_features, combined_targets)
        
        # Normalize features
        train_features_norm, scaler_params = self.processor.normalize_features(
            splits['train']['features'], fit_on_train=True
        )
        val_features_norm, _ = self.processor.normalize_features(
            splits['val']['features'], fit_on_train=False, scaler_params=scaler_params
        )
        test_features_norm, _ = self.processor.normalize_features(
            splits['test']['features'], fit_on_train=False, scaler_params=scaler_params
        )
        
        return {
            'train': {
                'features': train_features_norm,
                'targets': splits['train']['targets']
            },
            'val': {
                'features': val_features_norm,
                'targets': splits['val']['targets']
            },
            'test': {
                'features': test_features_norm,
                'targets': splits['test']['targets']
            },
            'scaler_params': scaler_params,
            'feature_names': [col for col in stock_data[list(stock_data.keys())[0]].columns 
                             if col not in ['Date'] and not col.startswith('target_')]
        }