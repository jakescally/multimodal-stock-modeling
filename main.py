"""
Multimodal Stock Recommendation Model
====================================

A hybrid model combining time series analysis with qualitative data
for short/medium/long-term stock performance prediction.

Architecture:
- TFT (Temporal Fusion Transformer) for time series data
- BERT-based encoder for text/news data  
- Employment data integration
- Cross-modal fusion with attention
- Multi-task prediction heads
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import model components
from models.tft_encoder import TFTEncoder
from models.text_encoder import FinancialTextEncoder
from models.employment_encoder import EmploymentEncoder
from models.fusion_layer import MultiModalFusionLayer, FusionConfig
from models.prediction_heads import MultiTaskPredictionHead, PredictionConfig

@dataclass
class ModelConfig:
    """Configuration for the multimodal stock model"""
    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    
    # Time series config
    sequence_length: int = 252  # ~1 trading year
    prediction_horizons: List[int] = None  # [30, 180, 365, 730] days
    
    # Text config
    text_model: str = "bert-base-uncased"
    max_text_length: int = 512
    
    # Fusion config
    fusion_strategy: str = "cross_attention"  # "cross_attention", "gated_fusion", "hierarchical", "adaptive"
    
    # Training config
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [30, 180, 365, 730]  # 1M, 6M, 1Y, 2Y
    
    def get_fusion_config(self) -> FusionConfig:
        """Get fusion layer configuration"""
        return FusionConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            fusion_strategy=self.fusion_strategy,
            stock_dim=self.d_model,
            text_dim=self.d_model,
            employment_dim=self.d_model
        )
    
    def get_prediction_config(self) -> PredictionConfig:
        """Get prediction head configuration"""
        return PredictionConfig(
            d_model=self.d_model,
            prediction_horizons=self.prediction_horizons,
            dropout=self.dropout
        )

class StockDataProcessor:
    """Handles data preprocessing for all modalities"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def process_time_series(self, df: pd.DataFrame) -> torch.Tensor:
        """Process OHLCV and technical indicators"""
        pass
        
    def process_text_data(self, texts: List[str]) -> torch.Tensor:
        """Process news, earnings, etc."""
        pass
        
    def process_employment_data(self, employment_df: pd.DataFrame) -> torch.Tensor:
        """Process job postings, layoffs, hiring data"""
        pass

class MultiModalStockModel(nn.Module):
    """Main multimodal model architecture"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Simple linear encoders for demonstration (can be replaced with full encoders)
        self.stock_encoder = nn.Sequential(
            nn.Linear(15, config.d_model),  # Assuming 15 stock features
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(10, config.d_model),  # Assuming 10 text features
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.employment_encoder = nn.Sequential(
            nn.Linear(8, config.d_model),   # Assuming 8 employment features
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Cross-modal fusion layer
        self.fusion_layer = MultiModalFusionLayer(config.get_fusion_config())
        
        # Multi-task prediction heads
        self.prediction_heads = MultiTaskPredictionHead(config.get_prediction_config())
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all modalities
        
        Args:
            batch: Dictionary containing:
                - 'stock': [batch, seq_len, stock_features]
                - 'news': [batch, news_features] or [batch, seq_len, news_features]
                - 'employment': [batch, employment_features] or [batch, seq_len, employment_features]
                
        Returns:
            Dictionary with all predictions and intermediate outputs
        """
        # Extract inputs
        stock_data = batch['stock']
        news_data = batch['news']
        employment_data = batch['employment']
        
        # Get dimensions from stock data
        batch_size, seq_len = stock_data.size(0), stock_data.size(1)
        
        # Handle different input shapes for news and employment data
        # If they are 2D [batch, features], expand to 3D [batch, seq_len, features]
        if len(news_data.shape) == 2:
            # Expand news data to match sequence length
            news_data = news_data.unsqueeze(1).expand(-1, seq_len, -1)
        
        if len(employment_data.shape) == 2:
            # Expand employment data to match sequence length  
            employment_data = employment_data.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Encode each modality
        stock_encoded = self.stock_encoder(stock_data)
        text_encoded = self.text_encoder(news_data)
        employment_encoded = self.employment_encoder(employment_data)
        
        # Cross-modal fusion
        modality_features = {
            'stock': stock_encoded,
            'text': text_encoded,
            'employment': employment_encoded
        }
        
        fusion_output = self.fusion_layer(modality_features)
        fused_features = fusion_output['fused_features']
        
        # Multi-task predictions
        predictions = self.prediction_heads(fused_features)
        
        # Return comprehensive output
        return {
            'predictions': predictions,
            'fusion_output': fusion_output,
            'encoded_features': {
                'stock': stock_encoded,
                'text': text_encoded,
                'employment': employment_encoded
            }
        }

if __name__ == "__main__":
    config = ModelConfig()
    print(f"Initialized model config with {len(config.prediction_horizons)} prediction horizons")
    print(f"Prediction horizons: {config.prediction_horizons} days")