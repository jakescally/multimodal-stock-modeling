"""
Basic functionality tests for multimodal stock model components
"""

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ModelConfig
from models.tft_encoder import TFTEncoder, GatedResidualNetwork
from models.employment_encoder import EmploymentEncoder, EmploymentFeatures, EmploymentSignalGenerator


class TestModelConfig:
    """Test model configuration"""
    
    def test_default_config(self):
        config = ModelConfig()
        assert config.d_model == 256
        assert config.n_heads == 8
        assert len(config.prediction_horizons) == 4
        assert 30 in config.prediction_horizons
        assert 730 in config.prediction_horizons
        
    def test_custom_config(self):
        config = ModelConfig(
            d_model=128,
            prediction_horizons=[30, 90]
        )
        assert config.d_model == 128
        assert config.prediction_horizons == [30, 90]


class TestTFTEncoder:
    """Test Temporal Fusion Transformer encoder"""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(d_model=64, sequence_length=20)
    
    @pytest.fixture  
    def encoder(self, config):
        return TFTEncoder(config)
    
    def test_encoder_initialization(self, encoder):
        assert encoder is not None
        assert encoder.config.d_model == 64
        
    def test_forward_pass(self, encoder):
        batch_size, seq_len = 2, 20
        static_features = torch.randn(batch_size, encoder.static_input_size)
        historical_features = torch.randn(batch_size, seq_len, encoder.historical_input_size)
        
        with torch.no_grad():
            outputs = encoder(static_features, historical_features)
            
        assert 'predictions' in outputs
        assert 'attention_weights' in outputs
        assert len(outputs['predictions']) == len(encoder.config.prediction_horizons)
        
        # Check prediction shapes
        for horizon_pred in outputs['predictions'].values():
            assert horizon_pred.shape == (batch_size, 1)


class TestGatedResidualNetwork:
    """Test Gated Residual Network component"""
    
    def test_grn_forward(self):
        grn = GatedResidualNetwork(10, 20, 15, dropout=0.0)
        x = torch.randn(5, 10)
        
        output = grn(x)
        assert output.shape == (5, 15)
        
    def test_grn_with_context(self):
        grn = GatedResidualNetwork(10, 20, 15, dropout=0.0, context_size=8)
        x = torch.randn(5, 10)
        context = torch.randn(5, 8)
        
        output = grn(x, context)
        assert output.shape == (5, 15)


class TestEmploymentEncoder:
    """Test employment encoder functionality"""
    
    @pytest.fixture
    def config(self):
        return ModelConfig(d_model=32)
    
    @pytest.fixture
    def encoder(self, config):
        return EmploymentEncoder(config)
    
    def test_encoder_initialization(self, encoder):
        assert encoder is not None
        assert encoder.config.d_model == 32
        
    def test_forward_pass(self, encoder):
        batch_size, seq_len = 2, 10
        
        employment_data = {
            'company_features': torch.randn(batch_size, seq_len, encoder.company_features_dim),
            'sector_features': torch.randn(batch_size, seq_len, encoder.sector_features_dim),
            'macro_features': torch.randn(batch_size, seq_len, encoder.macro_features_dim)
        }
        
        with torch.no_grad():
            outputs = encoder(employment_data)
            
        assert 'employment_embeddings' in outputs
        assert 'impact_scores' in outputs
        assert outputs['employment_embeddings'].shape == (batch_size, seq_len, encoder.config.d_model)


class TestEmploymentSignalGenerator:
    """Test employment signal generation"""
    
    @pytest.fixture
    def signal_gen(self):
        return EmploymentSignalGenerator()
    
    @pytest.fixture
    def sample_features(self):
        return EmploymentFeatures(
            job_postings_30d=100,
            job_postings_90d=300,
            layoffs_30d=0,
            layoffs_90d=10,
            hiring_velocity=0.2,
            ai_ml_demand=0.05,
            technical_skills_demand=0.15,
            leadership_demand=0.08,
            sector_hiring_ratio=1.1,
            sector_layoff_ratio=0.9,
            unemployment_rate=4.0,
            job_openings_rate=5.5
        )
    
    def test_signal_generation(self, signal_gen, sample_features):
        signals = signal_gen.generate_signals(sample_features)
        
        assert isinstance(signals, dict)
        assert 'hiring_acceleration' in signals
        assert 'layoff_deceleration' in signals
        assert 'skill_demand_growth' in signals
        assert 'sector_outperformance' in signals
        
        # Check signal values are in reasonable range
        for signal_value in signals.values():
            assert 0.0 <= signal_value <= 1.0
            
    def test_signal_aggregation(self, signal_gen, sample_features):
        signals = signal_gen.generate_signals(sample_features)
        overall_signal = signal_gen.aggregate_signal(signals)
        
        assert isinstance(overall_signal, float)
        assert 0.0 <= overall_signal <= 1.0


class TestDataProcessing:
    """Test data processing utilities"""
    
    def test_stock_data_creation(self):
        """Test creation of sample stock data"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        stock_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 200, 10),
            'high': np.random.uniform(150, 250, 10),
            'low': np.random.uniform(80, 150, 10),
            'close': np.random.uniform(90, 220, 10),
            'volume': np.random.randint(1000000, 10000000, 10)
        })
        
        assert len(stock_data) == 10
        assert 'date' in stock_data.columns
        assert 'close' in stock_data.columns
        
    def test_employment_data_creation(self):
        """Test creation of sample employment data"""
        employment_data = pd.DataFrame({
            'company_id': ['AAPL'] * 5,
            'posting_date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'description': ['Software Engineer'] * 5
        })
        
        assert len(employment_data) == 5
        assert 'company_id' in employment_data.columns
        assert 'posting_date' in employment_data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])