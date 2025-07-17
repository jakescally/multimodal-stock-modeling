"""
Multi-Task Prediction Heads
===========================

Prediction heads for different horizons and tasks including:
- Multi-horizon return prediction (30d, 180d, 365d, 730d)
- Volatility prediction
- Direction classification
- Economic indicator prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PredictionConfig:
    """Configuration for prediction heads"""
    d_model: int = 256
    prediction_horizons: List[int] = None
    dropout: float = 0.1
    
    # Task configurations
    predict_returns: bool = True
    predict_volatility: bool = True
    predict_direction: bool = True
    predict_economic: bool = True
    
    # Head architectures
    use_separate_heads: bool = True
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [30, 180, 365, 730]


class ReturnPredictionHead(nn.Module):
    """Predicts future returns for multiple horizons"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        self.horizons = config.prediction_horizons
        
        if config.use_separate_heads:
            # Separate head for each horizon
            self.horizon_heads = nn.ModuleDict()
            for horizon in self.horizons:
                self.horizon_heads[f'horizon_{horizon}'] = self._create_prediction_head(
                    config.d_model, 1, config.hidden_dim, config.n_hidden_layers, config.dropout
                )
        else:
            # Shared head with horizon embedding
            self.horizon_embedding = nn.Embedding(len(self.horizons), config.d_model // 4)
            self.shared_head = self._create_prediction_head(
                config.d_model + config.d_model // 4, len(self.horizons), 
                config.hidden_dim, config.n_hidden_layers, config.dropout
            )
        
        # Uncertainty estimation heads
        self.uncertainty_heads = nn.ModuleDict()
        for horizon in self.horizons:
            self.uncertainty_heads[f'horizon_{horizon}'] = nn.Sequential(
                nn.Linear(config.d_model, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
                nn.Softplus()  # Ensure positive uncertainty
            )
    
    def _create_prediction_head(self, input_dim: int, output_dim: int, 
                               hidden_dim: int, n_layers: int, dropout: float) -> nn.Module:
        """Create a multi-layer prediction head"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, d_model] - aggregated sequence features
            
        Returns:
            Dictionary with predictions and uncertainties for each horizon
        """
        predictions = {}
        uncertainties = {}
        
        if self.config.use_separate_heads:
            # Separate heads
            for horizon in self.horizons:
                horizon_key = f'horizon_{horizon}'
                predictions[horizon_key] = self.horizon_heads[horizon_key](features).squeeze(-1)
                uncertainties[horizon_key] = self.uncertainty_heads[horizon_key](features).squeeze(-1)
        else:
            # Shared head with horizon embeddings
            batch_size = features.size(0)
            horizon_indices = torch.arange(len(self.horizons), device=features.device)
            horizon_indices = horizon_indices.unsqueeze(0).expand(batch_size, -1)
            
            horizon_embeds = self.horizon_embedding(horizon_indices)  # [batch, n_horizons, embed_dim]
            
            # Expand features for each horizon
            features_expanded = features.unsqueeze(1).expand(-1, len(self.horizons), -1)
            
            # Concatenate features with horizon embeddings
            combined_features = torch.cat([features_expanded, horizon_embeds], dim=-1)
            combined_features = combined_features.view(-1, combined_features.size(-1))
            
            # Predict all horizons at once
            all_predictions = self.shared_head(combined_features)
            all_predictions = all_predictions.view(batch_size, len(self.horizons), -1)
            
            # Extract predictions for each horizon
            for i, horizon in enumerate(self.horizons):
                horizon_key = f'horizon_{horizon}'
                predictions[horizon_key] = all_predictions[:, i, 0]
                uncertainties[horizon_key] = self.uncertainty_heads[horizon_key](features).squeeze(-1)
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties
        }


class VolatilityPredictionHead(nn.Module):
    """Predicts future volatility for multiple horizons"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        self.horizons = config.prediction_horizons
        
        # Volatility prediction heads (always positive output)
        self.volatility_heads = nn.ModuleDict()
        for horizon in self.horizons:
            self.volatility_heads[f'horizon_{horizon}'] = nn.Sequential(
                nn.Linear(config.d_model, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Softplus()  # Ensure positive volatility
            )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict volatility for each horizon"""
        volatility_predictions = {}
        
        for horizon in self.horizons:
            horizon_key = f'horizon_{horizon}'
            volatility_predictions[horizon_key] = self.volatility_heads[horizon_key](features).squeeze(-1)
        
        return volatility_predictions


class DirectionClassificationHead(nn.Module):
    """Classifies price direction (up/down/flat) for multiple horizons"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        self.horizons = config.prediction_horizons
        self.n_classes = 3  # up, down, flat
        
        # Direction classification heads
        self.direction_heads = nn.ModuleDict()
        for horizon in self.horizons:
            self.direction_heads[f'horizon_{horizon}'] = nn.Sequential(
                nn.Linear(config.d_model, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, self.n_classes)
            )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict direction probabilities for each horizon"""
        direction_predictions = {}
        
        for horizon in self.horizons:
            horizon_key = f'horizon_{horizon}'
            logits = self.direction_heads[horizon_key](features)
            probabilities = F.softmax(logits, dim=-1)
            direction_predictions[horizon_key] = {
                'logits': logits,
                'probabilities': probabilities
            }
        
        return direction_predictions


class EconomicIndicatorHead(nn.Module):
    """Predicts broader economic indicators"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # Economic indicators to predict
        self.indicators = ['market_sentiment', 'sector_performance', 'economic_health']
        
        self.indicator_heads = nn.ModuleDict()
        for indicator in self.indicators:
            if indicator == 'market_sentiment':
                # Sentiment score (-1 to 1)
                self.indicator_heads[indicator] = nn.Sequential(
                    nn.Linear(config.d_model, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                    nn.Tanh()
                )
            elif indicator == 'sector_performance':
                # Relative sector performance (0 to 1)
                self.indicator_heads[indicator] = nn.Sequential(
                    nn.Linear(config.d_model, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                    nn.Sigmoid()
                )
            elif indicator == 'economic_health':
                # Economic health score (0 to 1)
                self.indicator_heads[indicator] = nn.Sequential(
                    nn.Linear(config.d_model, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                    nn.Sigmoid()
                )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict economic indicators"""
        economic_predictions = {}
        
        for indicator in self.indicators:
            economic_predictions[indicator] = self.indicator_heads[indicator](features).squeeze(-1)
        
        return economic_predictions


class MultiTaskPredictionHead(nn.Module):
    """Combined multi-task prediction head for all tasks"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # Feature aggregation layer
        self.feature_aggregator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        
        if config.predict_returns:
            self.heads['returns'] = ReturnPredictionHead(config)
        
        if config.predict_volatility:
            self.heads['volatility'] = VolatilityPredictionHead(config)
        
        if config.predict_direction:
            self.heads['direction'] = DirectionClassificationHead(config)
        
        if config.predict_economic:
            self.heads['economic'] = EconomicIndicatorHead(config)
        
        # Task importance weights (learnable)
        n_tasks = len(self.heads)
        self.task_weights = nn.Parameter(torch.ones(n_tasks))
        
    def forward(self, sequence_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequence_features: [batch, seq_len, d_model] or [batch, d_model]
            
        Returns:
            Dictionary with all task predictions
        """
        # Handle both sequence and aggregated features
        if sequence_features.dim() == 3:
            # Aggregate sequence features (use last timestep + global average)
            last_timestep = sequence_features[:, -1, :]
            global_avg = sequence_features.mean(dim=1)
            aggregated_features = (last_timestep + global_avg) / 2
        else:
            aggregated_features = sequence_features
        
        # Apply feature aggregation
        features = self.feature_aggregator(aggregated_features)
        
        # Apply each prediction head
        all_predictions = {}
        
        for task_name, head in self.heads.items():
            all_predictions[task_name] = head(features)
        
        # Add task weights for loss weighting
        task_weights = F.softmax(self.task_weights, dim=0)
        all_predictions['task_weights'] = {
            task_name: task_weights[i] 
            for i, task_name in enumerate(self.heads.keys())
        }
        
        return all_predictions


class AdaptiveAggregator(nn.Module):
    """Adaptive aggregation of sequence features for prediction"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Self-attention for sequence aggregation
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        
        # Learnable query for aggregation
        self.aggregation_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_features: [batch, seq_len, d_model]
            
        Returns:
            aggregated_features: [batch, d_model]
        """
        batch_size = sequence_features.size(0)
        
        # Expand query for batch
        query = self.aggregation_query.expand(batch_size, -1, -1)
        
        # Apply attention-based aggregation
        aggregated, attention_weights = self.self_attention(
            query, sequence_features, sequence_features
        )
        
        # Apply layer norm and extract single vector
        aggregated = self.layer_norm(aggregated.squeeze(1))
        
        return aggregated


class UncertaintyQuantification(nn.Module):
    """Quantifies prediction uncertainty using ensemble or Bayesian methods"""
    
    def __init__(self, d_model: int, n_samples: int = 10):
        super().__init__()
        self.d_model = d_model
        self.n_samples = n_samples
        
        # Dropout layers for Monte Carlo sampling
        self.mc_dropout = nn.Dropout(0.1)
        
        # Variance estimation head
        self.variance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        
    def forward(self, features: torch.Tensor, training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Estimate prediction uncertainty using Monte Carlo dropout
        
        Args:
            features: [batch, d_model]
            training: Whether to use MC dropout
            
        Returns:
            Dictionary with mean predictions and uncertainty estimates
        """
        if training or self.training:
            # During training, return single prediction with variance estimate
            variance = self.variance_head(features)
            return {
                'mean': features,
                'variance': variance,
                'std': torch.sqrt(variance + 1e-6)
            }
        else:
            # During inference, use Monte Carlo sampling
            predictions = []
            
            # Enable MC dropout
            self.train()
            
            for _ in range(self.n_samples):
                with torch.no_grad():
                    sample_features = self.mc_dropout(features)
                    predictions.append(sample_features)
            
            # Restore original mode
            self.eval()
            
            # Calculate statistics
            predictions = torch.stack(predictions, dim=0)  # [n_samples, batch, d_model]
            mean_pred = predictions.mean(dim=0)
            var_pred = predictions.var(dim=0)
            
            return {
                'mean': mean_pred,
                'variance': var_pred,
                'std': torch.sqrt(var_pred + 1e-6),
                'samples': predictions
            }