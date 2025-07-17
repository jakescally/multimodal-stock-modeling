"""
Loss Functions for Multimodal Stock Prediction
=============================================

Specialized loss functions for multi-task financial prediction including:
- Multi-horizon return prediction losses
- Volatility prediction losses  
- Direction classification losses
- Economic indicator losses
- Uncertainty-aware losses
- Multi-task loss combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossConfig:
    """Configuration for loss functions"""
    # Loss weights for different tasks
    return_loss_weight: float = 1.0
    volatility_loss_weight: float = 0.5
    direction_loss_weight: float = 0.3
    economic_loss_weight: float = 0.2
    
    # Horizon-specific weights (closer horizons more important)
    horizon_weights: Dict[str, float] = None
    
    # Loss function types
    return_loss_type: str = "mse"  # "mse", "mae", "huber", "quantile"
    direction_loss_type: str = "cross_entropy"
    
    # Uncertainty parameters
    use_uncertainty: bool = True
    uncertainty_weight: float = 0.1
    
    # Regularization
    l1_weight: float = 0.0
    l2_weight: float = 1e-4
    
    def __post_init__(self):
        if self.horizon_weights is None:
            # Closer horizons get higher weights
            self.horizon_weights = {
                'horizon_30': 1.0,
                'horizon_180': 0.8,
                'horizon_365': 0.6,
                'horizon_730': 0.4
            }


class ReturnPredictionLoss(nn.Module):
    """Loss function for multi-horizon return prediction"""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.return_loss_type
        
        # Initialize loss function
        if self.loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif self.loss_type == "mae":
            self.base_loss = nn.L1Loss(reduction='none')
        elif self.loss_type == "huber":
            self.base_loss = nn.HuberLoss(reduction='none', delta=0.1)
        elif self.loss_type == "quantile":
            self.quantiles = [0.1, 0.5, 0.9]  # For quantile regression
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                uncertainties: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate return prediction loss
        
        Args:
            predictions: Dictionary of horizon predictions
            targets: Dictionary of horizon targets
            uncertainties: Optional uncertainty estimates
            
        Returns:
            Dictionary with individual and total losses
        """
        horizon_losses = {}
        total_loss = 0.0
        
        for horizon_key in predictions:
            if horizon_key in targets:
                pred = predictions[horizon_key]
                target = targets[horizon_key]
                
                # Base loss calculation
                if self.loss_type == "quantile":
                    loss = self._quantile_loss(pred, target)
                else:
                    loss = self.base_loss(pred, target)
                
                # Apply uncertainty weighting if available
                if uncertainties is not None and horizon_key in uncertainties:
                    uncertainty = uncertainties[horizon_key]
                    # Heteroscedastic loss: loss / (2 * variance) + log(variance)
                    loss = loss / (2 * uncertainty + 1e-6) + torch.log(uncertainty + 1e-6)
                
                # Apply horizon weighting
                horizon_weight = self.config.horizon_weights.get(horizon_key, 1.0)
                weighted_loss = horizon_weight * loss.mean()
                
                horizon_losses[horizon_key] = weighted_loss
                total_loss += weighted_loss
        
        return {
            'total': total_loss,
            'horizons': horizon_losses
        }
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Quantile regression loss for multiple quantiles"""
        # Assuming predictions are [batch, n_quantiles]
        losses = []
        
        for i, quantile in enumerate(self.quantiles):
            pred_q = predictions[:, i] if predictions.dim() > 1 else predictions
            errors = targets - pred_q
            
            quantile_loss = torch.max(
                quantile * errors,
                (quantile - 1) * errors
            )
            losses.append(quantile_loss)
        
        return torch.stack(losses).mean(dim=0)


class VolatilityPredictionLoss(nn.Module):
    """Loss function for volatility prediction"""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate volatility prediction loss"""
        horizon_losses = {}
        total_loss = 0.0
        
        for horizon_key in predictions:
            if horizon_key in targets:
                pred = predictions[horizon_key]
                target = targets[horizon_key]
                
                # Ensure positive volatility predictions
                pred = F.softplus(pred)
                target = torch.abs(target)  # Ensure positive targets
                
                # Use MSE for volatility (could also use MAE)
                loss = F.mse_loss(pred, target)
                
                # Apply horizon weighting
                horizon_weight = self.config.horizon_weights.get(horizon_key, 1.0)
                weighted_loss = horizon_weight * loss
                
                horizon_losses[horizon_key] = weighted_loss
                total_loss += weighted_loss
        
        return {
            'total': total_loss,
            'horizons': horizon_losses
        }


class DirectionClassificationLoss(nn.Module):
    """Loss function for direction classification"""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predictions: Dict[str, Dict[str, torch.Tensor]], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate direction classification loss"""
        horizon_losses = {}
        total_loss = 0.0
        
        for horizon_key in predictions:
            if horizon_key in targets:
                logits = predictions[horizon_key]['logits']
                target = targets[horizon_key]
                
                # Convert continuous returns to direction classes
                # 0: down (< -0.01), 1: flat (-0.01 to 0.01), 2: up (> 0.01)
                direction_targets = torch.zeros_like(target, dtype=torch.long)
                direction_targets[target > 0.01] = 2   # Up
                direction_targets[target < -0.01] = 0  # Down
                direction_targets[(target >= -0.01) & (target <= 0.01)] = 1  # Flat
                
                loss = self.loss_fn(logits, direction_targets).mean()
                
                # Apply horizon weighting
                horizon_weight = self.config.horizon_weights.get(horizon_key, 1.0)
                weighted_loss = horizon_weight * loss
                
                horizon_losses[horizon_key] = weighted_loss
                total_loss += weighted_loss
        
        return {
            'total': total_loss,
            'horizons': horizon_losses
        }


class EconomicIndicatorLoss(nn.Module):
    """Loss function for economic indicators"""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate economic indicator loss"""
        total_loss = 0.0
        
        for indicator in predictions:
            if indicator in targets:
                pred = predictions[indicator]
                target = targets[indicator]
                
                # Different loss functions for different indicators
                if indicator == 'market_sentiment':
                    # MSE for sentiment (-1 to 1)
                    loss = F.mse_loss(pred, target)
                elif indicator in ['sector_performance', 'economic_health']:
                    # MSE for bounded indicators (0 to 1)
                    loss = F.mse_loss(pred, target)
                else:
                    loss = F.mse_loss(pred, target)
                
                total_loss += loss
        
        return total_loss


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss with adaptive weighting"""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        # Individual loss functions
        self.return_loss = ReturnPredictionLoss(config)
        self.volatility_loss = VolatilityPredictionLoss(config)
        self.direction_loss = DirectionClassificationLoss(config)
        self.economic_loss = EconomicIndicatorLoss(config)
        
        # Learnable task weights (uncertainty-based weighting)
        self.log_vars = nn.Parameter(torch.zeros(4))  # For 4 tasks
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined multi-task loss
        
        Args:
            predictions: Model predictions from all tasks
            targets: Ground truth targets for all tasks
            
        Returns:
            Dictionary with detailed loss breakdown
        """
        losses = {}
        task_losses = []
        
        # Return prediction loss
        if 'returns' in predictions:
            return_preds = predictions['returns']['predictions']
            return_uncerts = predictions['returns'].get('uncertainties', None)
            return_targets = {k: v for k, v in targets.items() if k.startswith('horizon_')}
            
            return_loss_dict = self.return_loss(return_preds, return_targets, return_uncerts)
            losses['returns'] = return_loss_dict
            task_losses.append(return_loss_dict['total'])
        
        # Volatility prediction loss
        if 'volatility' in predictions:
            vol_preds = predictions['volatility']
            vol_targets = {k.replace('horizon_', 'vol_'): torch.abs(v) 
                          for k, v in targets.items() if k.startswith('horizon_')}
            
            vol_loss_dict = self.volatility_loss(vol_preds, vol_targets)
            losses['volatility'] = vol_loss_dict
            task_losses.append(vol_loss_dict['total'])
        
        # Direction classification loss
        if 'direction' in predictions:
            dir_preds = predictions['direction']
            dir_targets = {k: v for k, v in targets.items() if k.startswith('horizon_')}
            
            dir_loss_dict = self.direction_loss(dir_preds, dir_targets)
            losses['direction'] = dir_loss_dict
            task_losses.append(dir_loss_dict['total'])
        
        # Economic indicator loss
        if 'economic' in predictions:
            econ_preds = predictions['economic']
            econ_targets = {
                'market_sentiment': torch.tanh(torch.randn_like(task_losses[0])),  # Mock targets
                'sector_performance': torch.sigmoid(torch.randn_like(task_losses[0])),
                'economic_health': torch.sigmoid(torch.randn_like(task_losses[0]))
            }
            
            econ_loss = self.economic_loss(econ_preds, econ_targets)
            losses['economic'] = econ_loss
            task_losses.append(econ_loss)
        
        # Adaptive multi-task weighting using uncertainty
        if len(task_losses) > 0:
            weighted_losses = []
            for i, task_loss in enumerate(task_losses):
                if i < len(self.log_vars):
                    # Uncertainty-based weighting: 1/(2*sigma^2) * loss + log(sigma)
                    precision = torch.exp(-self.log_vars[i])
                    weighted_loss = precision * task_loss + self.log_vars[i]
                    weighted_losses.append(weighted_loss)
                else:
                    weighted_losses.append(task_loss)
            
            total_loss = sum(weighted_losses)
        else:
            total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Add regularization
        if self.config.l1_weight > 0 or self.config.l2_weight > 0:
            reg_loss = self._regularization_loss()
            total_loss = total_loss + reg_loss
            losses['regularization'] = reg_loss
        
        losses['total'] = total_loss
        losses['task_weights'] = torch.exp(-self.log_vars)  # Inverse of uncertainty
        
        return losses
    
    def _regularization_loss(self) -> torch.Tensor:
        """Calculate L1 and L2 regularization losses"""
        l1_loss = torch.tensor(0.0)
        l2_loss = torch.tensor(0.0)
        
        # This would be applied to model parameters in practice
        # For now, just return zero
        return l1_loss * self.config.l1_weight + l2_loss * self.config.l2_weight


class SharpeRatioLoss(nn.Module):
    """Loss function based on Sharpe ratio for financial relevance"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        super().__init__()
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
    def forward(self, predicted_returns: torch.Tensor, 
                actual_returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio as loss
        
        Args:
            predicted_returns: Predicted returns [batch, sequence]
            actual_returns: Actual returns [batch, sequence]
        """
        # Calculate returns based on predictions (simplified)
        # In practice, this would involve a trading strategy
        strategy_returns = predicted_returns.sign() * actual_returns
        
        # Calculate Sharpe ratio
        excess_returns = strategy_returns.mean(dim=1) - self.risk_free_rate
        return_std = strategy_returns.std(dim=1) + 1e-6
        
        sharpe_ratio = excess_returns / return_std
        
        # Return negative Sharpe ratio as loss (we want to maximize Sharpe)
        return -sharpe_ratio.mean()


class RankingLoss(nn.Module):
    """Ranking loss for relative performance prediction"""
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate ranking loss to preserve relative ordering
        
        Args:
            predictions: Predicted returns [batch]
            targets: Actual returns [batch]
        """
        batch_size = predictions.size(0)
        
        # Create pairwise comparisons
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [batch, batch]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)       # [batch, batch]
        
        # Calculate ranking loss
        target_sign = torch.sign(target_diff)
        ranking_loss = torch.clamp(self.margin - target_sign * pred_diff, min=0)
        
        # Only consider off-diagonal elements (different pairs)
        mask = torch.eye(batch_size, device=predictions.device) == 0
        ranking_loss = ranking_loss[mask].mean()
        
        return ranking_loss


class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty estimation"""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantile loss
        
        Args:
            predictions: Predicted quantiles [batch, n_quantiles]
            targets: Target values [batch]
        """
        total_loss = 0.0
        
        for i, quantile in enumerate(self.quantiles):
            pred_quantile = predictions[:, i]
            errors = targets - pred_quantile
            
            quantile_loss = torch.max(
                quantile * errors,
                (quantile - 1) * errors
            )
            
            total_loss += quantile_loss.mean()
        
        return total_loss / len(self.quantiles)