"""
Financial Metrics for Model Evaluation
======================================

Comprehensive metrics for evaluating stock prediction models including:
- Return prediction accuracy (MSE, MAE, R²)
- Direction accuracy and classification metrics
- Financial metrics (Sharpe ratio, Maximum Drawdown, etc.)
- Risk-adjusted returns
- Portfolio performance metrics
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation"""
    # Risk-free rate for Sharpe ratio calculation
    risk_free_rate: float = 0.02  # Annual risk-free rate
    
    # Portfolio construction parameters
    portfolio_size: int = 10  # Number of stocks in portfolio
    rebalance_frequency: int = 30  # Days between rebalancing
    
    # Trading costs
    transaction_cost: float = 0.001  # 0.1% transaction cost
    
    # Prediction horizons for evaluation
    horizons: List[str] = None
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = ['horizon_30', 'horizon_180', 'horizon_365', 'horizon_730']


class FinancialMetrics:
    """Comprehensive financial metrics calculator"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.daily_risk_free_rate = config.risk_free_rate / 252  # Convert to daily
        
    def calculate_batch_metrics(self, predictions: Dict[str, Any], 
                               targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate metrics for a single batch"""
        metrics = {}
        
        # Return prediction metrics
        if 'returns' in predictions:
            return_metrics = self._calculate_return_metrics(
                predictions['returns'], targets
            )
            metrics.update(return_metrics)
        
        # Direction accuracy metrics  
        if 'direction' in predictions:
            direction_metrics = self._calculate_direction_metrics(
                predictions['direction'], targets
            )
            metrics.update(direction_metrics)
        
        # Volatility prediction metrics
        if 'volatility' in predictions:
            vol_metrics = self._calculate_volatility_metrics(
                predictions['volatility'], targets
            )
            metrics.update(vol_metrics)
        
        return metrics
    
    def _calculate_return_metrics(self, return_predictions: Dict[str, torch.Tensor], 
                                 targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate return prediction metrics"""
        metrics = {}
        
        if 'predictions' in return_predictions:
            preds = return_predictions['predictions']
            
            for horizon in self.config.horizons:
                if horizon in preds and horizon in targets:
                    pred = preds[horizon].detach().cpu().numpy()
                    target = targets[horizon].detach().cpu().numpy()
                    
                    # Basic regression metrics
                    mse = np.mean((pred - target) ** 2)
                    mae = np.mean(np.abs(pred - target))
                    
                    # R-squared
                    ss_res = np.sum((target - pred) ** 2)
                    ss_tot = np.sum((target - np.mean(target)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    
                    # Information Coefficient (IC)
                    ic = np.corrcoef(pred, target)[0, 1] if len(pred) > 1 else 0.0
                    
                    metrics.update({
                        f'{horizon}_mse': mse,
                        f'{horizon}_mae': mae,
                        f'{horizon}_r2': r2,
                        f'{horizon}_ic': ic if not np.isnan(ic) else 0.0
                    })
        
        return metrics
    
    def _calculate_direction_metrics(self, direction_predictions: Dict[str, Dict[str, torch.Tensor]], 
                                   targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate direction prediction metrics"""
        metrics = {}
        
        for horizon in self.config.horizons:
            if horizon in direction_predictions and horizon in targets:
                logits = direction_predictions[horizon]['logits'].detach().cpu().numpy()
                probs = direction_predictions[horizon]['probabilities'].detach().cpu().numpy()
                target_returns = targets[horizon].detach().cpu().numpy()
                
                # Convert returns to direction classes
                # 0: down (< -0.01), 1: flat (-0.01 to 0.01), 2: up (> 0.01)
                direction_targets = np.zeros_like(target_returns, dtype=int)
                direction_targets[target_returns > 0.01] = 2   # Up
                direction_targets[target_returns < -0.01] = 0  # Down
                direction_targets[(target_returns >= -0.01) & (target_returns <= 0.01)] = 1  # Flat
                
                # Predicted classes
                predicted_classes = np.argmax(probs, axis=1)
                
                # Calculate classification metrics
                accuracy = accuracy_score(direction_targets, predicted_classes)
                
                # Only calculate precision/recall/f1 if we have multiple classes
                if len(np.unique(direction_targets)) > 1:
                    precision = precision_score(direction_targets, predicted_classes, average='weighted', zero_division=0)
                    recall = recall_score(direction_targets, predicted_classes, average='weighted', zero_division=0)
                    f1 = f1_score(direction_targets, predicted_classes, average='weighted', zero_division=0)
                else:
                    precision = recall = f1 = 0.0
                
                metrics.update({
                    f'{horizon}_direction_accuracy': accuracy,
                    f'{horizon}_direction_precision': precision,
                    f'{horizon}_direction_recall': recall,
                    f'{horizon}_direction_f1': f1
                })
        
        return metrics
    
    def _calculate_volatility_metrics(self, volatility_predictions: Dict[str, torch.Tensor], 
                                    targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate volatility prediction metrics"""
        metrics = {}
        
        for horizon in self.config.horizons:
            if horizon in volatility_predictions and horizon in targets:
                pred_vol = volatility_predictions[horizon].detach().cpu().numpy()
                target_returns = targets[horizon].detach().cpu().numpy()
                
                # Calculate actual volatility (absolute returns as proxy)
                actual_vol = np.abs(target_returns)
                
                # Volatility prediction metrics
                vol_mse = np.mean((pred_vol - actual_vol) ** 2)
                vol_mae = np.mean(np.abs(pred_vol - actual_vol))
                
                metrics.update({
                    f'{horizon}_vol_mse': vol_mse,
                    f'{horizon}_vol_mae': vol_mae
                })
        
        return metrics
    
    def calculate_portfolio_metrics(self, predictions: np.ndarray, 
                                  returns: np.ndarray, 
                                  timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate portfolio-level performance metrics
        
        Args:
            predictions: Predicted returns [n_samples, n_stocks]
            returns: Actual returns [n_samples, n_stocks]
            timestamps: Optional timestamps for analysis
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Construct portfolio based on predictions
        portfolio_returns = self._construct_portfolio(predictions, returns)
        
        if len(portfolio_returns) == 0:
            return {}
        
        # Calculate performance metrics
        metrics = {}
        
        # Basic return metrics
        total_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        
        # Risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = portfolio_returns - self.daily_risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-8)
        
        # Win rate
        win_rate = np.mean(portfolio_returns > 0)
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / (downside_deviation + 1e-8)
        
        metrics.update({
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'sortino_ratio': sortino_ratio
        })
        
        return metrics
    
    def _construct_portfolio(self, predictions: np.ndarray, 
                           returns: np.ndarray) -> np.ndarray:
        """
        Construct portfolio based on predictions
        
        Args:
            predictions: Predicted returns [n_samples, n_stocks]
            returns: Actual returns [n_samples, n_stocks]
            
        Returns:
            Portfolio returns array
        """
        n_samples, n_stocks = predictions.shape
        portfolio_returns = []
        
        for i in range(n_samples):
            if i == 0:
                continue  # Skip first day (no previous predictions)
            
            # Use previous day's predictions for today's positions
            prev_predictions = predictions[i-1]
            current_returns = returns[i]
            
            # Rank stocks by predicted returns and select top performers
            ranked_indices = np.argsort(prev_predictions)[::-1]
            top_stocks = ranked_indices[:self.config.portfolio_size]
            
            # Equal weight portfolio of top stocks
            weights = np.zeros(n_stocks)
            weights[top_stocks] = 1.0 / len(top_stocks)
            
            # Calculate portfolio return
            portfolio_return = np.sum(weights * current_returns)
            portfolio_returns.append(portfolio_return)
        
        return np.array(portfolio_returns)
    
    def calculate_information_ratio(self, portfolio_returns: np.ndarray, 
                                  benchmark_returns: np.ndarray) -> float:
        """Calculate Information Ratio vs benchmark"""
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(active_returns) * 252 / tracking_error
    
    def calculate_beta(self, portfolio_returns: np.ndarray, 
                      market_returns: np.ndarray) -> float:
        """Calculate portfolio beta vs market"""
        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 1.0
        
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def calculate_alpha(self, portfolio_returns: np.ndarray, 
                       market_returns: np.ndarray, 
                       beta: Optional[float] = None) -> float:
        """Calculate Jensen's alpha"""
        if beta is None:
            beta = self.calculate_beta(portfolio_returns, market_returns)
        
        portfolio_return = np.mean(portfolio_returns) * 252
        market_return = np.mean(market_returns) * 252
        
        alpha = portfolio_return - (self.config.risk_free_rate + 
                                   beta * (market_return - self.config.risk_free_rate))
        
        return alpha
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def calculate_hit_rate(self, predictions: np.ndarray, 
                          actual_returns: np.ndarray, 
                          threshold: float = 0.0) -> float:
        """Calculate hit rate (correct directional predictions)"""
        pred_direction = (predictions > threshold).astype(int)
        actual_direction = (actual_returns > threshold).astype(int)
        
        return np.mean(pred_direction == actual_direction)
    
    def generate_performance_report(self, predictions: Dict[str, np.ndarray], 
                                  targets: Dict[str, np.ndarray],
                                  timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'summary': {},
            'by_horizon': {},
            'portfolio_metrics': {},
            'risk_metrics': {}
        }
        
        # Calculate metrics for each horizon
        for horizon in self.config.horizons:
            if horizon in predictions and horizon in targets:
                horizon_pred = predictions[horizon]
                horizon_target = targets[horizon]
                
                # Basic metrics
                mse = np.mean((horizon_pred - horizon_target) ** 2)
                mae = np.mean(np.abs(horizon_pred - horizon_target))
                ic = np.corrcoef(horizon_pred, horizon_target)[0, 1]
                hit_rate = self.calculate_hit_rate(horizon_pred, horizon_target)
                
                report['by_horizon'][horizon] = {
                    'mse': mse,
                    'mae': mae,
                    'information_coefficient': ic if not np.isnan(ic) else 0.0,
                    'hit_rate': hit_rate
                }
                
                # Portfolio metrics if we have enough data
                if len(horizon_pred.shape) > 1:
                    portfolio_metrics = self.calculate_portfolio_metrics(
                        horizon_pred, horizon_target, timestamps
                    )
                    report['portfolio_metrics'][horizon] = portfolio_metrics
        
        # Overall summary
        if report['by_horizon']:
            all_ics = [metrics['information_coefficient'] 
                      for metrics in report['by_horizon'].values() 
                      if not np.isnan(metrics['information_coefficient'])]
            
            report['summary'] = {
                'average_ic': np.mean(all_ics) if all_ics else 0.0,
                'ic_std': np.std(all_ics) if all_ics else 0.0,
                'num_horizons': len(report['by_horizon'])
            }
        
        return report