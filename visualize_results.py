#!/usr/bin/env python3
"""
Model Performance Visualization Tools
====================================

Comprehensive visualization suite for analyzing multimodal stock prediction model performance.
Includes training monitoring, prediction analysis, and financial performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelPerformanceVisualizer:
    """Comprehensive visualization suite for model performance analysis"""
    
    def __init__(self, experiment_dir: str, figsize: Tuple[int, int] = (15, 10)):
        self.experiment_dir = Path(experiment_dir)
        self.figsize = figsize
        self.results = self._load_results()
        self.predictions = self._load_predictions()
        
    def _load_results(self) -> Dict:
        """Load experiment results"""
        results_path = self.experiment_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found at {results_path}")
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def _load_predictions(self) -> Optional[Dict]:
        """Load model predictions if available"""
        pred_path = self.experiment_dir / "test_predictions.pth"
        if pred_path.exists():
            return torch.load(pred_path, map_location='cpu')
        return None
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves"""
        if 'training_history' not in self.results:
            print("No training history found in results")
            return
        
        history = self.results['training_history']
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Training and validation loss
        axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        if 'learning_rate' in history:
            axes[0, 1].plot(history['learning_rate'], color='red', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Validation metrics over time
        if 'val_metrics' in history:
            val_metrics = history['val_metrics']
            if 'horizon_30_ic' in val_metrics:
                axes[1, 0].plot(val_metrics['horizon_30_ic'], label='30-day IC', linewidth=2)
                axes[1, 0].plot(val_metrics['horizon_180_ic'], label='180-day IC', linewidth=2)
                axes[1, 0].set_title('Information Coefficient Over Time')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Information Coefficient')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Direction accuracy
        if 'val_metrics' in history:
            val_metrics = history['val_metrics']
            if 'horizon_30_direction_accuracy' in val_metrics:
                axes[1, 1].plot(val_metrics['horizon_30_direction_accuracy'], label='30-day Accuracy', linewidth=2)
                axes[1, 1].plot(val_metrics['horizon_180_direction_accuracy'], label='180-day Accuracy', linewidth=2)
                axes[1, 1].set_title('Direction Accuracy Over Time')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, save_path: Optional[str] = None):
        """Plot comprehensive performance metrics"""
        test_metrics = self.results['test_metrics']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        horizons = ['30', '180', '365', '730']
        
        # MSE across horizons
        mse_values = [test_metrics[f'horizon_{h}_mse'] for h in horizons]
        axes[0, 0].bar(horizons, mse_values, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Mean Squared Error by Horizon')
        axes[0, 0].set_xlabel('Prediction Horizon (days)')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE across horizons
        mae_values = [test_metrics[f'horizon_{h}_mae'] for h in horizons]
        axes[0, 1].bar(horizons, mae_values, color='lightgreen', edgecolor='darkgreen')
        axes[0, 1].set_title('Mean Absolute Error by Horizon')
        axes[0, 1].set_xlabel('Prediction Horizon (days)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Information Coefficient
        ic_values = [test_metrics[f'horizon_{h}_ic'] for h in horizons]
        axes[0, 2].bar(horizons, ic_values, color='coral', edgecolor='darkred')
        axes[0, 2].set_title('Information Coefficient by Horizon')
        axes[0, 2].set_xlabel('Prediction Horizon (days)')
        axes[0, 2].set_ylabel('IC')
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Direction Accuracy
        dir_acc_values = [test_metrics[f'horizon_{h}_direction_accuracy'] for h in horizons]
        axes[1, 0].bar(horizons, dir_acc_values, color='gold', edgecolor='orange')
        axes[1, 0].set_title('Direction Accuracy by Horizon')
        axes[1, 0].set_xlabel('Prediction Horizon (days)')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        f1_values = [test_metrics[f'horizon_{h}_direction_f1'] for h in horizons]
        axes[1, 1].bar(horizons, f1_values, color='plum', edgecolor='purple')
        axes[1, 1].set_title('Direction F1 Score by Horizon')
        axes[1, 1].set_xlabel('Prediction Horizon (days)')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Volatility Prediction
        vol_mae_values = [test_metrics[f'horizon_{h}_vol_mae'] for h in horizons]
        axes[1, 2].bar(horizons, vol_mae_values, color='lightcoral', edgecolor='darkred')
        axes[1, 2].set_title('Volatility Prediction MAE by Horizon')
        axes[1, 2].set_xlabel('Prediction Horizon (days)')
        axes[1, 2].set_ylabel('Volatility MAE')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_analysis(self, save_path: Optional[str] = None):
        """Plot prediction vs actual analysis"""
        if self.predictions is None:
            print("No predictions available. Run training with --save_predictions flag.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Extract predictions and targets for 30-day horizon
        predictions = self.predictions['predictions']
        targets = self.predictions['targets']
        
        # Scatter plot: Predicted vs Actual (30-day)
        if 'horizon_30' in predictions and 'horizon_30' in targets:
            pred_30 = predictions['horizon_30'].numpy().flatten()
            actual_30 = targets['horizon_30'].numpy().flatten()
            
            axes[0, 0].scatter(actual_30, pred_30, alpha=0.6, s=20)
            axes[0, 0].plot([actual_30.min(), actual_30.max()], 
                          [actual_30.min(), actual_30.max()], 
                          'r--', lw=2, label='Perfect Prediction')
            axes[0, 0].set_title('30-Day Predictions vs Actual')
            axes[0, 0].set_xlabel('Actual Returns')
            axes[0, 0].set_ylabel('Predicted Returns')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        if 'horizon_30' in predictions and 'horizon_30' in targets:
            residuals = pred_30 - actual_30
            axes[0, 1].scatter(pred_30, residuals, alpha=0.6, s=20)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_title('Residual Plot (30-Day)')
            axes[0, 1].set_xlabel('Predicted Returns')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        if 'horizon_30' in predictions and 'horizon_30' in targets:
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Error Distribution (30-Day)')
            axes[1, 0].set_xlabel('Prediction Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative returns comparison
        if 'horizon_30' in predictions and 'horizon_30' in targets:
            cum_actual = np.cumprod(1 + actual_30)
            cum_predicted = np.cumprod(1 + pred_30)
            
            axes[1, 1].plot(cum_actual, label='Actual Cumulative Returns', linewidth=2)
            axes[1, 1].plot(cum_predicted, label='Predicted Cumulative Returns', linewidth=2)
            axes[1, 1].set_title('Cumulative Returns Comparison')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Cumulative Returns')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_financial_performance(self, save_path: Optional[str] = None):
        """Plot financial performance metrics"""
        if self.predictions is None:
            print("No predictions available for financial analysis.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Financial Performance Analysis', fontsize=16, fontweight='bold')
        
        predictions = self.predictions['predictions']
        targets = self.predictions['targets']
        
        # Calculate financial metrics for each horizon
        horizons = ['30', '180', '365', '730']
        sharpe_ratios = []
        max_drawdowns = []
        
        for horizon in horizons:
            if f'horizon_{horizon}' in predictions and f'horizon_{horizon}' in targets:
                pred = predictions[f'horizon_{horizon}'].numpy().flatten()
                actual = targets[f'horizon_{horizon}'].numpy().flatten()
                
                # Calculate Sharpe ratio (simplified)
                if np.std(pred) > 0:
                    sharpe = np.mean(pred) / np.std(pred) * np.sqrt(252)  # Annualized
                else:
                    sharpe = 0
                sharpe_ratios.append(sharpe)
                
                # Calculate max drawdown
                cum_returns = np.cumprod(1 + pred)
                running_max = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns - running_max) / running_max
                max_dd = np.min(drawdown)
                max_drawdowns.append(abs(max_dd))
        
        # Sharpe ratio by horizon
        axes[0, 0].bar(horizons, sharpe_ratios, color='green', alpha=0.7, edgecolor='darkgreen')
        axes[0, 0].set_title('Sharpe Ratio by Horizon')
        axes[0, 0].set_xlabel('Prediction Horizon (days)')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Max drawdown by horizon
        axes[0, 1].bar(horizons, max_drawdowns, color='red', alpha=0.7, edgecolor='darkred')
        axes[0, 1].set_title('Maximum Drawdown by Horizon')
        axes[0, 1].set_xlabel('Prediction Horizon (days)')
        axes[0, 1].set_ylabel('Max Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Return distribution (30-day)
        if 'horizon_30' in predictions:
            pred_30 = predictions['horizon_30'].numpy().flatten()
            axes[1, 0].hist(pred_30, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Return Distribution (30-Day Predictions)')
            axes[1, 0].set_xlabel('Predicted Returns')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling performance metrics
        if 'horizon_30' in predictions and 'horizon_30' in targets:
            pred_30 = predictions['horizon_30'].numpy().flatten()
            actual_30 = targets['horizon_30'].numpy().flatten()
            
            # Calculate rolling correlation
            window = 30
            rolling_corr = []
            for i in range(window, len(pred_30)):
                corr = np.corrcoef(pred_30[i-window:i], actual_30[i-window:i])[0, 1]
                rolling_corr.append(corr if not np.isnan(corr) else 0)
            
            axes[1, 1].plot(rolling_corr, linewidth=2)
            axes[1, 1].set_title('Rolling Correlation (30-Day Window)')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Correlation')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """Plot model performance comparison with baselines"""
        test_metrics = self.results['test_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Model vs Baseline Comparison', fontsize=16, fontweight='bold')
        
        horizons = ['30', '180', '365', '730']
        
        # Information Coefficient comparison
        ic_values = [test_metrics[f'horizon_{h}_ic'] for h in horizons]
        baseline_ic = [0.0] * len(horizons)  # Random baseline
        
        x = np.arange(len(horizons))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, ic_values, width, label='Model', color='skyblue', edgecolor='navy')
        axes[0, 0].bar(x + width/2, baseline_ic, width, label='Random', color='lightcoral', edgecolor='darkred')
        axes[0, 0].set_title('Information Coefficient Comparison')
        axes[0, 0].set_xlabel('Prediction Horizon (days)')
        axes[0, 0].set_ylabel('IC')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(horizons)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Direction accuracy comparison
        dir_acc_values = [test_metrics[f'horizon_{h}_direction_accuracy'] for h in horizons]
        baseline_acc = [0.5] * len(horizons)  # Random baseline
        
        axes[0, 1].bar(x - width/2, dir_acc_values, width, label='Model', color='lightgreen', edgecolor='darkgreen')
        axes[0, 1].bar(x + width/2, baseline_acc, width, label='Random', color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_title('Direction Accuracy Comparison')
        axes[0, 1].set_xlabel('Prediction Horizon (days)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(horizons)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance summary heatmap
        metrics = ['MSE', 'MAE', 'IC', 'Direction Accuracy', 'F1 Score']
        data = []
        for horizon in horizons:
            row = [
                test_metrics[f'horizon_{horizon}_mse'],
                test_metrics[f'horizon_{horizon}_mae'],
                test_metrics[f'horizon_{horizon}_ic'],
                test_metrics[f'horizon_{horizon}_direction_accuracy'],
                test_metrics[f'horizon_{horizon}_direction_f1']
            ]
            data.append(row)
        
        # Normalize data for heatmap
        data_normalized = []
        for col in range(len(metrics)):
            col_data = [float(row[col]) for row in data]
            col_min, col_max = min(col_data), max(col_data)
            if col_max > col_min:
                normalized = [(x - col_min) / (col_max - col_min) for x in col_data]
            else:
                normalized = [0.5] * len(col_data)
            data_normalized.append(normalized)
        
        data_normalized = np.array(data_normalized).T
        
        im = axes[1, 0].imshow(data_normalized, cmap='RdYlBu_r', aspect='auto')
        axes[1, 0].set_title('Performance Heatmap (Normalized)')
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Prediction Horizon (days)')
        axes[1, 0].set_xticks(range(len(metrics)))
        axes[1, 0].set_xticklabels(metrics, rotation=45)
        axes[1, 0].set_yticks(range(len(horizons)))
        axes[1, 0].set_yticklabels(horizons)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Normalized Performance')
        
        # Training summary
        if 'training_duration' in self.results:
            training_info = f"""
Training Summary:
• Duration: {self.results['training_duration']}
• Epochs: {self.results['experiment_config']['epochs']}
• Batch Size: {self.results['experiment_config']['batch_size']}
• Learning Rate: {self.results['experiment_config']['learning_rate']}
• Symbols: {', '.join(self.results['experiment_config']['symbols'])}
• Total Samples: {self.results['dataset_info']['total_samples']}
• Model Parameters: {self.results['model_config']['d_model']} dimensions
"""
            axes[1, 1].text(0.1, 0.9, training_info, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Training Summary')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, save_dir: Optional[str] = None):
        """Generate comprehensive performance report"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 Generating Model Performance Report...")
        print("=" * 60)
        
        # Generate all plots
        plots = [
            ("training_history", "Training History"),
            ("performance_metrics", "Performance Metrics"), 
            ("prediction_analysis", "Prediction Analysis"),
            ("financial_performance", "Financial Performance"),
            ("model_comparison", "Model Comparison")
        ]
        
        for plot_name, plot_title in plots:
            print(f"📈 Generating {plot_title}...")
            save_path = save_dir / f"{plot_name}.png" if save_dir else None
            
            if hasattr(self, f'plot_{plot_name}'):
                getattr(self, f'plot_{plot_name}')(save_path)
            
        print("\n✅ Report generation complete!")
        if save_dir:
            print(f"📁 Plots saved to: {save_dir}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Visualize Model Performance")
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--save_dir', type=str, help='Directory to save plots')
    parser.add_argument('--plot', type=str, choices=[
        'training_history', 'performance_metrics', 'prediction_analysis',
        'financial_performance', 'model_comparison', 'all'
    ], default='all', help='Which plot to generate')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelPerformanceVisualizer(args.experiment_dir)
    
    # Generate requested plots
    if args.plot == 'all':
        visualizer.generate_report(args.save_dir)
    else:
        save_path = Path(args.save_dir) / f"{args.plot}.png" if args.save_dir else None
        getattr(visualizer, f'plot_{args.plot}')(save_path)

if __name__ == "__main__":
    main()