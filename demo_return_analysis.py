#!/usr/bin/env python3
"""
Demo: Return Analysis Visualization
===================================

Demonstrates the new return analysis visualization functions with simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def demo_return_analysis():
    """Demo the new return analysis visualization functions"""
    
    print("📊 Demo: Return Analysis Visualization")
    print("=" * 50)
    
    # Simulate realistic return prediction data
    np.random.seed(42)
    
    horizons = ['30', '180', '365', '730']
    n_samples = 126  # Test set size
    
    # Create realistic return data
    simulated_data = {}
    
    for horizon in horizons:
        # Simulate actual returns (more volatile for longer horizons)
        volatility = 0.02 * (1 + int(horizon) / 365)  # Scale volatility with horizon
        actual_returns = np.random.normal(0, volatility, n_samples)
        
        # Simulate predicted returns (with some correlation to actual)
        correlation = 0.7 - (int(horizon) / 1000)  # Correlation decreases with horizon
        noise = np.random.normal(0, volatility * 0.5, n_samples)
        predicted_returns = actual_returns * correlation + noise
        
        simulated_data[horizon] = {
            'actual': actual_returns,
            'predicted': predicted_returns
        }
    
    # Demo 1: Return Analysis by Horizon
    print("📈 Demo 1: Return Analysis by Horizon")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Return Prediction Analysis by Horizon (Demo)', fontsize=16, fontweight='bold')
    
    colors = ['skyblue', 'lightgreen', 'coral', 'gold']
    
    for i, horizon in enumerate(horizons):
        data = simulated_data[horizon]
        pred_pct = data['predicted'] * 100
        actual_pct = data['actual'] * 100
        
        row, col = i // 2, i % 2
        
        # Scatter plot
        axes[row, col].scatter(actual_pct, pred_pct, alpha=0.6, s=20, color=colors[i])
        
        # Perfect prediction line
        min_val = min(actual_pct.min(), pred_pct.min())
        max_val = max(actual_pct.max(), pred_pct.max())
        axes[row, col].plot([min_val, max_val], [min_val, max_val], 
                          'r--', lw=2, label='Perfect Prediction')
        
        # Calculate metrics
        correlation = np.corrcoef(actual_pct, pred_pct)[0, 1]
        mape = np.mean(np.abs(pred_pct - actual_pct))
        
        axes[row, col].set_title(f'{horizon}-Day Returns\nCorr: {correlation:.3f}, MAPE: {mape:.2f}%')
        axes[row, col].set_xlabel('Actual Returns (%)')
        axes[row, col].set_ylabel('Predicted Returns (%)')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/demo_return_analysis_by_horizon.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: /tmp/demo_return_analysis_by_horizon.png")
    
    # Demo 2: Return Accuracy Analysis
    print("📈 Demo 2: Return Accuracy Analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Return Prediction Accuracy Analysis (Demo)', fontsize=16, fontweight='bold')
    
    # Error band thresholds
    error_bands = [1, 2, 5, 10, 15, 20]
    
    # Calculate accuracy data
    accuracy_data = []
    horizon_names = []
    
    for horizon in horizons:
        data = simulated_data[horizon]
        pred_pct = data['predicted'] * 100
        actual_pct = data['actual'] * 100
        
        # Calculate absolute errors
        abs_errors = np.abs(pred_pct - actual_pct)
        
        # Calculate accuracy for different error bands
        band_accuracies = []
        for band in error_bands:
            accuracy = (abs_errors <= band).mean() * 100
            band_accuracies.append(accuracy)
        
        accuracy_data.append(band_accuracies)
        horizon_names.append(f'{horizon}-day')
    
    # Plot 1: Accuracy by error bands
    x = np.arange(len(error_bands))
    width = 0.2
    
    for i, (horizon_acc, horizon_name) in enumerate(zip(accuracy_data, horizon_names)):
        axes[0, 0].bar(x + i * width, horizon_acc, width, label=horizon_name, alpha=0.8)
    
    axes[0, 0].set_title('Prediction Accuracy by Error Tolerance')
    axes[0, 0].set_xlabel('Error Tolerance (% points)')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels([f'±{band}%' for band in error_bands])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    error_distributions = []
    for horizon in horizons:
        data = simulated_data[horizon]
        abs_errors = np.abs((data['predicted'] - data['actual']) * 100)
        error_distributions.append(abs_errors)
    
    axes[0, 1].boxplot(error_distributions, labels=[f'{h}-day' for h in horizons])
    axes[0, 1].set_title('Error Distribution by Horizon')
    axes[0, 1].set_xlabel('Prediction Horizon')
    axes[0, 1].set_ylabel('Absolute Error (% points)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MAPE by horizon
    mape_values = []
    for horizon in horizons:
        data = simulated_data[horizon]
        mape = np.mean(np.abs((data['predicted'] - data['actual']) * 100))
        mape_values.append(mape)
    
    bars = axes[1, 0].bar(horizons, mape_values, color=colors, alpha=0.8)
    axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)')
    axes[1, 0].set_xlabel('Prediction Horizon (days)')
    axes[1, 0].set_ylabel('MAPE (% points)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mape in zip(bars, mape_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mape:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Correlation and R²
    correlations = []
    r_squared_values = []
    
    for horizon in horizons:
        data = simulated_data[horizon]
        corr = np.corrcoef(data['actual'], data['predicted'])[0, 1]
        correlations.append(corr)
        r_squared_values.append(corr ** 2)
    
    x = np.arange(len(horizons))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, correlations, width, label='Correlation', alpha=0.8, color='lightblue')
    axes[1, 1].bar(x + width/2, r_squared_values, width, label='R²', alpha=0.8, color='lightcoral')
    
    axes[1, 1].set_title('Correlation and R² by Horizon')
    axes[1, 1].set_xlabel('Prediction Horizon (days)')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'{h}-day' for h in horizons])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/demo_return_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: /tmp/demo_return_accuracy_analysis.png")
    
    # Print summary
    print("\n📊 Demo Summary:")
    print("=" * 30)
    print("New visualization functions added:")
    print("1. plot_return_analysis_by_horizon() - Shows predicted vs actual returns for each horizon")
    print("2. plot_return_accuracy_analysis() - Shows accuracy bands and error distributions")
    print()
    print("Key metrics displayed:")
    print("• Correlation between predicted and actual returns")
    print("• Mean Absolute Percentage Error (MAPE)")
    print("• Accuracy within different error tolerance bands (±1%, ±2%, ±5%, etc.)")
    print("• Error distribution boxplots")
    print("• R² values for each horizon")
    print()
    print("Usage after training with --save_predictions:")
    print("python visualize_results.py checkpoints/experiment_dir --plot return_analysis_by_horizon")
    print("python visualize_results.py checkpoints/experiment_dir --plot return_accuracy_analysis")
    print("python visualize_results.py checkpoints/experiment_dir --plot all")

if __name__ == "__main__":
    demo_return_analysis()