#!/usr/bin/env python3
"""
Quick Model Analysis
===================

Simple script for quick model performance analysis and visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import argparse

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def quick_analysis(experiment_dir: str):
    """Generate quick analysis of model performance"""
    experiment_dir = Path(experiment_dir)
    
    # Load results
    results_path = experiment_dir / "results.json"
    if not results_path.exists():
        print(f"❌ Results not found at {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    test_metrics = results['test_metrics']
    
    # Print summary
    print("📊 Model Performance Summary")
    print("=" * 50)
    print(f"Experiment: {experiment_dir.name}")
    print(f"Training Duration: {results.get('training_duration', 'N/A')}")
    print(f"Symbols: {', '.join(results['experiment_config']['symbols'])}")
    print(f"Total Samples: {results['dataset_info']['total_samples']:,}")
    print()
    
    # Performance by horizon
    horizons = ['30', '180', '365', '730']
    print("📈 Performance by Horizon:")
    print("-" * 50)
    print(f"{'Horizon':<10} {'IC':<10} {'Dir Acc':<10} {'MAE':<10} {'MSE':<10}")
    print("-" * 50)
    
    for horizon in horizons:
        ic = test_metrics[f'horizon_{horizon}_ic']
        dir_acc = test_metrics[f'horizon_{horizon}_direction_accuracy']
        mae = test_metrics[f'horizon_{horizon}_mae']
        mse = test_metrics[f'horizon_{horizon}_mse']
        
        print(f"{horizon + ' days':<10} {float(ic):<10.4f} {float(dir_acc):<10.4f} {float(mae):<10.4f} {float(mse):<10.4f}")
    
    # Create quick visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Quick Analysis: {experiment_dir.name}', fontsize=14, fontweight='bold')
    
    # Information Coefficient
    ic_values = [test_metrics[f'horizon_{h}_ic'] for h in horizons]
    axes[0, 0].bar(horizons, ic_values, color='skyblue', edgecolor='navy')
    axes[0, 0].set_title('Information Coefficient')
    axes[0, 0].set_xlabel('Horizon (days)')
    axes[0, 0].set_ylabel('IC')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Direction Accuracy
    dir_acc_values = [test_metrics[f'horizon_{h}_direction_accuracy'] for h in horizons]
    axes[0, 1].bar(horizons, dir_acc_values, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_title('Direction Accuracy')
    axes[0, 1].set_xlabel('Horizon (days)')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean Absolute Error
    mae_values = [test_metrics[f'horizon_{h}_mae'] for h in horizons]
    axes[1, 0].bar(horizons, mae_values, color='coral', edgecolor='darkred')
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].set_xlabel('Horizon (days)')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    f1_values = [test_metrics[f'horizon_{h}_direction_f1'] for h in horizons]
    axes[1, 1].bar(horizons, f1_values, color='gold', edgecolor='orange')
    axes[1, 1].set_title('Direction F1 Score')
    axes[1, 1].set_xlabel('Horizon (days)')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance insights
    print("\n🔍 Performance Insights:")
    print("-" * 30)
    
    # Best performing horizon
    best_ic_horizon = horizons[np.argmax(ic_values)]
    best_acc_horizon = horizons[np.argmax(dir_acc_values)]
    
    print(f"• Best IC: {best_ic_horizon}-day horizon ({max(ic_values):.4f})")
    print(f"• Best Direction Accuracy: {best_acc_horizon}-day horizon ({max(dir_acc_values):.4f})")
    
    # Performance assessment
    avg_ic = np.mean(ic_values)
    avg_acc = np.mean(dir_acc_values)
    
    if avg_ic > 0.05:
        ic_assessment = "Excellent"
    elif avg_ic > 0.02:
        ic_assessment = "Good"
    elif avg_ic > 0:
        ic_assessment = "Moderate"
    else:
        ic_assessment = "Poor"
    
    if avg_acc > 0.55:
        acc_assessment = "Excellent"
    elif avg_acc > 0.52:
        acc_assessment = "Good"
    elif avg_acc > 0.5:
        acc_assessment = "Moderate"
    else:
        acc_assessment = "Poor"
    
    print(f"• Overall IC Performance: {ic_assessment} (avg: {avg_ic:.4f})")
    print(f"• Overall Direction Performance: {acc_assessment} (avg: {avg_acc:.4f})")
    
    # Training efficiency
    if 'training_duration' in results:
        duration = results['training_duration']
        samples = results['dataset_info']['total_samples']
        print(f"• Training Efficiency: {samples:,} samples in {duration}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Quick Model Analysis")
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    
    args = parser.parse_args()
    quick_analysis(args.experiment_dir)

if __name__ == "__main__":
    main()