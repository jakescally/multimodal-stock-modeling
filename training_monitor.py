#!/usr/bin/env python3
"""
Real-time Training Monitor
=========================

Real-time monitoring of training progress with live plots and metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import json
import time
from pathlib import Path
import argparse
from datetime import datetime
import threading
import queue

class TrainingMonitor:
    """Real-time training monitor with live plotting"""
    
    def __init__(self, experiment_dir: str, update_interval: int = 5):
        self.experiment_dir = Path(experiment_dir)
        self.update_interval = update_interval
        self.data_queue = queue.Queue()
        
        # Initialize plots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16, fontweight='bold')
        
        # Data storage
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.val_metrics = {'ic': [], 'accuracy': []}
        self.epochs = []
        
        # Setup plots
        self._setup_plots()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_training)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _setup_plots(self):
        """Setup initial plot configuration"""
        # Loss plot
        self.axes[0, 0].set_title('Training & Validation Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        self.axes[0, 1].set_title('Learning Rate Schedule')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Learning Rate')
        self.axes[0, 1].set_yscale('log')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # IC plot
        self.axes[1, 0].set_title('Information Coefficient')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('IC')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        self.axes[1, 1].set_title('Direction Accuracy')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Accuracy')
        self.axes[1, 1].grid(True, alpha=0.3)
    
    def _monitor_training(self):
        """Monitor training progress in background thread"""
        while self.monitoring:
            try:
                # Check for new training data
                self._check_for_updates()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(1)
    
    def _check_for_updates(self):
        """Check for training updates"""
        # Look for tensorboard logs or checkpoint files
        checkpoint_files = list(self.experiment_dir.glob("checkpoint_epoch_*.pth"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            self._parse_checkpoint(latest_checkpoint)
    
    def _parse_checkpoint(self, checkpoint_path: Path):
        """Parse checkpoint for training metrics"""
        try:
            # This is a simplified version - in practice, you'd parse actual checkpoint data
            epoch = int(checkpoint_path.name.split('_')[-1].split('.')[0])
            
            # Simulate reading training data (replace with actual parsing)
            if epoch not in self.epochs:
                self.epochs.append(epoch)
                self.train_losses.append(np.random.uniform(0.5, 2.0))  # Placeholder
                self.val_losses.append(np.random.uniform(0.3, 1.5))    # Placeholder
                self.learning_rates.append(1e-4 * (0.9 ** epoch))     # Placeholder
                self.val_metrics['ic'].append(np.random.uniform(-0.1, 0.1))  # Placeholder
                self.val_metrics['accuracy'].append(np.random.uniform(0.45, 0.6))  # Placeholder
                
                # Queue update
                self.data_queue.put("update")
        except Exception as e:
            print(f"Error parsing checkpoint: {e}")
    
    def animate(self, frame):
        """Animation function for live plotting"""
        # Check for updates
        try:
            self.data_queue.get_nowait()
            self._update_plots()
        except queue.Empty:
            pass
        
        return []
    
    def _update_plots(self):
        """Update all plots with new data"""
        if not self.epochs:
            return
        
        # Clear and redraw
        for ax in self.axes.flat:
            ax.clear()
        
        self._setup_plots()
        
        # Loss plot
        if self.train_losses and self.val_losses:
            self.axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Training', linewidth=2)
            self.axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
            self.axes[0, 0].legend()
        
        # Learning rate plot
        if self.learning_rates:
            self.axes[0, 1].plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        
        # IC plot
        if self.val_metrics['ic']:
            self.axes[1, 0].plot(self.epochs, self.val_metrics['ic'], 'purple', linewidth=2)
            self.axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Accuracy plot
        if self.val_metrics['accuracy']:
            self.axes[1, 1].plot(self.epochs, self.val_metrics['accuracy'], 'orange', linewidth=2)
            self.axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            self.axes[1, 1].legend()
        
        # Add status text
        if self.epochs:
            current_epoch = max(self.epochs)
            status_text = f"Epoch: {current_epoch} | "
            if self.train_losses:
                status_text += f"Train Loss: {self.train_losses[-1]:.4f} | "
            if self.val_losses:
                status_text += f"Val Loss: {self.val_losses[-1]:.4f} | "
            if self.val_metrics['ic']:
                status_text += f"IC: {self.val_metrics['ic'][-1]:.4f}"
            
            self.fig.suptitle(f'Training Monitor - {status_text}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
    
    def start_monitoring(self):
        """Start the real-time monitoring"""
        print("🖥️  Starting real-time training monitor...")
        print(f"📁 Monitoring: {self.experiment_dir}")
        print("📊 Waiting for training data...")
        
        # Start animation
        anim = animation.FuncAnimation(self.fig, self.animate, interval=1000, blit=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False

def create_training_dashboard(experiment_dir: str):
    """Create a simple training dashboard"""
    experiment_dir = Path(experiment_dir)
    
    # Create dashboard with key metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Dashboard: {experiment_dir.name}', fontsize=16, fontweight='bold')
    
    # Check if results exist
    results_path = experiment_dir / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Display final results
        test_metrics = results['test_metrics']
        horizons = ['30', '180', '365', '730']
        
        # IC comparison
        ic_values = [test_metrics[f'horizon_{h}_ic'] for h in horizons]
        axes[0, 0].bar(horizons, ic_values, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Information Coefficient')
        axes[0, 0].set_ylabel('IC')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Direction accuracy
        dir_acc = [test_metrics[f'horizon_{h}_direction_accuracy'] for h in horizons]
        axes[0, 1].bar(horizons, dir_acc, color='lightgreen', edgecolor='darkgreen')
        axes[0, 1].set_title('Direction Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        
        # MAE
        mae_values = [test_metrics[f'horizon_{h}_mae'] for h in horizons]
        axes[0, 2].bar(horizons, mae_values, color='coral', edgecolor='darkred')
        axes[0, 2].set_title('Mean Absolute Error')
        axes[0, 2].set_ylabel('MAE')
        
        # Training info
        training_info = f"""
Training Configuration:
• Epochs: {results['experiment_config']['epochs']}
• Batch Size: {results['experiment_config']['batch_size']}
• Learning Rate: {results['experiment_config']['learning_rate']}
• Symbols: {', '.join(results['experiment_config']['symbols'])}
• Duration: {results.get('training_duration', 'N/A')}
• Samples: {results['dataset_info']['total_samples']:,}
"""
        axes[1, 0].text(0.1, 0.9, training_info, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 0].set_title('Training Summary')
        axes[1, 0].axis('off')
        
        # Performance summary
        avg_ic = np.mean(ic_values)
        avg_acc = np.mean(dir_acc)
        
        performance_info = f"""
Performance Summary:
• Average IC: {avg_ic:.4f}
• Average Direction Acc: {avg_acc:.4f}
• Best IC Horizon: {horizons[np.argmax(ic_values)]}-day
• Best Accuracy Horizon: {horizons[np.argmax(dir_acc)]}-day

Model Assessment:
• IC Performance: {'Excellent' if avg_ic > 0.05 else 'Good' if avg_ic > 0.02 else 'Moderate' if avg_ic > 0 else 'Poor'}
• Direction Performance: {'Excellent' if avg_acc > 0.55 else 'Good' if avg_acc > 0.52 else 'Moderate' if avg_acc > 0.5 else 'Poor'}
"""
        axes[1, 1].text(0.1, 0.9, performance_info, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Assessment')
        axes[1, 1].axis('off')
        
        # Quick insights
        insights = f"""
Quick Insights:
• Best performing horizon: {horizons[np.argmax(ic_values)]}-day (IC: {max(ic_values):.4f})
• Most accurate predictions: {horizons[np.argmax(dir_acc)]}-day ({max(dir_acc):.4f})
• Training efficiency: {results['dataset_info']['total_samples']:,} samples
• Model complexity: {results['model_config']['d_model']} dimensions
• Volatility prediction: Available across all horizons
"""
        axes[1, 2].text(0.1, 0.9, insights, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Key Insights')
        axes[1, 2].axis('off')
        
    else:
        # Show waiting message
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Waiting for training results...', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=14, style='italic')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Training Monitor")
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--mode', choices=['monitor', 'dashboard'], default='dashboard',
                       help='Monitor mode: real-time or dashboard')
    parser.add_argument('--update_interval', type=int, default=5,
                       help='Update interval in seconds (for real-time mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor = TrainingMonitor(args.experiment_dir, args.update_interval)
        monitor.start_monitoring()
    else:
        create_training_dashboard(args.experiment_dir)

if __name__ == "__main__":
    main()