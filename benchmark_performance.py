#!/usr/bin/env python3
"""
Performance Benchmark Script
============================

Compare training performance with and without MacBook optimizations.
"""

import subprocess
import time
import sys
import multiprocessing as mp
import torch

def get_system_info():
    """Get system information"""
    print("System Information:")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  PyTorch version: {torch.__version__}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  Device: Apple Silicon with MPS support")
    elif torch.cuda.is_available():
        print(f"  Device: CUDA ({torch.cuda.get_device_name()})")
    else:
        print("  Device: CPU only")
    
    print()

def run_benchmark(optimized=False):
    """Run a training benchmark"""
    cmd = [
        sys.executable, "train.py",
        "--use_mock_data",
        "--epochs", "2",
        "--batch_size", "32",
        "--experiment_name", f"benchmark_{'optimized' if optimized else 'default'}",
        "--learning_rate", "1e-3"
    ]
    
    if optimized:
        cmd.extend([
            "--optimize_for_mac",
            "--num_workers", "4"
        ])
    else:
        cmd.extend([
            "--num_workers", "2"
        ])
    
    print(f"Running {'optimized' if optimized else 'default'} benchmark...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.returncode == 0:
        print(f"✓ {'Optimized' if optimized else 'Default'} training completed in {duration:.2f} seconds")
        
        # Extract key metrics from output
        lines = result.stdout.split('\n')
        for line in lines:
            if "Model initialized with" in line:
                print(f"  {line.split('INFO:__main__:')[1]}")
            elif "Final settings" in line and optimized:
                print(f"  {line.split('INFO:__main__:')[1]}")
            elif "Training completed!" in line:
                break
        
        return duration
    else:
        print(f"✗ {'Optimized' if optimized else 'Default'} training failed")
        print(f"Error: {result.stderr}")
        return None

def main():
    print("="*60)
    print("MACBOOK MULTI-THREADING PERFORMANCE BENCHMARK")
    print("="*60)
    
    get_system_info()
    
    # Run benchmarks
    print("Running performance comparison...")
    print()
    
    # Default training
    default_time = run_benchmark(optimized=False)
    print()
    
    # Optimized training
    optimized_time = run_benchmark(optimized=True)
    print()
    
    # Compare results
    if default_time and optimized_time:
        improvement = (default_time - optimized_time) / default_time * 100
        speedup = default_time / optimized_time
        
        print("="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"Default training time:   {default_time:.2f} seconds")
        print(f"Optimized training time: {optimized_time:.2f} seconds")
        print(f"Speedup:                 {speedup:.2f}x")
        print(f"Improvement:             {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"\n🚀 MacBook optimization provides {improvement:.1f}% faster training!")
        else:
            print(f"\n⚠️  MacBook optimization is {-improvement:.1f}% slower (may vary by workload)")
    
    print("\nOptimization flags:")
    print("  --optimize_for_mac: Auto-detects optimal settings for Mac hardware")
    print("  --num_workers: Number of data loading workers")
    print("  --pin_memory: Faster data transfer to GPU/MPS")
    print("  --persistent_workers: Keep workers alive between epochs")

if __name__ == "__main__":
    main()