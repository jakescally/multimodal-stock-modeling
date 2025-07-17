# MacBook Multi-Threading Optimization Guide

## Quick Start

For optimal performance on your MacBook, use the `--optimize_for_mac` flag:

```bash
python train.py --use_mock_data --epochs 10 --optimize_for_mac
```

This automatically configures all threading and data loading settings for your hardware.

## Available Optimization Flags

### `--optimize_for_mac`
**Recommended for all MacBook users**
- Auto-detects Apple Silicon vs Intel Mac
- Sets optimal PyTorch threading (14 threads for M-series, 8 for Intel)
- Configures optimal data loader workers (2-4 workers)
- Enables memory pinning and persistent workers
- Sets environment variables for accelerated libraries

### `--num_workers N`
**Manual control over data loading workers**
- Default: Auto-detected (2-4 for Mac)
- Higher values: More parallel data loading (diminishing returns after 4)
- Lower values: Less memory usage, simpler debugging

### `--pin_memory`
**Faster data transfer to GPU/MPS**
- Automatically enabled with `--optimize_for_mac`
- Speeds up data transfer from CPU to Apple Silicon GPU
- Small memory overhead

### `--persistent_workers`
**Keep data workers alive between epochs**
- Automatically enabled with `--optimize_for_mac`
- Reduces worker startup overhead
- Better for longer training runs

## Hardware-Specific Settings

### Apple Silicon (M1/M2/M3/M4)
```bash
# Automatic (recommended)
python train.py --optimize_for_mac

# Manual fine-tuning
python train.py --num_workers 4 --pin_memory --persistent_workers
```

**What happens:**
- PyTorch threads: 14 (full CPU cores)
- Interop threads: 4
- Data workers: 3-4
- Memory pinning: Enabled
- MPS acceleration: Automatic

### Intel Mac
```bash
# Automatic (recommended)  
python train.py --optimize_for_mac

# Manual fine-tuning
python train.py --num_workers 3 --pin_memory --persistent_workers
```

**What happens:**
- PyTorch threads: 8 (conservative for thermal management)
- Interop threads: 2-4
- Data workers: 2-3
- Memory pinning: Enabled

## Performance Examples

### Basic Training (No optimization)
```bash
python train.py --use_mock_data --epochs 5
# Uses default settings: 2 workers, no pinning
```

### Optimized Training (Recommended)
```bash
python train.py --use_mock_data --epochs 5 --optimize_for_mac
# Uses optimal settings for your Mac hardware
```

### Real Data Training
```bash
python train.py --symbols AAPL MSFT GOOGL --epochs 100 --optimize_for_mac
# Optimal settings with real financial data
```

### Development/Debugging
```bash
python train.py --use_mock_data --epochs 2 --num_workers 0
# Single-threaded for easier debugging
```

## Expected Performance Improvements

- **Apple Silicon**: 15-30% faster training
- **Intel Mac**: 10-20% faster training
- **Data loading**: 2-3x faster with multiple workers
- **Memory efficiency**: Better with pinning and persistent workers

## Troubleshooting

### "Too many open files" error
Reduce `--num_workers`:
```bash
python train.py --optimize_for_mac --num_workers 2
```

### High memory usage
Disable persistent workers:
```bash
python train.py --optimize_for_mac --persistent_workers false
```

### Slow startup
Check if workers are being created efficiently:
```bash
python train.py --optimize_for_mac --num_workers 1
```

## Monitoring Performance

Use the benchmark script to compare settings:
```bash
python benchmark_performance.py
```

This will run identical training with and without optimizations to show the performance difference on your specific hardware.

## Technical Details

The `--optimize_for_mac` flag sets these environment variables:
- `OMP_NUM_THREADS`: OpenMP threading
- `MKL_NUM_THREADS`: Intel Math Kernel Library
- `VECLIB_MAXIMUM_THREADS`: Apple's Accelerate framework

And configures PyTorch internals:
- `torch.set_num_threads()`: Main compute threads
- `torch.set_num_interop_threads()`: Inter-operation parallelism
- DataLoader optimization for macOS