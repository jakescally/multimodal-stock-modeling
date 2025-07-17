# Multimodal Stock Prediction Model

A hybrid deep learning model that combines time series analysis with qualitative data for comprehensive stock performance prediction. This model integrates stock prices, financial news, and employment indicators to predict stock movements across multiple time horizons.

## 🚀 Features

- **Multimodal Data Integration**: Combines stock prices, financial news, and employment data
- **Real-time Data Fetching**: Automatic data retrieval from Yahoo Finance, news RSS feeds, and employment sources
- **Multi-horizon Predictions**: Predicts stock movements for 30, 180, 365, and 730 days
- **Advanced Architecture**: Uses cross-modal fusion with attention mechanisms
- **Production Ready**: Includes caching, experiment tracking, and comprehensive metrics
- **Hardware Optimized**: Special optimizations for MacBook (Apple Silicon & Intel)

## 📊 Model Architecture

```
Stock Data (OHLCV + Technical Indicators)
    ↓
Stock Encoder → Cross-Modal Fusion Layer → Multi-Task Prediction Heads
    ↑                     ↑                        ↓
Text Encoder ←→ News Data    Employment Data → Direction, Returns, Volatility
    ↓
Employment Encoder
```

### Key Components:
- **Stock Encoder**: Processes OHLCV data and technical indicators
- **Text Encoder**: Encodes financial news and sentiment
- **Employment Encoder**: Processes job market indicators
- **Cross-Modal Fusion**: Attention-based fusion of all modalities
- **Multi-Task Heads**: Separate prediction heads for different objectives

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
```bash
git clone https://github.com/your-username/multimodal-stock-modelling.git
cd multimodal-stock-modelling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### Basic Training
```bash
python train.py --symbols AAPL MSFT GOOGL --epochs 100 --optimize_for_mac
```

### Advanced Training Configuration
```bash
python train.py \
    --symbols AAPL MSFT GOOGL AMZN TSLA \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --start_date 2020-01-01 \
    --sequence_length 252 \
    --fusion_strategy cross_attention \
    --optimize_for_mac \
    --experiment_name my_experiment
```

## 📈 Data Sources

### Stock Data
- **Source**: Yahoo Finance API (yfinance)
- **Features**: OHLCV, technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Frequency**: Daily

### News Data
- **Sources**: Yahoo Finance, Reuters, MarketWatch RSS feeds
- **Processing**: Sentiment analysis and text embeddings
- **Features**: Title, summary, publication date, sentiment scores

### Employment Data
- **Source**: Federal Reserve Economic Data (FRED)
- **Features**: Unemployment rate, job openings, labor participation, etc.
- **Frequency**: Monthly

## 🔧 Configuration Options

### Training Parameters
```bash
--symbols AAPL MSFT GOOGL        # Stock symbols to train on
--epochs 100                     # Number of training epochs
--batch_size 32                  # Batch size for training
--learning_rate 0.0001           # Learning rate
--start_date 2021-01-01          # Start date for training data
--sequence_length 252            # Input sequence length (trading days)
--fusion_strategy cross_attention # Fusion strategy for multimodal data
```

### Hardware Optimization
```bash
--optimize_for_mac              # Enable MacBook-specific optimizations
--num_workers 4                 # Number of data loading workers
--pin_memory                    # Enable memory pinning for faster GPU transfer
--mixed_precision               # Use mixed precision training
```

### Experiment Tracking
```bash
--experiment_name my_experiment  # Name for experiment tracking
--checkpoint_dir checkpoints    # Directory for saving checkpoints
--save_predictions              # Save model predictions to file
```

## 📊 Model Performance

The model provides comprehensive evaluation metrics:

- **Regression Metrics**: MSE, MAE, R²
- **Classification Metrics**: Direction accuracy, precision, recall, F1-score
- **Financial Metrics**: Information Coefficient (IC), Sharpe ratio
- **Volatility Prediction**: Volatility MSE and MAE

## 📈 Visualization Tools

Three powerful visualization tools are available to analyze model performance:

### Quick Analysis
```bash
python quick_analysis.py checkpoints/experiment_20250717_135243
```
- Performance summary with key metrics
- Quick bar charts for all prediction horizons
- Performance assessment and insights

### Comprehensive Visualization
```bash
python visualize_results.py checkpoints/experiment_20250717_135243 --plot all
```
- Training history curves
- Performance metrics across horizons
- Prediction vs actual analysis
- **NEW: Return analysis by horizon** - Shows predicted vs actual returns for each time horizon
- **NEW: Return accuracy analysis** - Shows accuracy bands and error distributions
- Financial performance metrics
- Model comparison with baselines

### Training Monitor
```bash
python training_monitor.py checkpoints/experiment_20250717_135243 --mode dashboard
```
- Real-time training monitoring
- Training dashboard with key insights
- Performance assessment and recommendations

### Visualization Options

**Save plots to files:**
```bash
python visualize_results.py checkpoints/experiment_20250717_135243 --save_dir plots/
```

**Generate specific plot types:**
```bash
python visualize_results.py checkpoints/experiment_20250717_135243 --plot performance_metrics
python visualize_results.py checkpoints/experiment_20250717_135243 --plot training_history
python visualize_results.py checkpoints/experiment_20250717_135243 --plot prediction_analysis
python visualize_results.py checkpoints/experiment_20250717_135243 --plot return_analysis_by_horizon
python visualize_results.py checkpoints/experiment_20250717_135243 --plot return_accuracy_analysis
```

**Real-time monitoring during training:**
```bash
python training_monitor.py checkpoints/experiment_20250717_135243 --mode monitor
```

### 📊 Return Analysis Features

The new return analysis visualizations provide detailed insights into prediction accuracy:

**Return Analysis by Horizon:**
- Scatter plots of predicted vs actual returns for each horizon (30, 180, 365, 730 days)
- Correlation coefficients between predictions and actual returns
- Mean Absolute Percentage Error (MAPE) for each horizon
- Perfect prediction reference lines

**Return Accuracy Analysis:**
- Accuracy within different error tolerance bands (±1%, ±2%, ±5%, ±10%, ±15%, ±20%)
- Error distribution boxplots showing prediction spread
- MAPE comparison across horizons
- Correlation and R² values for each prediction horizon

**Key Metrics:**
- **MAPE**: How many percentage points off the predictions are on average
- **Correlation**: How well predictions track actual return direction and magnitude
- **Error Bands**: Percentage of predictions within specific accuracy thresholds

## 🏗️ Project Structure

```
multimodal-stock-modelling/
├── train.py                    # Main training script
├── main.py                     # Core model definition
├── quick_analysis.py           # Quick performance analysis tool
├── visualize_results.py        # Comprehensive visualization suite
├── training_monitor.py         # Real-time training monitoring
├── data/
│   ├── real_data_fetcher.py    # Real data fetching from APIs
│   └── real_dataset_builder.py # Dataset construction and preprocessing
├── models/
│   ├── tft_encoder.py          # Temporal Fusion Transformer
│   ├── text_encoder.py         # Financial text processing
│   ├── employment_encoder.py   # Employment data processing
│   ├── fusion_layer.py         # Cross-modal fusion mechanisms
│   └── prediction_heads.py     # Multi-task prediction heads
├── training/
│   ├── trainer.py              # Training orchestration
│   ├── loss_functions.py       # Custom loss functions
│   └── metrics.py              # Evaluation metrics
├── data_cache/                 # Cached data storage
├── checkpoints/                # Model checkpoints and results
└── requirements.txt            # Python dependencies
```

## 🔍 Monitoring and Visualization

### TensorBoard Integration
```bash
tensorboard --logdir runs/
```

### Results Analysis
Training results are automatically saved to `checkpoints/experiment_*/results.json` containing:
- Training history and metrics
- Model configuration
- Dataset statistics
- Final test performance

## 📋 Example Usage

### Single Stock Analysis
```bash
python train.py --symbols AAPL --epochs 50 --optimize_for_mac
```

### Portfolio Training
```bash
python train.py --symbols AAPL MSFT GOOGL AMZN TSLA NVDA META NFLX --epochs 200
```

### Custom Date Range
```bash
python train.py --symbols AAPL --start_date 2020-01-01 --end_date 2024-01-01 --epochs 100
```

## 🚀 Advanced Features

### Fusion Strategies
- `cross_attention`: Cross-modal attention fusion
- `gated_fusion`: Gated fusion with learnable weights
- `hierarchical`: Hierarchical pairwise fusion
- `adaptive`: Adaptive combination of multiple strategies

### MacBook Optimization
The model includes special optimizations for MacBook hardware:
- Apple Silicon (M1/M2) MPS acceleration
- Optimized threading for macOS
- Memory management for unified memory architecture

## 🔧 Training Script Arguments

```bash
python train.py [OPTIONS]

Data Configuration:
  --start_date DATE         Start date for training data (default: 2021-01-01)
  --end_date DATE           End date for training data (default: current)
  --symbols LIST            Stock symbols to train on (default: AAPL MSFT GOOGL AMZN TSLA NVDA META NFLX)
  --cache_dir STR           Directory for caching data (default: data_cache)
  --cache_expiry_hours INT  Cache expiry in hours (default: 24)

Model Configuration:
  --d_model INT             Model dimension (default: 256)
  --n_heads INT             Number of attention heads (default: 8)
  --sequence_length INT     Input sequence length (default: 252)
  --fusion_strategy STR     Fusion strategy (default: cross_attention)

Training Configuration:
  --epochs INT              Number of training epochs (default: 100)
  --batch_size INT          Batch size (default: 32)
  --learning_rate FLOAT     Learning rate (default: 0.0001)
  --weight_decay FLOAT      Weight decay (default: 0.0001)
  --optimizer STR           Optimizer (adam|adamw|sgd) (default: adamw)
  --scheduler STR           LR scheduler (cosine|step|plateau|none) (default: cosine)

Loss Configuration:
  --return_loss_weight FLOAT      Weight for return prediction loss (default: 1.0)
  --volatility_loss_weight FLOAT  Weight for volatility prediction loss (default: 0.5)
  --direction_loss_weight FLOAT   Weight for direction classification loss (default: 0.3)
  --economic_loss_weight FLOAT    Weight for economic indicator loss (default: 0.2)

Optimization:
  --optimize_for_mac        Enable MacBook-specific optimizations
  --num_workers INT         Number of data loader workers (auto-detected if not specified)
  --pin_memory              Pin memory for faster data transfer
  --persistent_workers      Keep data loading workers alive between epochs
  --mixed_precision         Use mixed precision training

Experiment:
  --experiment_name STR     Experiment name for tracking
  --checkpoint_dir STR      Directory for saving checkpoints (default: checkpoints)
  --resume STR              Resume training from checkpoint
  --device STR              Device to use (auto|cpu|cuda|mps) (default: auto)

Evaluation:
  --eval_only               Only run evaluation on test set
  --save_predictions        Save model predictions to file
```

## 🧪 Testing

### Quick Test
```bash
python train.py --symbols AAPL --epochs 1 --optimize_for_mac
```

### Full Training
```bash
python train.py --symbols AAPL MSFT GOOGL --epochs 100 --optimize_for_mac
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citations

If you use this model in your research, please cite:
```bibtex
@misc{multimodal-stock-prediction,
  title={Multimodal Stock Prediction Model},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/multimodal-stock-modelling}
}
```

## 🔗 Links

- [Project Repository](https://github.com/your-username/multimodal-stock-modelling)
- [Documentation](https://github.com/your-username/multimodal-stock-modelling/wiki)
- [Issue Tracker](https://github.com/your-username/multimodal-stock-modelling/issues)

---

**⚠️ Disclaimer**: This model is for educational and research purposes only. Do not use for actual financial trading without proper risk management and validation.