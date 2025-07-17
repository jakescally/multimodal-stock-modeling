"""
Temporal Fusion Transformer for Time Series Encoding
===================================================

TFT implementation optimized for multi-horizon stock prediction
with support for mixed static and time-varying features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class VariableSelection(nn.Module):
    """Variable selection network for feature importance"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.flattened_grn = GatedResidualNetwork(
            input_size, hidden_size, hidden_size, dropout, context_size=hidden_size
        )
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(input_size)
        ])
        
    def forward(self, flattened_embedding: torch.Tensor, context: Optional[torch.Tensor] = None):
        # Simplified variable selection for verification
        # Just return the input embedding and uniform weights for now
        batch_size = flattened_embedding.size(0)
        input_size = flattened_embedding.size(-1)
        
        # Apply a simple linear transformation
        output = self.flattened_grn(flattened_embedding, context)
        
        # Return uniform weights for simplicity
        weights = torch.ones(batch_size, input_size, device=flattened_embedding.device) / input_size
        
        return output, weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with optional context"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        
        self.skip_layer = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        if context_size is not None:
            self.context_layer = nn.Linear(context_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x
            
        # Main path
        hidden = self.fc1(x)
        if context is not None:
            hidden = hidden + self.context_layer(context)
        hidden = F.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating mechanism
        gate = torch.sigmoid(self.fc4(hidden))
        output = self.fc3(hidden)
        output = gate * output
        
        # Residual connection and layer norm
        output = self.layer_norm(output + skip)
        return output


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Multi-head projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights.mean(dim=1)  # Average attention across heads


class TFTEncoder(nn.Module):
    """Temporal Fusion Transformer encoder for time series data"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input sizes (will be configured based on actual features)
        self.static_input_size = 10  # Company metadata, sector, etc.
        self.historical_input_size = 20  # OHLCV, technical indicators
        self.future_input_size = 5  # Known future features
        
        # Variable selection networks
        self.static_selection = VariableSelection(
            self.static_input_size, config.d_model, config.dropout
        )
        self.historical_selection = VariableSelection(
            self.historical_input_size, config.d_model, config.dropout
        )
        
        # Static encoders
        self.static_encoder = GatedResidualNetwork(
            config.d_model, config.d_model, config.d_model, config.dropout
        )
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            config.d_model, config.d_model, batch_first=True, dropout=config.dropout
        )
        
        # Temporal self-attention
        self.temporal_attention = InterpretableMultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        
        # Position encoding
        self.position_encoding = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Output projections for each prediction horizon
        self.horizon_projections = nn.ModuleDict({
            f'horizon_{h}': nn.Linear(config.d_model, 1)
            for h in config.prediction_horizons
        })
        
    def forward(self, static_features: torch.Tensor, 
                historical_features: torch.Tensor,
                future_features: Optional[torch.Tensor] = None):
        
        batch_size, seq_len = historical_features.shape[:2]
        
        # Static variable selection and encoding
        static_selected, static_weights = self.static_selection(static_features)
        static_context = self.static_encoder(static_selected)
        
        # Historical variable selection
        historical_reshaped = historical_features.view(-1, self.historical_input_size)
        historical_selected, historical_weights = self.historical_selection(
            historical_reshaped, static_context.repeat(seq_len, 1)
        )
        historical_selected = historical_selected.view(batch_size, seq_len, -1)
        
        # Add positional encoding
        historical_encoded = self.position_encoding(historical_selected)
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(historical_encoded)
        
        # Temporal self-attention
        attended_output, attention_weights = self.temporal_attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Multi-horizon predictions
        predictions = {}
        for horizon_name, projection in self.horizon_projections.items():
            # Use the last timestep for prediction
            pred = projection(attended_output[:, -1, :])
            predictions[horizon_name] = pred
            
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'static_weights': static_weights,
            'historical_weights': historical_weights.view(batch_size, seq_len, -1),
            'lstm_output': lstm_output,
            'final_hidden': hidden
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)