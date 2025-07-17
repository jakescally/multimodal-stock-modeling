"""
Cross-Modal Fusion Layer
=======================

Advanced fusion mechanisms for combining time series, text, and employment data
using attention-based architectures and learned feature interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for fusion layer"""
    d_model: int = 256
    n_heads: int = 8
    n_fusion_layers: int = 3
    dropout: float = 0.1
    
    # Modality dimensions
    stock_dim: int = 256
    text_dim: int = 256  
    employment_dim: int = 256
    
    # Fusion strategy
    fusion_strategy: str = "cross_attention"  # "cross_attention", "concatenate", "gated_fusion"
    use_residual: bool = True
    use_layer_norm: bool = True


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different modalities"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Query, Key, Value projections for each modality
        self.stock_q = nn.Linear(config.stock_dim, config.d_model, bias=False)
        self.stock_k = nn.Linear(config.stock_dim, config.d_model, bias=False)
        self.stock_v = nn.Linear(config.stock_dim, config.d_model, bias=False)
        
        self.text_q = nn.Linear(config.text_dim, config.d_model, bias=False)
        self.text_k = nn.Linear(config.text_dim, config.d_model, bias=False)
        self.text_v = nn.Linear(config.text_dim, config.d_model, bias=False)
        
        self.employment_q = nn.Linear(config.employment_dim, config.d_model, bias=False)
        self.employment_k = nn.Linear(config.employment_dim, config.d_model, bias=False)
        self.employment_v = nn.Linear(config.employment_dim, config.d_model, bias=False)
        
        # Output projections
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, stock_features: torch.Tensor, 
                text_features: torch.Tensor,
                employment_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            stock_features: [batch, seq_len, stock_dim]
            text_features: [batch, seq_len, text_dim]  
            employment_features: [batch, seq_len, employment_dim]
            
        Returns:
            Dictionary with fused features and attention weights
        """
        batch_size, seq_len = stock_features.size(0), stock_features.size(1)
        
        # Project to common dimensionality
        stock_q = self.stock_q(stock_features)
        stock_k = self.stock_k(stock_features)
        stock_v = self.stock_v(stock_features)
        
        text_q = self.text_q(text_features)
        text_k = self.text_k(text_features)
        text_v = self.text_v(text_features)
        
        employment_q = self.employment_q(employment_features)
        employment_k = self.employment_k(employment_features)
        employment_v = self.employment_v(employment_features)
        
        # Reshape for multi-head attention
        stock_q = stock_q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        stock_k = stock_k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        stock_v = stock_v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        text_q = text_q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        text_k = text_k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        text_v = text_v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        employment_q = employment_q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        employment_k = employment_k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        employment_v = employment_v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Cross-modal attention: each modality attends to others
        fused_modalities = []
        attention_weights = {}
        
        # Stock attending to text and employment
        stock_fused, stock_attn = self._cross_attend(
            stock_q, torch.cat([text_k, employment_k], dim=2), torch.cat([text_v, employment_v], dim=2)
        )
        fused_modalities.append(stock_fused)
        attention_weights['stock_to_others'] = stock_attn
        
        # Text attending to stock and employment  
        text_fused, text_attn = self._cross_attend(
            text_q, torch.cat([stock_k, employment_k], dim=2), torch.cat([stock_v, employment_v], dim=2)
        )
        fused_modalities.append(text_fused)
        attention_weights['text_to_others'] = text_attn
        
        # Employment attending to stock and text
        employment_fused, employment_attn = self._cross_attend(
            employment_q, torch.cat([stock_k, text_k], dim=2), torch.cat([stock_v, text_v], dim=2)
        )
        fused_modalities.append(employment_fused)
        attention_weights['employment_to_others'] = employment_attn
        
        # Weighted combination of modalities
        modality_weights = F.softmax(self.modality_weights, dim=0)
        fused_output = sum(w * mod for w, mod in zip(modality_weights, fused_modalities))
        
        # Output projection and residual connection
        fused_output = fused_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.output_projection(fused_output)
        
        if self.config.use_layer_norm:
            output = self.layer_norm(output)
            
        output = self.dropout(output)
        
        return {
            'fused_features': output,
            'attention_weights': attention_weights,
            'modality_weights': modality_weights
        }
    
    def _cross_attend(self, query: torch.Tensor, key: torch.Tensor, 
                     value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform cross-attention between query and key/value"""
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)
        
        return attended, attention_weights


class GatedFusion(nn.Module):
    """Gated fusion mechanism with learnable gates for each modality"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Input projections to common space
        self.stock_projection = nn.Linear(config.stock_dim, config.d_model)
        self.text_projection = nn.Linear(config.text_dim, config.d_model)
        self.employment_projection = nn.Linear(config.employment_dim, config.d_model)
        
        # Gating networks
        self.stock_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.Sigmoid()
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.Sigmoid()
        )
        
        self.employment_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.Sigmoid()
        )
        
        # Cross-modal interaction layers
        self.interaction_layer = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, stock_features: torch.Tensor,
                text_features: torch.Tensor,
                employment_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            stock_features: [batch, seq_len, stock_dim]
            text_features: [batch, seq_len, text_dim]
            employment_features: [batch, seq_len, employment_dim]
        """
        # Project to common space
        stock_proj = self.stock_projection(stock_features)
        text_proj = self.text_projection(text_features)
        employment_proj = self.employment_projection(employment_features)
        
        # Apply gating
        stock_gated = stock_proj * self.stock_gate(stock_proj)
        text_gated = text_proj * self.text_gate(text_proj)
        employment_gated = employment_proj * self.employment_gate(employment_proj)
        
        # Concatenate for cross-modal interaction
        concatenated = torch.cat([stock_gated, text_gated, employment_gated], dim=-1)
        
        # Cross-modal interaction
        fused = self.interaction_layer(concatenated)
        
        # Residual connection and normalization
        if self.config.use_residual:
            # Average of input projections as residual
            residual = (stock_proj + text_proj + employment_proj) / 3
            fused = fused + residual
            
        if self.config.use_layer_norm:
            fused = self.layer_norm(fused)
        
        return {
            'fused_features': fused,
            'gate_values': {
                'stock': self.stock_gate(stock_proj),
                'text': self.text_gate(text_proj),
                'employment': self.employment_gate(employment_proj)
            }
        }


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion that first fuses pairs, then combines all"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Pairwise fusion layers
        self.stock_text_fusion = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.stock_employment_fusion = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.text_employment_fusion = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        
        # Final fusion layer
        self.final_fusion = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        
        # Input projections
        self.stock_projection = nn.Linear(config.stock_dim, config.d_model)
        self.text_projection = nn.Linear(config.text_dim, config.d_model)
        self.employment_projection = nn.Linear(config.employment_dim, config.d_model)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.layer_norm3 = nn.LayerNorm(config.d_model)
        self.layer_norm_final = nn.LayerNorm(config.d_model)
        
    def forward(self, stock_features: torch.Tensor,
                text_features: torch.Tensor,
                employment_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hierarchical fusion process"""
        
        # Project to common space
        stock_proj = self.stock_projection(stock_features)
        text_proj = self.text_projection(text_features)
        employment_proj = self.employment_projection(employment_features)
        
        # Pairwise fusion
        stock_text_fused, _ = self.stock_text_fusion(stock_proj, text_proj, text_proj)
        stock_text_fused = self.layer_norm1(stock_text_fused + stock_proj)
        
        stock_emp_fused, _ = self.stock_employment_fusion(stock_proj, employment_proj, employment_proj)
        stock_emp_fused = self.layer_norm2(stock_emp_fused + stock_proj)
        
        text_emp_fused, _ = self.text_employment_fusion(text_proj, employment_proj, employment_proj)
        text_emp_fused = self.layer_norm3(text_emp_fused + text_proj)
        
        # Combine pairwise fusions
        combined = (stock_text_fused + stock_emp_fused + text_emp_fused) / 3
        
        # Final fusion
        final_fused, final_attn = self.final_fusion(combined, combined, combined)
        final_fused = self.layer_norm_final(final_fused + combined)
        
        return {
            'fused_features': final_fused,
            'attention_weights': final_attn,
            'pairwise_fusions': {
                'stock_text': stock_text_fused,
                'stock_employment': stock_emp_fused,
                'text_employment': text_emp_fused
            }
        }


class AdaptiveFusion(nn.Module):
    """Adaptive fusion that learns to weight different fusion strategies"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Multiple fusion strategies
        self.cross_attention_fusion = CrossModalAttention(config)
        self.gated_fusion = GatedFusion(config)
        self.hierarchical_fusion = HierarchicalFusion(config)
        
        # Strategy weighting network
        self.strategy_weights = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 3),  # 3 strategies
            nn.Softmax(dim=-1)
        )
        
        # Final combination layer
        self.final_combination = nn.Linear(config.d_model * 3, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, stock_features: torch.Tensor,
                text_features: torch.Tensor,
                employment_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adaptive fusion combining multiple strategies"""
        
        # Apply different fusion strategies
        cross_attn_output = self.cross_attention_fusion(stock_features, text_features, employment_features)
        gated_output = self.gated_fusion(stock_features, text_features, employment_features)
        hierarchical_output = self.hierarchical_fusion(stock_features, text_features, employment_features)
        
        # Extract fused features
        cross_attn_features = cross_attn_output['fused_features']
        gated_features = gated_output['fused_features']
        hierarchical_features = hierarchical_output['fused_features']
        
        # Compute strategy weights based on concatenated features
        strategy_input = torch.cat([
            cross_attn_features.mean(dim=1),  # Global average pooling
            gated_features.mean(dim=1),
            hierarchical_features.mean(dim=1)
        ], dim=-1)
        
        strategy_weights = self.strategy_weights(strategy_input)  # [batch, 3]
        
        # Weighted combination
        combined_features = (
            strategy_weights[:, 0:1].unsqueeze(1) * cross_attn_features +
            strategy_weights[:, 1:2].unsqueeze(1) * gated_features +
            strategy_weights[:, 2:3].unsqueeze(1) * hierarchical_features
        )
        
        # Final processing
        final_features = self.layer_norm(combined_features)
        
        return {
            'fused_features': final_features,
            'strategy_weights': strategy_weights,
            'individual_outputs': {
                'cross_attention': cross_attn_output,
                'gated': gated_output,
                'hierarchical': hierarchical_output
            }
        }


class MultiModalFusionLayer(nn.Module):
    """Main fusion layer that supports multiple fusion strategies"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Select fusion strategy
        if config.fusion_strategy == "cross_attention":
            self.fusion_module = CrossModalAttention(config)
        elif config.fusion_strategy == "gated_fusion":
            self.fusion_module = GatedFusion(config)
        elif config.fusion_strategy == "hierarchical":
            self.fusion_module = HierarchicalFusion(config)
        elif config.fusion_strategy == "adaptive":
            self.fusion_module = AdaptiveFusion(config)
        else:
            raise ValueError(f"Unknown fusion strategy: {config.fusion_strategy}")
        
        # Temporal processing after fusion
        self.temporal_processor = nn.LSTM(
            config.d_model, 
            config.d_model // 2, 
            num_layers=2, 
            batch_first=True, 
            dropout=config.dropout,
            bidirectional=True
        )
        
        # Final output projection
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_features: Dictionary containing:
                - 'stock': [batch, seq_len, stock_dim]
                - 'text': [batch, seq_len, text_dim]
                - 'employment': [batch, seq_len, employment_dim]
        
        Returns:
            Dictionary with fused features and additional outputs
        """
        stock_features = modality_features['stock']
        text_features = modality_features['text']
        employment_features = modality_features['employment']
        
        # Apply fusion
        fusion_output = self.fusion_module(stock_features, text_features, employment_features)
        fused_features = fusion_output['fused_features']
        
        # Temporal processing
        temporal_output, (hidden, cell) = self.temporal_processor(fused_features)
        
        # Final projection
        output_features = self.output_projection(temporal_output)
        
        return {
            'fused_features': output_features,
            'temporal_hidden': hidden,
            'temporal_cell': cell,
            'fusion_details': fusion_output,
            'sequence_representation': temporal_output[:, -1, :]  # Last timestep
        }


class ModalityAlignment(nn.Module):
    """Aligns different modalities to handle temporal misalignment"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Learnable temporal offsets for each modality
        self.stock_offset = nn.Parameter(torch.zeros(1))
        self.text_offset = nn.Parameter(torch.zeros(1))
        self.employment_offset = nn.Parameter(torch.zeros(1))
        
        # Temporal interpolation layers
        self.stock_interpolator = nn.Conv1d(config.stock_dim, config.stock_dim, 
                                          kernel_size=3, padding=1)
        self.text_interpolator = nn.Conv1d(config.text_dim, config.text_dim,
                                         kernel_size=3, padding=1)
        self.employment_interpolator = nn.Conv1d(config.employment_dim, config.employment_dim,
                                               kernel_size=3, padding=1)
        
    def forward(self, stock_features: torch.Tensor,
                text_features: torch.Tensor,
                employment_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align modalities temporally"""
        
        # Apply learned temporal offsets (simplified version)
        # In practice, this would involve more sophisticated temporal alignment
        
        # Apply interpolation for smoothing
        stock_aligned = self.stock_interpolator(stock_features.transpose(1, 2)).transpose(1, 2)
        text_aligned = self.text_interpolator(text_features.transpose(1, 2)).transpose(1, 2)
        employment_aligned = self.employment_interpolator(employment_features.transpose(1, 2)).transpose(1, 2)
        
        return stock_aligned, text_aligned, employment_aligned