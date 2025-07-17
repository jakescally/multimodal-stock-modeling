"""
Employment Data Encoder
=======================

Processes employment-related signals including job postings, layoffs,
hiring trends, and skill demand patterns for stock prediction.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EmploymentFeatures:
    """Structure for employment-related features"""
    # Company-level features
    job_postings_30d: float
    job_postings_90d: float
    layoffs_30d: float
    layoffs_90d: float
    hiring_velocity: float  # rate of change in job postings
    
    # Skill demand features
    ai_ml_demand: float
    technical_skills_demand: float
    leadership_demand: float
    
    # Industry context
    sector_hiring_ratio: float
    sector_layoff_ratio: float
    
    # Economic context
    unemployment_rate: float
    job_openings_rate: float


class EmploymentDataProcessor:
    """Processes raw employment data into features"""
    
    def __init__(self):
        # Skill categories for classification
        self.skill_categories = {
            'ai_ml': ['machine learning', 'artificial intelligence', 'ai', 'ml', 
                     'deep learning', 'neural network', 'data science', 'nlp'],
            'technical': ['python', 'java', 'javascript', 'sql', 'aws', 'docker',
                         'kubernetes', 'react', 'angular', 'node.js'],
            'leadership': ['team lead', 'manager', 'director', 'vp', 'cto', 'ceo',
                          'leadership', 'management', 'strategic planning']
        }
        
    def extract_features_from_job_postings(self, postings_df: pd.DataFrame, 
                                         company_id: str, date: str) -> Dict:
        """Extract features from job postings data"""
        company_postings = postings_df[postings_df['company_id'] == company_id]
        
        # Time-based filtering
        date_30d = pd.to_datetime(date) - pd.Timedelta(days=30)
        date_90d = pd.to_datetime(date) - pd.Timedelta(days=90)
        
        postings_30d = company_postings[company_postings['posting_date'] >= date_30d]
        postings_90d = company_postings[company_postings['posting_date'] >= date_90d]
        
        # Basic counts
        job_count_30d = len(postings_30d)
        job_count_90d = len(postings_90d)
        
        # Hiring velocity (trend)
        if len(postings_90d) > 30:  # Need sufficient data
            recent_avg = len(postings_30d) / 30
            older_avg = (len(postings_90d) - len(postings_30d)) / 60
            hiring_velocity = (recent_avg - older_avg) / (older_avg + 1e-6)
        else:
            hiring_velocity = 0.0
            
        # Skill demand analysis
        all_descriptions = ' '.join(postings_30d['description'].fillna('').str.lower())
        
        ai_ml_score = self._calculate_skill_demand(all_descriptions, 'ai_ml')
        technical_score = self._calculate_skill_demand(all_descriptions, 'technical')
        leadership_score = self._calculate_skill_demand(all_descriptions, 'leadership')
        
        return {
            'job_postings_30d': job_count_30d,
            'job_postings_90d': job_count_90d,
            'hiring_velocity': hiring_velocity,
            'ai_ml_demand': ai_ml_score,
            'technical_skills_demand': technical_score,
            'leadership_demand': leadership_score
        }
        
    def extract_layoff_features(self, layoffs_df: pd.DataFrame, 
                               company_id: str, date: str) -> Dict:
        """Extract layoff-related features"""
        company_layoffs = layoffs_df[layoffs_df['company_id'] == company_id]
        
        date_30d = pd.to_datetime(date) - pd.Timedelta(days=30)
        date_90d = pd.to_datetime(date) - pd.Timedelta(days=90)
        
        layoffs_30d = company_layoffs[company_layoffs['layoff_date'] >= date_30d]
        layoffs_90d = company_layoffs[company_layoffs['layoff_date'] >= date_90d]
        
        return {
            'layoffs_30d': layoffs_30d['employees_affected'].sum(),
            'layoffs_90d': layoffs_90d['employees_affected'].sum()
        }
        
    def calculate_sector_context(self, sector_data: pd.DataFrame, 
                                sector: str, date: str) -> Dict:
        """Calculate sector-level employment context"""
        sector_subset = sector_data[sector_data['sector'] == sector]
        
        # Calculate ratios relative to sector average
        sector_hiring = sector_subset['job_postings'].mean()
        sector_layoffs = sector_subset['layoffs'].mean()
        
        return {
            'sector_hiring_ratio': sector_hiring,
            'sector_layoff_ratio': sector_layoffs
        }
        
    def _calculate_skill_demand(self, text: str, skill_category: str) -> float:
        """Calculate demand score for a skill category"""
        skills = self.skill_categories[skill_category]
        
        total_mentions = 0
        for skill in skills:
            total_mentions += text.count(skill)
            
        # Normalize by text length (approximate)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        return total_mentions / word_count * 1000  # Scale for readability


class EmploymentEncoder(nn.Module):
    """Neural encoder for employment features"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Employment feature dimensions
        self.company_features_dim = 6  # job postings, layoffs, velocity, skills
        self.sector_features_dim = 2   # sector ratios
        self.macro_features_dim = 2    # unemployment, job openings rate
        
        self.total_features = (self.company_features_dim + 
                              self.sector_features_dim + 
                              self.macro_features_dim)
        
        # Feature processing layers
        self.company_encoder = nn.Sequential(
            nn.Linear(self.company_features_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.d_model // 2)
        )
        
        self.sector_encoder = nn.Sequential(
            nn.Linear(self.sector_features_dim, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, config.d_model // 4)
        )
        
        self.macro_encoder = nn.Sequential(
            nn.Linear(self.macro_features_dim, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, config.d_model // 4)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Temporal encoding for employment trends
        self.temporal_conv = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1
        )
        
        # Employment impact predictor (auxiliary task)
        self.impact_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, employment_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            employment_data: Dictionary containing:
                - company_features: [batch, seq_len, company_features_dim]
                - sector_features: [batch, seq_len, sector_features_dim]  
                - macro_features: [batch, seq_len, macro_features_dim]
        """
        # Encode each feature type
        company_encoded = self.company_encoder(employment_data['company_features'])
        sector_encoded = self.sector_encoder(employment_data['sector_features'])
        macro_encoded = self.macro_encoder(employment_data['macro_features'])
        
        # Concatenate all encodings
        combined = torch.cat([company_encoded, sector_encoded, macro_encoded], dim=-1)
        
        # Fusion layer
        fused = self.fusion(combined)
        
        # Temporal convolution for trend detection
        # [batch, seq_len, d_model] -> [batch, d_model, seq_len] -> [batch, d_model, seq_len]
        temporal_input = fused.transpose(1, 2)
        temporal_output = self.temporal_conv(temporal_input)
        temporal_features = temporal_output.transpose(1, 2)
        
        # Employment impact prediction (auxiliary task)
        impact_scores = self.impact_predictor(temporal_features)
        
        return {
            'employment_embeddings': temporal_features,
            'impact_scores': impact_scores,
            'company_features': company_encoded,
            'sector_features': sector_encoded,
            'macro_features': macro_encoded
        }


class EmploymentSignalGenerator:
    """Generates trading signals from employment data"""
    
    def __init__(self):
        self.signal_weights = {
            'hiring_acceleration': 0.3,
            'layoff_deceleration': 0.2,
            'skill_demand_growth': 0.2,
            'sector_outperformance': 0.3
        }
        
    def generate_signals(self, employment_features: EmploymentFeatures) -> Dict[str, float]:
        """Generate buy/sell signals from employment features"""
        signals = {}
        
        # Hiring acceleration signal
        if employment_features.hiring_velocity > 0.1:
            signals['hiring_acceleration'] = min(employment_features.hiring_velocity, 1.0)
        else:
            signals['hiring_acceleration'] = 0.0
            
        # Layoff signal (inverse)
        if employment_features.layoffs_30d == 0:
            signals['layoff_deceleration'] = 1.0
        else:
            signals['layoff_deceleration'] = max(0.0, 1.0 - employment_features.layoffs_30d / 1000)
            
        # Skill demand signal
        ai_signal = min(employment_features.ai_ml_demand / 0.1, 1.0)  # Normalize
        signals['skill_demand_growth'] = ai_signal
        
        # Sector relative performance
        if employment_features.sector_hiring_ratio > 0:
            relative_performance = (employment_features.job_postings_30d / 
                                  (employment_features.sector_hiring_ratio + 1e-6))
            signals['sector_outperformance'] = min(relative_performance / 2.0, 1.0)
        else:
            signals['sector_outperformance'] = 0.0
            
        return signals
        
    def aggregate_signal(self, signals: Dict[str, float]) -> float:
        """Aggregate individual signals into overall employment signal"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for signal_name, value in signals.items():
            if signal_name in self.signal_weights:
                weight = self.signal_weights[signal_name]
                weighted_sum += weight * value
                total_weight += weight
                
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0  # Neutral signal


class MacroEmploymentProcessor:
    """Processes macro-level employment indicators"""
    
    def __init__(self):
        # BLS series IDs for key employment indicators
        self.bls_series = {
            'unemployment_rate': 'LNS14000000',
            'job_openings_rate': 'JTS000000000000000JOR',
            'layoffs_rate': 'JTS000000000000000LSR',
            'quits_rate': 'JTS000000000000000QUR'
        }
        
    def process_bls_data(self, bls_data: pd.DataFrame, date: str) -> Dict[str, float]:
        """Process Bureau of Labor Statistics data"""
        target_date = pd.to_datetime(date)
        
        # Get most recent data before target date
        recent_data = bls_data[bls_data['date'] <= target_date].tail(1)
        
        if len(recent_data) == 0:
            return {key: 0.0 for key in self.bls_series.keys()}
            
        return {
            'unemployment_rate': recent_data['unemployment_rate'].iloc[0],
            'job_openings_rate': recent_data['job_openings_rate'].iloc[0],
            'layoffs_rate': recent_data.get('layoffs_rate', [0.0]).iloc[0],
            'quits_rate': recent_data.get('quits_rate', [0.0]).iloc[0]
        }
        
    def calculate_labor_market_tightness(self, macro_data: Dict[str, float]) -> float:
        """Calculate overall labor market tightness score"""
        # Higher job openings + lower unemployment = tighter market
        # Tighter market generally positive for wages/spending but may indicate overheating
        
        job_openings = macro_data.get('job_openings_rate', 0.0)
        unemployment = macro_data.get('unemployment_rate', 5.0)
        
        # Normalize to 0-1 scale
        tightness = (job_openings / 10.0) * (1.0 - unemployment / 15.0)
        return max(0.0, min(1.0, tightness))