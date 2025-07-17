"""
Text Encoder for Qualitative Data Processing
===========================================

BERT-based encoder for processing news, earnings transcripts,
SEC filings, and other textual information relevant to stock performance.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import numpy as np


class FinancialTextEncoder(nn.Module):
    """BERT-based encoder fine-tuned for financial text"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        self.bert = AutoModel.from_pretrained(config.text_model)
        
        # Freeze BERT initially (can unfreeze for fine-tuning)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Financial domain adaptation layers
        self.financial_adapter = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Sentiment analysis head
        self.sentiment_head = nn.Linear(config.d_model, 3)  # negative, neutral, positive
        
        # Topic classification head for financial topics
        self.topic_head = nn.Linear(config.d_model, 10)  # earnings, M&A, regulation, etc.
        
        # Temporal aggregation for daily text embeddings
        self.temporal_aggregator = TemporalTextAggregator(config.d_model)
        
    def encode_single_text(self, text: str) -> torch.Tensor:
        """Encode a single piece of text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_text_length,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Apply financial domain adaptation
        financial_embedding = self.financial_adapter(cls_embedding)
        
        return financial_embedding
        
    def forward(self, text_batch: List[str]) -> Dict[str, torch.Tensor]:
        """Process a batch of texts"""
        # Tokenize batch
        inputs = self.tokenizer(
            text_batch,
            return_tensors="pt",
            max_length=self.config.max_text_length,
            truncation=True,
            padding=True
        )
        
        # BERT encoding
        with torch.no_grad():
            outputs = self.bert(**inputs)
            
        # Extract [CLS] representations
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Financial domain adaptation
        financial_embeddings = self.financial_adapter(cls_embeddings)
        
        # Auxiliary predictions
        sentiments = self.sentiment_head(financial_embeddings)
        topics = self.topic_head(financial_embeddings)
        
        return {
            'embeddings': financial_embeddings,
            'sentiments': sentiments,
            'topics': topics,
            'attention_mask': inputs['attention_mask']
        }
        
    def process_daily_texts(self, daily_texts: List[List[str]]) -> torch.Tensor:
        """Process texts grouped by day and aggregate"""
        daily_embeddings = []
        
        for day_texts in daily_texts:
            if not day_texts:
                # No news for this day - use zero embedding
                zero_embed = torch.zeros(1, self.config.d_model)
                daily_embeddings.append(zero_embed)
            else:
                # Encode all texts for the day
                day_outputs = self.forward(day_texts)
                day_embedding = self.temporal_aggregator(day_outputs['embeddings'])
                daily_embeddings.append(day_embedding)
                
        return torch.cat(daily_embeddings, dim=0)


class TemporalTextAggregator(nn.Module):
    """Aggregates multiple text embeddings for a single time period"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Attention-based aggregation
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Learnable query for aggregation
        self.aggregation_query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [num_texts, d_model] - embeddings for texts from same day
        Returns:
            aggregated: [1, d_model] - single embedding for the day
        """
        if embeddings.size(0) == 1:
            return embeddings
            
        # Add batch dimension
        embeddings = embeddings.unsqueeze(0)  # [1, num_texts, d_model]
        
        # Attention-based aggregation using learnable query
        query = self.aggregation_query.expand(embeddings.size(0), -1, -1)
        aggregated, _ = self.attention(query, embeddings, embeddings)
        
        # Layer normalization
        aggregated = self.layer_norm(aggregated)
        
        return aggregated  # [1, 1, d_model]


class NewsProcessor:
    """Utility class for processing news data"""
    
    def __init__(self):
        # Financial keywords for relevance filtering
        self.financial_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'acquisition', 'merger', 'ipo', 'dividend', 'buyback', 'debt',
            'partnership', 'contract', 'regulation', 'lawsuit', 'fda',
            'expansion', 'restructuring', 'layoffs', 'hiring'
        }
        
    def filter_relevant_news(self, news_items: List[Dict]) -> List[Dict]:
        """Filter news items for financial relevance"""
        relevant_news = []
        
        for item in news_items:
            text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
            
            # Check for financial keywords
            if any(keyword in text for keyword in self.financial_keywords):
                relevant_news.append(item)
                
        return relevant_news
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        return text
        
    def extract_company_mentions(self, text: str, company_names: List[str]) -> List[str]:
        """Extract mentions of specific companies from text"""
        mentions = []
        text_lower = text.lower()
        
        for company in company_names:
            if company.lower() in text_lower:
                mentions.append(company)
                
        return mentions


class SentimentAnalyzer:
    """Enhanced sentiment analysis for financial text"""
    
    def __init__(self):
        # Financial sentiment lexicon (subset)
        self.positive_words = {
            'growth', 'profit', 'beat', 'exceed', 'strong', 'bullish',
            'upgrade', 'outperform', 'buy', 'positive', 'gain', 'rise'
        }
        
        self.negative_words = {
            'loss', 'miss', 'weak', 'bearish', 'downgrade', 'underperform',
            'sell', 'negative', 'fall', 'decline', 'risk', 'concern'
        }
        
    def lexicon_sentiment(self, text: str) -> float:
        """Simple lexicon-based sentiment scoring"""
        words = text.lower().split()
        
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        
        if positive_score + negative_score == 0:
            return 0.0  # Neutral
            
        return (positive_score - negative_score) / (positive_score + negative_score)
        
    def market_impact_score(self, text: str, company_name: str) -> float:
        """Estimate potential market impact of news"""
        # Simple heuristic based on keywords and company mentions
        impact_keywords = {
            'acquisition': 0.8, 'merger': 0.8, 'earnings': 0.6,
            'fda approval': 0.7, 'lawsuit': -0.6, 'bankruptcy': -0.9,
            'partnership': 0.4, 'contract': 0.3, 'layoffs': -0.4
        }
        
        text_lower = text.lower()
        company_mentioned = company_name.lower() in text_lower
        
        if not company_mentioned:
            return 0.0
            
        max_impact = 0.0
        for keyword, impact in impact_keywords.items():
            if keyword in text_lower:
                max_impact = max(max_impact, abs(impact))
                
        return max_impact