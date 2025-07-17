"""
News Data Loader
===============

Handles downloading and preprocessing of financial news data.
Supports web scraping, RSS feeds, and API-based news sources.
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import pickle
import re
from bs4 import BeautifulSoup
import feedparser
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsDataLoader:
    """Main class for loading financial news data"""
    
    def __init__(self, cache_dir: str = "data/news_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Common financial news RSS feeds
        self.rss_feeds = {
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'reuters_markets': 'https://feeds.reuters.com/reuters/marketsNews',
            'yahoo_finance': 'https://finance.yahoo.com/rss/',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
        }
        
        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
    def get_rss_news(self, feed_name: str, max_articles: int = 100) -> List[Dict]:
        """
        Get news from RSS feeds
        
        Args:
            feed_name: Name of RSS feed from self.rss_feeds
            max_articles: Maximum number of articles to retrieve
            
        Returns:
            List of news articles with metadata
        """
        if feed_name not in self.rss_feeds:
            logger.error(f"Unknown RSS feed: {feed_name}")
            return []
            
        cache_file = self.cache_dir / f"rss_{feed_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        # Check cache (daily cache)
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            logger.info(f"Fetching RSS feed: {feed_name}")
            feed_url = self.rss_feeds[feed_name]
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed_name,
                    'date_collected': datetime.now().isoformat()
                }
                
                # Parse publication date
                try:
                    if 'published_parsed' in entry:
                        article['published_date'] = datetime(*entry.published_parsed[:6])
                    else:
                        article['published_date'] = datetime.now()
                except:
                    article['published_date'] = datetime.now()
                    
                articles.append(article)
            
            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump(articles, f)
                
            logger.info(f"Retrieved {len(articles)} articles from {feed_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed_name}: {e}")
            return []
    
    def get_all_rss_news(self, max_articles_per_feed: int = 50) -> List[Dict]:
        """Get news from all RSS feeds"""
        all_articles = []
        
        for feed_name in self.rss_feeds.keys():
            articles = self.get_rss_news(feed_name, max_articles_per_feed)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting
            
        return all_articles
    
    def search_company_news(self, company_name: str, ticker: str, 
                           days_back: int = 30) -> List[Dict]:
        """
        Search for news about a specific company
        
        Args:
            company_name: Full company name
            ticker: Stock ticker symbol
            days_back: How many days back to search
            
        Returns:
            List of relevant news articles
        """
        # Get recent news
        all_news = self.get_all_rss_news()
        
        # Filter for company mentions
        relevant_news = []
        search_terms = [company_name.lower(), ticker.lower()]
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for article in all_news:
            # Check date
            if article['published_date'] < cutoff_date:
                continue
                
            # Check for company mentions
            text = (article['title'] + ' ' + article['summary']).lower()
            
            if any(term in text for term in search_terms):
                article['relevance_score'] = self._calculate_relevance(text, search_terms)
                relevant_news.append(article)
        
        # Sort by relevance and date
        relevant_news.sort(key=lambda x: (x['relevance_score'], x['published_date']), 
                          reverse=True)
        
        return relevant_news
    
    def _calculate_relevance(self, text: str, search_terms: List[str]) -> float:
        """Calculate relevance score for an article"""
        score = 0.0
        
        for term in search_terms:
            # Count mentions
            mentions = text.count(term)
            score += mentions
            
            # Bonus for title mentions
            if term in text[:100]:  # Approximate title area
                score += 0.5
        
        return score


class NewsProcessor:
    """Process and clean news data for ML models"""
    
    def __init__(self):
        # Financial keywords for filtering
        self.financial_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'acquisition', 'merger', 'ipo', 'dividend', 'buyback', 'debt',
            'partnership', 'contract', 'regulation', 'lawsuit', 'fda',
            'expansion', 'restructuring', 'layoffs', 'hiring', 'ceo', 'cfo'
        }
        
        # Sentiment indicators
        self.positive_words = {
            'growth', 'profit', 'beat', 'exceed', 'strong', 'bullish',
            'upgrade', 'outperform', 'buy', 'positive', 'gain', 'rise',
            'success', 'record', 'milestone', 'breakthrough', 'innovation'
        }
        
        self.negative_words = {
            'loss', 'miss', 'weak', 'bearish', 'downgrade', 'underperform',
            'sell', 'negative', 'fall', 'decline', 'risk', 'concern',
            'warning', 'struggle', 'challenging', 'difficult', 'crisis'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\$\%]', '', text)
        
        return text.strip()
    
    def calculate_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores for text"""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': max(0.0, neutral_score)
        }
    
    def extract_financial_entities(self, text: str) -> Dict[str, int]:
        """Extract financial entities and concepts"""
        text_lower = text.lower()
        
        entities = {}
        
        # Count financial keywords
        for keyword in self.financial_keywords:
            count = text_lower.count(keyword)
            if count > 0:
                entities[keyword] = count
        
        # Extract monetary amounts
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?'
        money_matches = re.findall(money_pattern, text, re.IGNORECASE)
        entities['monetary_mentions'] = len(money_matches)
        
        # Extract percentages
        percent_pattern = r'\d+(?:\.\d+)?%'
        percent_matches = re.findall(percent_pattern, text)
        entities['percentage_mentions'] = len(percent_matches)
        
        return entities
    
    def process_news_batch(self, articles: List[Dict]) -> pd.DataFrame:
        """Process a batch of news articles"""
        processed_articles = []
        
        for article in articles:
            # Clean text
            title_clean = self.clean_text(article.get('title', ''))
            summary_clean = self.clean_text(article.get('summary', ''))
            full_text = title_clean + ' ' + summary_clean
            
            # Calculate sentiment
            sentiment = self.calculate_sentiment(full_text)
            
            # Extract entities
            entities = self.extract_financial_entities(full_text)
            
            # Create processed article
            processed = {
                'title': title_clean,
                'summary': summary_clean,
                'full_text': full_text,
                'source': article.get('source', ''),
                'published_date': article.get('published_date'),
                'link': article.get('link', ''),
                'text_length': len(full_text),
                'word_count': len(full_text.split()),
                'sentiment_positive': sentiment['positive'],
                'sentiment_negative': sentiment['negative'],
                'sentiment_neutral': sentiment['neutral'],
                'financial_keyword_count': sum(entities.values()),
                'monetary_mentions': entities.get('monetary_mentions', 0),
                'percentage_mentions': entities.get('percentage_mentions', 0)
            }
            
            # Add individual financial keywords
            for keyword in self.financial_keywords:
                processed[f'keyword_{keyword}'] = entities.get(keyword, 0)
            
            processed_articles.append(processed)
        
        return pd.DataFrame(processed_articles)


class NewsAggregator:
    """Aggregate news data for time series alignment"""
    
    def __init__(self):
        self.processor = NewsProcessor()
    
    def aggregate_daily_news(self, news_df: pd.DataFrame, 
                           date_column: str = 'published_date') -> pd.DataFrame:
        """
        Aggregate news by day for time series alignment
        
        Args:
            news_df: DataFrame with processed news articles
            date_column: Column containing publication dates
            
        Returns:
            DataFrame with daily aggregated news features
        """
        # Convert to datetime if needed
        news_df[date_column] = pd.to_datetime(news_df[date_column])
        
        # Extract date only (no time)
        news_df['date'] = news_df[date_column].dt.date
        
        # Aggregation functions
        agg_functions = {
            'title': 'count',  # Number of articles
            'text_length': ['mean', 'sum'],
            'word_count': ['mean', 'sum'],
            'sentiment_positive': ['mean', 'max'],
            'sentiment_negative': ['mean', 'max'],
            'sentiment_neutral': 'mean',
            'financial_keyword_count': ['sum', 'mean'],
            'monetary_mentions': 'sum',
            'percentage_mentions': 'sum'
        }
        
        # Add keyword aggregations
        keyword_cols = [col for col in news_df.columns if col.startswith('keyword_')]
        for col in keyword_cols:
            agg_functions[col] = 'sum'
        
        # Perform aggregation
        daily_agg = news_df.groupby('date').agg(agg_functions)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() if col[1] else col[0] 
                           for col in daily_agg.columns.values]
        
        # Rename count column
        daily_agg = daily_agg.rename(columns={'title_count': 'article_count'})
        
        # Reset index
        daily_agg = daily_agg.reset_index()
        
        # Calculate derived features
        daily_agg['avg_sentiment_score'] = (
            daily_agg['sentiment_positive_mean'] - daily_agg['sentiment_negative_mean']
        )
        
        daily_agg['news_intensity'] = (
            daily_agg['article_count'] * daily_agg['text_length_mean'] / 1000
        )
        
        return daily_agg
    
    def align_with_stock_data(self, stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align news data with stock trading days
        
        Args:
            stock_df: DataFrame with stock data (must have date index or column)
            news_df: DataFrame with daily aggregated news data
            
        Returns:
            DataFrame with aligned stock and news data
        """
        # Ensure stock_df has a date column
        if isinstance(stock_df.index, pd.DatetimeIndex):
            stock_df = stock_df.reset_index()
            stock_df['date'] = stock_df['Date'].dt.date
        else:
            stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
        
        # Merge with news data
        merged = stock_df.merge(news_df, on='date', how='left')
        
        # Fill missing news data with zeros/defaults
        news_columns = [col for col in news_df.columns if col != 'date']
        merged[news_columns] = merged[news_columns].fillna(0)
        
        return merged


class CompanyNewsMapper:
    """Map news articles to specific companies and tickers"""
    
    def __init__(self):
        # Common company name variations and aliases
        self.company_aliases = {
            'AAPL': ['apple inc', 'apple', 'iphone', 'ipad', 'mac'],
            'MSFT': ['microsoft', 'windows', 'office', 'azure', 'xbox'],
            'GOOGL': ['google', 'alphabet', 'youtube', 'android', 'gmail'],
            'AMZN': ['amazon', 'aws', 'alexa', 'prime'],
            'TSLA': ['tesla', 'spacex', 'elon musk'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp'],
            'NVDA': ['nvidia', 'geforce', 'cuda'],
            'JPM': ['jpmorgan', 'jp morgan', 'chase'],
            'JNJ': ['johnson & johnson', 'johnson and johnson', 'j&j']
        }
    
    def map_articles_to_companies(self, articles: List[Dict], 
                                company_tickers: List[str]) -> Dict[str, List[Dict]]:
        """
        Map news articles to specific companies
        
        Args:
            articles: List of news articles
            company_tickers: List of stock tickers to map to
            
        Returns:
            Dictionary mapping tickers to relevant articles
        """
        company_news = {ticker: [] for ticker in company_tickers}
        
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('summary', '')).lower()
            
            for ticker in company_tickers:
                # Check direct ticker mention
                if ticker.lower() in text:
                    article_copy = article.copy()
                    article_copy['match_type'] = 'ticker'
                    article_copy['confidence'] = 0.9
                    company_news[ticker].append(article_copy)
                    continue
                
                # Check company aliases
                if ticker in self.company_aliases:
                    for alias in self.company_aliases[ticker]:
                        if alias in text:
                            article_copy = article.copy()
                            article_copy['match_type'] = 'alias'
                            article_copy['confidence'] = 0.7
                            company_news[ticker].append(article_copy)
                            break
        
        return company_news