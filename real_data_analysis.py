#!/usr/bin/env python3
"""
Real Data Availability Analysis
==============================

Analyze what historical data is actually available for training.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_stock_data_availability(symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']):
    """Analyze stock data availability"""
    print("📈 Stock Data Availability Analysis")
    print("=" * 50)
    
    results = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get maximum available data
            hist = ticker.history(period="max")
            
            if not hist.empty:
                start_date = hist.index[0].strftime('%Y-%m-%d')
                end_date = hist.index[-1].strftime('%Y-%m-%d')
                total_days = len(hist)
                
                results[symbol] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_trading_days': total_days,
                    'years_available': total_days / 252  # Approximate trading days per year
                }
                
                print(f"  {symbol}: {start_date} to {end_date} ({total_days:,} days, {total_days/252:.1f} years)")
            
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
            results[symbol] = None
    
    return results

def analyze_news_data_availability():
    """Analyze news data sources and availability"""
    print("\n📰 News Data Availability Analysis")
    print("=" * 50)
    
    news_sources = {
        'Yahoo Finance RSS': {
            'url_pattern': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'historical_depth': '2-3 years',
            'reliability': 'High',
            'cost': 'Free'
        },
        'Reuters Business RSS': {
            'url_pattern': 'https://feeds.reuters.com/reuters/businessNews',
            'historical_depth': '1-2 years',
            'reliability': 'High', 
            'cost': 'Free'
        },
        'Alpha Vantage News': {
            'url_pattern': 'API based',
            'historical_depth': '2-4 years',
            'reliability': 'High',
            'cost': 'Free tier limited'
        },
        'NewsAPI': {
            'url_pattern': 'API based',
            'historical_depth': '1 month (free), 2+ years (paid)',
            'reliability': 'Medium',
            'cost': 'Paid for historical'
        }
    }
    
    print("Available Sources:")
    for source, info in news_sources.items():
        print(f"  📰 {source}")
        print(f"      Historical depth: {info['historical_depth']}")
        print(f"      Reliability: {info['reliability']}")
        print(f"      Cost: {info['cost']}")
        print()
    
    return news_sources

def analyze_employment_data_availability():
    """Analyze employment data sources"""
    print("💼 Employment Data Availability Analysis")
    print("=" * 50)
    
    employment_sources = {
        'FRED Economic Data': {
            'description': 'Federal Reserve Economic Data',
            'historical_depth': '20+ years',
            'data_types': ['unemployment rate', 'job openings', 'hiring rate'],
            'frequency': 'Monthly',
            'cost': 'Free'
        },
        'Bureau of Labor Statistics': {
            'description': 'Official US employment statistics',
            'historical_depth': '10+ years',
            'data_types': ['employment by sector', 'layoffs', 'job openings'],
            'frequency': 'Monthly',
            'cost': 'Free'
        },
        'Indeed Job Trends': {
            'description': 'Job posting trends by company/sector',
            'historical_depth': '2-3 years',
            'data_types': ['job postings volume', 'hiring trends'],
            'frequency': 'Daily/Weekly',
            'cost': 'Free (limited)'
        }
    }
    
    print("Available Sources:")
    for source, info in employment_sources.items():
        print(f"  💼 {source}")
        print(f"      Description: {info['description']}")
        print(f"      Historical depth: {info['historical_depth']}")
        print(f"      Data types: {', '.join(info['data_types'])}")
        print(f"      Frequency: {info['frequency']}")
        print(f"      Cost: {info['cost']}")
        print()
    
    return employment_sources

def recommend_training_window():
    """Recommend optimal training window based on data availability"""
    print("🎯 Training Window Recommendation")
    print("=" * 50)
    
    current_date = datetime.now()
    
    # Different window options
    windows = {
        '2 years': current_date - timedelta(days=2*365),
        '3 years': current_date - timedelta(days=3*365),
        '4 years': current_date - timedelta(days=4*365),
        '5 years': current_date - timedelta(days=5*365)
    }
    
    print("Window Options Analysis:")
    print()
    
    for window_name, start_date in windows.items():
        trading_days = len(pd.bdate_range(start_date, current_date))
        
        print(f"📅 {window_name} ({start_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}):")
        print(f"    Trading days: ~{trading_days:,}")
        print(f"    Samples for training: ~{trading_days * 5:,} (assuming 5 stocks)")  # Rough estimate
        
        # Assess data availability
        if window_name == '2 years':
            news_quality = "Excellent"
            employment_quality = "Excellent" 
            market_relevance = "Very High"
            recommendation = "Good for initial testing"
        elif window_name == '3 years':
            news_quality = "Very Good"
            employment_quality = "Very Good"
            market_relevance = "High"
            recommendation = "⭐ RECOMMENDED - Best balance"
        elif window_name == '4 years':
            news_quality = "Good"
            employment_quality = "Good"
            market_relevance = "Medium-High"
            recommendation = "Good for robust training"
        else:  # 5+ years
            news_quality = "Limited"
            employment_quality = "Fair"
            market_relevance = "Medium"
            recommendation = "Risk of outdated patterns"
        
        print(f"    News data quality: {news_quality}")
        print(f"    Employment data quality: {employment_quality}")
        print(f"    Market relevance: {market_relevance}")
        print(f"    Recommendation: {recommendation}")
        print()
    
    print("🏆 OPTIMAL CHOICE: 3-4 years")
    print("Reasons:")
    print("  ✅ Sufficient training data (~750-1000 trading days)")
    print("  ✅ News data readily available and reliable")
    print("  ✅ Employment data comprehensive")
    print("  ✅ Recent enough to be market-relevant")
    print("  ✅ Includes various market conditions without major regime changes")
    print("  ✅ Avoids COVID-19 market disruption (if starting from 2021)")
    
    # Specific recommendation
    recommended_start = datetime(2021, 1, 1)  # Post-COVID normalization
    print(f"\n🎯 SPECIFIC RECOMMENDATION:")
    print(f"   Start Date: {recommended_start.strftime('%Y-%m-%d')} (January 1, 2021)")
    print(f"   End Date: {current_date.strftime('%Y-%m-%d')} (Current)")
    print(f"   Duration: ~{(current_date - recommended_start).days / 365:.1f} years")
    print(f"   Rationale: Post-COVID market normalization, excellent data availability")
    
    return recommended_start, current_date

def test_data_fetching_speed():
    """Test how quickly we can fetch real data"""
    print("\n⚡ Data Fetching Speed Test")
    print("=" * 50)
    
    import time
    
    # Test stock data fetching
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2021-01-01'
    end_date = '2024-12-01'
    
    print(f"Testing stock data fetch for {len(symbols)} symbols from {start_date} to {end_date}")
    
    start_time = time.time()
    
    stock_data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        stock_data[symbol] = hist
        print(f"  ✅ {symbol}: {len(hist)} days fetched")
    
    fetch_time = time.time() - start_time
    
    total_rows = sum(len(df) for df in stock_data.values())
    
    print(f"\n📊 Fetching Results:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Fetch time: {fetch_time:.2f} seconds")
    print(f"  Speed: {total_rows/fetch_time:.0f} rows/second")
    
    if fetch_time < 10:
        print("  ✅ Fast - Real-time fetching feasible")
    elif fetch_time < 30:
        print("  ⚠️  Moderate - Consider caching for large datasets")
    else:
        print("  ❌ Slow - Implement aggressive caching")
    
    return stock_data

def main():
    """Run complete data availability analysis"""
    print("🔍 REAL DATA AVAILABILITY ANALYSIS")
    print("=" * 60)
    print()
    
    # Analyze each data source
    stock_results = analyze_stock_data_availability()
    news_sources = analyze_news_data_availability()
    employment_sources = analyze_employment_data_availability()
    
    # Get recommendation
    start_date, end_date = recommend_training_window()
    
    # Test fetching speed
    stock_data = test_data_fetching_speed()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY & NEXT STEPS")
    print("=" * 60)
    
    print("\n✅ RECOMMENDED TRAINING SETUP:")
    print(f"   📅 Time Window: January 1, 2021 to present (~{(datetime.now() - datetime(2021, 1, 1)).days / 365:.1f} years)")
    print(f"   📈 Stock Symbols: 5-10 major stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, etc.)")
    print(f"   📰 News Sources: Yahoo Finance RSS + Reuters (free, reliable)")
    print(f"   💼 Employment: FRED economic data + BLS statistics")
    
    print("\n🚀 IMPLEMENTATION PLAN:")
    print("   1. Update data loading modules for real data fetching")
    print("   2. Implement data caching and error handling")
    print("   3. Add proper train/validation/test time-based splits")
    print("   4. Handle missing data and data quality issues")
    print("   5. Scale up to larger batch sizes for production training")
    
    print("\n📊 EXPECTED DATASET SIZE:")
    trading_days = len(pd.bdate_range(datetime(2021, 1, 1), datetime.now()))
    estimated_samples = trading_days * 5  # 5 stocks
    print(f"   Trading days: ~{trading_days:,}")
    print(f"   Estimated samples: ~{estimated_samples:,}")
    print(f"   Memory footprint: ~{estimated_samples * 0.001:.1f} MB (rough estimate)")

if __name__ == "__main__":
    main()