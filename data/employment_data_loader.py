"""
Employment Data Loader
=====================

Handles downloading and preprocessing of employment data including
job postings, layoffs, hiring trends, and macroeconomic indicators.
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import pickle
import re
from bs4 import BeautifulSoup
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmploymentDataLoader:
    """Main class for loading employment-related data"""
    
    def __init__(self, cache_dir: str = "data/employment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        # BLS API endpoint for economic data
        self.bls_api_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        
        # Common BLS series IDs for employment indicators
        self.bls_series = {
            'unemployment_rate': 'LNS14000000',
            'job_openings_rate': 'JTS000000000000000JOR', 
            'layoffs_rate': 'JTS000000000000000LSR',
            'quits_rate': 'JTS000000000000000QUR',
            'labor_force_participation': 'LNS11300000',
            'employment_population_ratio': 'LNS12300000'
        }
        
    def get_bls_data(self, series_id: str, start_year: int = 2020, 
                    end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Get data from Bureau of Labor Statistics API
        
        Args:
            series_id: BLS series identifier
            start_year: Starting year for data
            end_year: Ending year for data (default: current year)
            
        Returns:
            DataFrame with BLS data
        """
        if end_year is None:
            end_year = datetime.now().year
            
        cache_file = self.cache_dir / f"bls_{series_id}_{start_year}_{end_year}.pkl"
        
        # Check cache (daily cache)
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(days=1):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            logger.info(f"Fetching BLS data for series {series_id}")
            
            # Prepare request data
            data = {
                "seriesid": [series_id],
                "startyear": str(start_year),
                "endyear": str(end_year)
            }
            
            response = requests.post(
                self.bls_api_url, 
                data=json.dumps(data),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                json_data = response.json()
                
                if json_data['status'] == 'REQUEST_SUCCEEDED':
                    series_data = json_data['Results']['series'][0]['data']
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(series_data)
                    df['date'] = pd.to_datetime(df['year'] + df['period'].str.replace('M', '-'), 
                                              format='%Y-%m', errors='coerce')
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df[['date', 'value']].dropna()
                    df = df.sort_values('date').reset_index(drop=True)
                    
                    # Cache the results
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                    
                    logger.info(f"Retrieved {len(df)} data points for {series_id}")
                    return df
                else:
                    logger.error(f"BLS API error: {json_data.get('message', 'Unknown error')}")
                    return pd.DataFrame()
            else:
                logger.error(f"HTTP error {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching BLS data: {e}")
            return pd.DataFrame()
    
    def get_all_bls_indicators(self, start_year: int = 2020) -> Dict[str, pd.DataFrame]:
        """Get all standard employment indicators from BLS"""
        indicators = {}
        
        for indicator_name, series_id in self.bls_series.items():
            df = self.get_bls_data(series_id, start_year)
            if not df.empty:
                indicators[indicator_name] = df
            time.sleep(0.5)  # Rate limiting
            
        return indicators


class JobPostingsScraper:
    """Scraper for job postings data (simulated/mock data for demo)"""
    
    def __init__(self, cache_dir: str = "data/employment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Skill categories for analysis
        self.skill_categories = {
            'ai_ml': ['machine learning', 'artificial intelligence', 'ai', 'ml', 
                     'deep learning', 'neural network', 'data science', 'nlp', 
                     'computer vision', 'tensorflow', 'pytorch'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 
                     'devops', 'microservices'],
            'programming': ['python', 'java', 'javascript', 'c++', 'sql', 'react', 
                          'node.js', 'angular', 'django'],
            'leadership': ['team lead', 'manager', 'director', 'vp', 'cto', 'ceo',
                          'leadership', 'management', 'strategic planning']
        }
        
    def generate_mock_job_data(self, companies: List[str], 
                              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate mock job postings data for demonstration
        In a real implementation, this would scrape actual job sites
        """
        logger.info("Generating mock job postings data")
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        job_data = []
        
        for company in companies:
            for date in date_range:
                # Simulate varying job posting activity
                base_postings = np.random.poisson(5)  # Average 5 postings per day
                
                # Add weekly and seasonal patterns
                day_of_week = date.weekday()
                if day_of_week >= 5:  # Weekend
                    base_postings = int(base_postings * 0.3)
                
                # Generate individual job postings
                for _ in range(base_postings):
                    # Random job title and description
                    job_titles = [
                        'Software Engineer', 'Data Scientist', 'Product Manager',
                        'Marketing Manager', 'Sales Representative', 'DevOps Engineer',
                        'UI/UX Designer', 'Business Analyst', 'Project Manager'
                    ]
                    
                    title = np.random.choice(job_titles)
                    
                    # Generate description with skills
                    skills = []
                    for category, skill_list in self.skill_categories.items():
                        if np.random.random() < 0.3:  # 30% chance of including category
                            skills.extend(np.random.choice(skill_list, 
                                        size=np.random.randint(1, 3), 
                                        replace=False))
                    
                    description = f"{title} position requiring {', '.join(skills[:5])}"
                    
                    job_data.append({
                        'company_id': company,
                        'posting_date': date,
                        'title': title,
                        'description': description,
                        'required_skills': ', '.join(skills),
                        'location': 'Various',
                        'job_type': np.random.choice(['Full-time', 'Part-time', 'Contract'])
                    })
        
        return pd.DataFrame(job_data)
    
    def analyze_skill_demand(self, job_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze skill demand trends from job postings
        
        Args:
            job_df: DataFrame with job postings data
            
        Returns:
            DataFrame with skill demand analysis by date and company
        """
        skill_analysis = []
        
        # Group by company and date
        for (company, date), group in job_df.groupby(['company_id', 'posting_date']):
            all_descriptions = ' '.join(group['description'].fillna('').str.lower())
            
            analysis = {
                'company_id': company,
                'date': date,
                'total_postings': len(group),
            }
            
            # Count skills by category
            for category, skills in self.skill_categories.items():
                skill_count = sum(all_descriptions.count(skill) for skill in skills)
                analysis[f'{category}_demand'] = skill_count
                analysis[f'{category}_density'] = skill_count / len(group) if len(group) > 0 else 0
            
            skill_analysis.append(analysis)
        
        return pd.DataFrame(skill_analysis)


class LayoffDataLoader:
    """Loader for layoff and downsizing data"""
    
    def __init__(self, cache_dir: str = "data/employment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_mock_layoff_data(self, companies: List[str], 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate mock layoff data for demonstration
        In production, this would integrate with WARN notices and news sources
        """
        logger.info("Generating mock layoff data")
        
        layoff_data = []
        
        for company in companies:
            # Simulate occasional layoffs (rare events)
            if np.random.random() < 0.1:  # 10% chance of layoffs
                layoff_date = start_date + timedelta(
                    days=np.random.randint(0, (end_date - start_date).days)
                )
                
                employees_affected = np.random.randint(10, 500)
                
                layoff_data.append({
                    'company_id': company,
                    'layoff_date': layoff_date,
                    'employees_affected': employees_affected,
                    'reason': np.random.choice([
                        'restructuring', 'cost reduction', 'market conditions', 
                        'strategic realignment', 'automation'
                    ]),
                    'department': np.random.choice([
                        'engineering', 'sales', 'marketing', 'operations', 'support'
                    ])
                })
        
        return pd.DataFrame(layoff_data)


class EmploymentProcessor:
    """Process and aggregate employment data for modeling"""
    
    def __init__(self):
        self.skill_weights = {
            'ai_ml': 1.5,      # Higher weight for AI/ML skills
            'cloud': 1.2,      # Cloud skills are in demand
            'programming': 1.0, # Standard programming skills
            'leadership': 0.8   # Leadership skills
        }
        
    def calculate_hiring_velocity(self, job_df: pd.DataFrame, 
                                window_days: int = 30) -> pd.DataFrame:
        """
        Calculate hiring velocity (rate of change in job postings)
        
        Args:
            job_df: DataFrame with job postings
            window_days: Rolling window for velocity calculation
            
        Returns:
            DataFrame with hiring velocity metrics
        """
        # Group by company and date
        daily_postings = job_df.groupby(['company_id', 'posting_date']).size().reset_index()
        daily_postings.columns = ['company_id', 'date', 'postings']
        
        # Calculate rolling averages and velocity
        velocity_data = []
        
        for company in daily_postings['company_id'].unique():
            company_data = daily_postings[daily_postings['company_id'] == company].copy()
            company_data = company_data.sort_values('date')
            
            # Rolling average
            company_data['postings_ma'] = company_data['postings'].rolling(
                window=window_days, min_periods=1
            ).mean()
            
            # Velocity (rate of change)
            company_data['hiring_velocity'] = company_data['postings_ma'].pct_change(
                periods=window_days
            ).fillna(0)
            
            velocity_data.append(company_data)
        
        return pd.concat(velocity_data, ignore_index=True)
    
    def calculate_employment_signals(self, job_df: pd.DataFrame, 
                                   layoff_df: pd.DataFrame, 
                                   skill_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive employment signals for each company and date
        
        Args:
            job_df: Job postings data
            layoff_df: Layoff data
            skill_df: Skill demand analysis
            
        Returns:
            DataFrame with employment signals
        """
        # Get date range
        start_date = min(job_df['posting_date'].min(), skill_df['date'].min())
        end_date = max(job_df['posting_date'].max(), skill_df['date'].max())
        
        companies = job_df['company_id'].unique()
        
        signals = []
        
        for company in companies:
            company_job_data = job_df[job_df['company_id'] == company]
            company_layoff_data = layoff_df[layoff_df['company_id'] == company]
            company_skill_data = skill_df[skill_df['company_id'] == company]
            
            # Daily aggregation
            daily_jobs = company_job_data.groupby('posting_date').size()
            daily_skills = company_skill_data.set_index('date')
            
            date_range = pd.date_range(start_date, end_date, freq='D')
            
            for date in date_range:
                # Job posting metrics
                jobs_30d = daily_jobs[daily_jobs.index >= (date - timedelta(days=30))].sum()
                jobs_90d = daily_jobs[daily_jobs.index >= (date - timedelta(days=90))].sum()
                
                # Layoff metrics
                layoffs_30d = company_layoff_data[
                    company_layoff_data['layoff_date'] >= (date - timedelta(days=30))
                ]['employees_affected'].sum()
                
                layoffs_90d = company_layoff_data[
                    company_layoff_data['layoff_date'] >= (date - timedelta(days=90))
                ]['employees_affected'].sum()
                
                # Hiring velocity
                if jobs_90d > 30:  # Need sufficient data
                    recent_avg = jobs_30d / 30
                    older_avg = (jobs_90d - jobs_30d) / 60
                    hiring_velocity = (recent_avg - older_avg) / (older_avg + 1e-6)
                else:
                    hiring_velocity = 0.0
                
                # Skill demand (from recent skill analysis)
                recent_skills = daily_skills[daily_skills.index <= date].tail(1)
                
                if not recent_skills.empty:
                    ai_ml_demand = recent_skills['ai_ml_demand'].iloc[0]
                    cloud_demand = recent_skills['cloud_demand'].iloc[0]
                    programming_demand = recent_skills['programming_demand'].iloc[0]
                    leadership_demand = recent_skills['leadership_demand'].iloc[0]
                else:
                    ai_ml_demand = cloud_demand = programming_demand = leadership_demand = 0
                
                # Calculate composite employment score
                employment_score = self._calculate_employment_score(
                    jobs_30d, jobs_90d, layoffs_30d, layoffs_90d, hiring_velocity,
                    ai_ml_demand, cloud_demand, programming_demand, leadership_demand
                )
                
                signals.append({
                    'company_id': company,
                    'date': date,
                    'jobs_30d': jobs_30d,
                    'jobs_90d': jobs_90d,
                    'layoffs_30d': layoffs_30d,
                    'layoffs_90d': layoffs_90d,
                    'hiring_velocity': hiring_velocity,
                    'ai_ml_demand': ai_ml_demand,
                    'cloud_demand': cloud_demand,
                    'programming_demand': programming_demand,
                    'leadership_demand': leadership_demand,
                    'employment_score': employment_score
                })
        
        return pd.DataFrame(signals)
    
    def _calculate_employment_score(self, jobs_30d: int, jobs_90d: int, 
                                  layoffs_30d: int, layoffs_90d: int, 
                                  hiring_velocity: float, ai_ml_demand: float,
                                  cloud_demand: float, programming_demand: float,
                                  leadership_demand: float) -> float:
        """Calculate a composite employment health score (0-1)"""
        
        # Hiring activity score
        hiring_score = min(jobs_30d / 100, 1.0)  # Normalize to 0-1
        
        # Layoff impact (negative)
        layoff_penalty = min(layoffs_30d / 1000, 1.0)
        
        # Velocity score
        velocity_score = max(0, min(hiring_velocity, 1.0))
        
        # Skill demand score (weighted)
        skill_score = (
            ai_ml_demand * self.skill_weights['ai_ml'] +
            cloud_demand * self.skill_weights['cloud'] +
            programming_demand * self.skill_weights['programming'] +
            leadership_demand * self.skill_weights['leadership']
        ) / 100  # Normalize
        
        skill_score = min(skill_score, 1.0)
        
        # Composite score
        composite = (
            0.3 * hiring_score +
            0.2 * velocity_score +
            0.3 * skill_score -
            0.2 * layoff_penalty
        )
        
        return max(0.0, min(1.0, composite))
    
    def align_with_stock_data(self, stock_df: pd.DataFrame, 
                            employment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align employment data with stock trading days
        
        Args:
            stock_df: DataFrame with stock data
            employment_df: DataFrame with employment signals
            
        Returns:
            DataFrame with aligned stock and employment data
        """
        # Ensure date columns are properly formatted
        if isinstance(stock_df.index, pd.DatetimeIndex):
            stock_df = stock_df.reset_index()
            stock_df['date'] = stock_df['Date'].dt.date
        else:
            stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
        
        employment_df['date'] = pd.to_datetime(employment_df['date']).dt.date
        
        # Merge data
        merged = stock_df.merge(
            employment_df, 
            left_on=['Symbol', 'date'], 
            right_on=['company_id', 'date'], 
            how='left'
        )
        
        # Fill missing employment data
        employment_columns = [col for col in employment_df.columns 
                            if col not in ['company_id', 'date']]
        merged[employment_columns] = merged[employment_columns].fillna(0)
        
        return merged