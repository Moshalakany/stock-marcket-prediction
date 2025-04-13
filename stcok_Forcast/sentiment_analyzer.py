import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import os
import datetime
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analyzer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize the sentiment analyzer with custom API."""
        self.api_url = api_url
        self.initialized = True
        self.news_sources = {
            'seeking_alpha': 'https://seekingalpha.com/symbol/{symbol}/news',
            'yahoo_finance': 'https://finance.yahoo.com/quote/{symbol}/news',
            'market_watch': 'https://www.marketwatch.com/investing/stock/{symbol}'
        }
        logger.info(f"Using custom sentiment analysis API at {api_url}")
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a text using custom API."""
        try:
            response = requests.post(
                f"{self.api_url}/sentiment/analyze",
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Map API response to the expected format
            scores = result.get('scores', {})
            sentiment_score = scores.get('positive', 0) - scores.get('negative', 0)
            
            return {
                'negative': float(scores.get('negative', 0)),
                'neutral': float(scores.get('neutral', 0)),
                'positive': float(scores.get('positive', 0)),
                'sentiment_score': float(sentiment_score)
            }
        except Exception as e:
            logger.error(f"Error calling sentiment API: {e}")
            return {
                'negative': 0.0,
                'neutral': 1.0,
                'positive': 0.0,
                'sentiment_score': 0.0
            }
    
    def batch_analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts using the custom API in batch mode."""
        try:
            response = requests.post(
                f"{self.api_url}/sentiment/batch",
                json={"texts": texts},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            sentiment_results = []
            for item in result.get('results', []):
                scores = item.get('scores', {})
                sentiment_score = scores.get('positive', 0) - scores.get('negative', 0)
                
                sentiment_results.append({
                    'negative': float(scores.get('negative', 0)),
                    'neutral': float(scores.get('neutral', 0)),
                    'positive': float(scores.get('positive', 0)),
                    'sentiment_score': float(sentiment_score)
                })
            
            return sentiment_results
        except Exception as e:
            logger.error(f"Error calling batch sentiment API: {e}")
            # Return default sentiment for each text
            return [{
                'negative': 0.0,
                'neutral': 1.0,
                'positive': 0.0,
                'sentiment_score': 0.0
            } for _ in texts]
    
    def scrape_news(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """Scrape news for a given stock symbol."""
        news_items = []
        today = datetime.datetime.now()
        cutoff_date = today - datetime.timedelta(days=days_back)
        
        # Try to get news from Yahoo Finance via yfinance API first
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            if news:
                for item in news:
                    news_date = datetime.datetime.fromtimestamp(item['providerPublishTime'])
                    if news_date >= cutoff_date:
                        news_items.append({
                            'title': item['title'],
                            'date': news_date.strftime('%Y-%m-%d'),
                            'text': item.get('summary', ''),
                            'source': 'Yahoo Finance'
                        })
                logger.info(f"Retrieved {len(news_items)} news items from Yahoo Finance for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to get news from Yahoo Finance for {symbol}: {e}")
        
        # If we couldn't get enough news from the API, try web scraping
        if len(news_items) < 5:
            for source_name, url_template in self.news_sources.items():
                try:
                    url = url_template.format(symbol=symbol.lower())
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Different parsing logic for different sources
                        if source_name == 'seeking_alpha':
                            articles = soup.find_all('div', class_='title')
                            for article in articles[:10]:  # Limit to 10 articles
                                title_elem = article.find('a')
                                if title_elem:
                                    news_items.append({
                                        'title': title_elem.text.strip(),
                                        'date': datetime.datetime.now().strftime('%Y-%m-%d'),  # Approximate date
                                        'text': title_elem.text.strip(),
                                        'source': 'Seeking Alpha'
                                    })
                        # Add parsing for other sources as needed
                        
                    time.sleep(2)  # Be polite to servers
                except Exception as e:
                    logger.warning(f"Failed to scrape news from {source_name} for {symbol}: {e}")
        
        return news_items
    
    def analyze_sentiment_for_stock(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Analyze sentiment for a stock over the given period."""
        news_items = self.scrape_news(symbol, days_back)
        
        sentiment_data = []
        
        # Use batch processing if we have multiple news items
        if len(news_items) > 1:
            texts = [item['text'] if item['text'] else item['title'] for item in news_items]
            try:
                batch_results = self.batch_analyze_texts(texts)
                
                for i, (item, sentiment) in enumerate(zip(news_items, batch_results)):
                    sentiment_data.append({
                        'date': item['date'],
                        'title': item['title'],
                        'source': item['source'],
                        'negative': sentiment['negative'],
                        'neutral': sentiment['neutral'],
                        'positive': sentiment['positive'],
                        'sentiment_score': sentiment['sentiment_score']
                    })
            except Exception as e:
                logger.error(f"Error in batch sentiment analysis: {e}")
        else:
            # Process one by one if only one item
            for item in news_items:
                try:
                    sentiment = self.analyze_text(item['text'] if item['text'] else item['title'])
                    sentiment_data.append({
                        'date': item['date'],
                        'title': item['title'],
                        'source': item['source'],
                        'negative': sentiment['negative'],
                        'neutral': sentiment['neutral'],
                        'positive': sentiment['positive'],
                        'sentiment_score': sentiment['sentiment_score']
                    })
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for news item: {e}")
        
        # Create DataFrame and aggregate by date
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Aggregate sentiments by date
            daily_sentiment = df.groupby(df['date'].dt.date).agg({
                'negative': 'mean',
                'neutral': 'mean',
                'positive': 'mean',
                'sentiment_score': 'mean',
                'title': 'count'
            }).reset_index()
            
            daily_sentiment = daily_sentiment.rename(columns={'title': 'news_count'})
            daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment = daily_sentiment.drop('date', axis=1)
            
            # Fill in missing dates
            date_range = pd.date_range(
                start=pd.to_datetime(daily_sentiment['Date'].min()),
                end=pd.to_datetime(daily_sentiment['Date'].max())
            )
            daily_sentiment = daily_sentiment.set_index('Date').reindex(date_range).fillna(method='ffill').reset_index()
            daily_sentiment = daily_sentiment.rename(columns={'index': 'Date'})
            
            return daily_sentiment
        
        # Return empty DataFrame with correct columns if no data
        return pd.DataFrame(columns=['Date', 'negative', 'neutral', 'positive', 'sentiment_score', 'news_count'])
