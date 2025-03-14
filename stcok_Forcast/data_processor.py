import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Union, Optional
import os
import yfinance as yf
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        """Initialize the data processor."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.sp500_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "sp500_companies.csv"))
        self.sectors = self.sp500_data['Sector'].unique().tolist()
        self.sector_stocks = {sector: self.sp500_data[self.sp500_data['Sector'] == sector]['Symbol'].tolist() 
                              for sector in self.sectors}
        
    def fetch_historical_data(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical price data for a given symbol."""
        try:
            stock_data = yf.Ticker(symbol).history(period=period)
            if not stock_data.empty:
                stock_data.index = pd.to_datetime(stock_data.index)
                stock_data = stock_data.rename_axis('Date').reset_index()
                return stock_data
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators for a price dataframe."""
        if df.empty:
            return df
            
        # Ensure we have the necessary columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Create a copy to avoid warnings
        result = df.copy()
        
        # Simple Moving Averages
        result['SMA_5'] = ta.sma(result['Close'], length=5)
        result['SMA_20'] = ta.sma(result['Close'], length=20)
        result['SMA_50'] = ta.sma(result['Close'], length=50)
        result['SMA_200'] = ta.sma(result['Close'], length=200)
        
        # Exponential Moving Averages
        result['EMA_5'] = ta.ema(result['Close'], length=5)
        result['EMA_20'] = ta.ema(result['Close'], length=20)
        result['EMA_50'] = ta.ema(result['Close'], length=50)
        
        # RSI
        result['RSI_14'] = ta.rsi(result['Close'], length=14)
        
        # MACD
        macd = ta.macd(result['Close'])
        result = pd.concat([result, macd], axis=1)
        
        # Bollinger Bands
        bbands = ta.bbands(result['Close'], length=20)
        result = pd.concat([result, bbands], axis=1)
        
        # Volume-based indicators
        result['OBV'] = ta.obv(result['Close'], result['Volume'])
        result['Volume_SMA_20'] = ta.sma(result['Volume'], length=20)
        result['Price_Volume_Ratio'] = result['Close'] / result['Volume']
        
        # Volatility
        result['ATR_14'] = ta.atr(result['High'], result['Low'], result['Close'], length=14)
        
        # Momentum
        result['ROC_10'] = ta.roc(result['Close'], length=10)
        result['MOM_10'] = ta.mom(result['Close'], length=10)
        
        return result
        
    def compute_sentiment_indicators(self, df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment-based indicators."""
        if df.empty or sentiment_df.empty:
            return df
            
        # Ensure sentiment data has date and sentiment columns
        if not all(col in sentiment_df.columns for col in ['Date', 'sentiment_score']):
            raise ValueError("Sentiment DataFrame must contain 'Date' and 'sentiment_score' columns")
        
        # Merge price data with sentiment data
        result = pd.merge(df, sentiment_df, on='Date', how='left')
        
        # Fill missing sentiment values
        result['sentiment_score'] = result['sentiment_score'].fillna(method='ffill')
        
        # Compute sentiment indicators
        result['Sentiment_SMA_5'] = ta.sma(result['sentiment_score'], length=5)
        result['Sentiment_SMA_10'] = ta.sma(result['sentiment_score'], length=10)
        
        # Compute sentiment momentum (rate of change)
        result['Sentiment_Momentum'] = ta.roc(result['sentiment_score'], length=5)
        
        # Compute sentiment volatility
        result['Sentiment_Volatility'] = result['sentiment_score'].rolling(window=10).std()
        
        # Compute sentiment z-score
        result['Sentiment_ZScore'] = ((result['sentiment_score'] - 
                                      result['sentiment_score'].rolling(window=20).mean()) / 
                                      result['sentiment_score'].rolling(window=20).std())
        
        return result
    
    def get_sector_data(self, sector: str, period: str = "5y") -> Dict[str, pd.DataFrame]:
        """Get processed data for all stocks in a sector."""
        stocks = self.sector_stocks.get(sector, [])
        sector_data = {}
        
        for symbol in stocks:
            df = self.fetch_historical_data(symbol, period)
            if not df.empty:
                sector_data[symbol] = self.compute_technical_indicators(df)
        
        return sector_data
        
    def prepare_dataset(self, df: pd.DataFrame, window_size: int = 20, prediction_horizon: int = 5) -> tuple:
        """Prepare dataset for training with sliding window approach."""
        if df.empty:
            return None, None
            
        # Select features and target
        features = [col for col in df.columns if col not in ['Date', 'Stock Splits', 'Dividends']]
        X, y = [], []
        
        for i in range(len(df) - window_size - prediction_horizon + 1):
            X.append(df[features].iloc[i:i+window_size].values)
            # Target is the price change in the prediction horizon
            price_change = (df['Close'].iloc[i+window_size+prediction_horizon-1] / 
                           df['Close'].iloc[i+window_size-1]) - 1
            y.append(price_change)
            
        return np.array(X), np.array(y)
