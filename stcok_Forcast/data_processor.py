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
        # Try to load SP500 data if available
        try:
            self.sp500_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "sp500_companies.csv"))
            self.sectors = self.sp500_data['Sector'].unique().tolist()
            self.sector_stocks = {sector: self.sp500_data[self.sp500_data['Sector'] == sector]['Symbol'].tolist() 
                                for sector in self.sectors}
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("Warning: SP500 company data not found")
            self.sp500_data = pd.DataFrame()
            self.sectors = []
            self.sector_stocks = {}
        
    def fetch_historical_data(self, symbol: str, period: str = "7y") -> pd.DataFrame:
        """
        Fetch historical price data for a given stock symbol from local CSV files
        
        Parameters:
        -----------
        symbol : str
            The stock symbol (e.g., 'AAPL', 'NVDA')
        period : str
            The period of data to fetch (not used when reading from CSV, kept for compatibility)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing historical price data with columns:
            date, close, volume, open, high, low
        """
        import os
        
        # Define path to CSV files
        csv_path = os.path.join(self.data_dir, "historical_prices", f'{symbol}.csv')
        
        try:
            # Read CSV file into DataFrame
            df = pd.read_csv(csv_path)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Rename columns to match expected format
            df.rename(columns={
                'date': 'Date',
                'close': 'Close',
                'volume': 'Volume',
                'open': 'Open',
                'high': 'High',
                'low': 'Low'
            }, inplace=True, errors='ignore')
            
            return df
        except FileNotFoundError:
            print(f"No historical data found for symbol: {symbol}. Check if {csv_path} exists.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading historical data for {symbol}: {str(e)}")
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
        result['MACD'] = macd['MACD_12_26_9']
        result['MACD_Signal'] = macd['MACDs_12_26_9']
        result['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(result['Close'], length=20)
        result['BB_Upper'] = bb['BBU_20_2.0']
        result['BB_Middle'] = bb['BBM_20_2.0']
        result['BB_Lower'] = bb['BBL_20_2.0']
        
        # Stochastic Oscillator
        stoch = ta.stoch(result['High'], result['Low'], result['Close'])
        result['Stoch_k'] = stoch['STOCHk_14_3_3']
        result['Stoch_d'] = stoch['STOCHd_14_3_3']
        
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
            
        # Ensure column names for merging
        date_col_df = 'Date' if 'Date' in df.columns else 'date'
        date_col_sentiment = 'date'  # Standardize on 'date' for sentiment data
        
        # Make a copy of the dataframes to avoid warnings
        df_copy = df.copy()
        sentiment_df_copy = sentiment_df.copy()
        
        # Ensure date columns are datetime
        df_copy[date_col_df] = pd.to_datetime(df_copy[date_col_df])
        sentiment_df_copy[date_col_sentiment] = pd.to_datetime(sentiment_df_copy[date_col_sentiment])
        
        # Rename date column in df if necessary to match sentiment_df
        if date_col_df != date_col_sentiment:
            df_copy.rename(columns={date_col_df: date_col_sentiment}, inplace=True)
        
        # Merge dataframes on date
        result = pd.merge(df_copy, sentiment_df_copy, on=date_col_sentiment, how='left')
        
        # Fill missing sentiment values with forward fill then backward fill
        result['sentiment_score'] = result['sentiment_score'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
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
                                      result['sentiment_score'].rolling(window=20).std()).fillna(0)
        
        return result
    
    def get_sector_data(self, sector: str, period: str = "7y") -> Dict[str, pd.DataFrame]:
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
    
    def process_sentiment_data(self, ticker: str) -> pd.DataFrame:
        """
        Process sentiment data for a given ticker symbol
        
        Args:
            ticker: The ticker symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with processed sentiment data
        """
        # Define path to sentiment data
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sentiment_data_path = os.path.join(base_dir, 'sentiment_data', f'{ticker}.csv')
        
        # Load sentiment data
        sentiment_data = pd.read_csv(sentiment_data_path)
        print(f"Loaded sentiment data with {len(sentiment_data)} rows")
        
        # Convert datetime to date
        sentiment_data['datetime'] = pd.to_datetime(sentiment_data['datetime'])
        sentiment_data['date'] = pd.to_datetime(sentiment_data['datetime'].dt.date)  # Convert to datetime for easier merging
        
        # Sort data chronologically (from oldest to newest)
        sentiment_data = sentiment_data.sort_values('datetime')
        
        # Group by date
        grouped = sentiment_data.groupby('date')
        
        # Initialize lists to store results
        dates = []
        pos_scores = []
        neg_scores = []
        neu_scores = []
        sentiment_scores = []
        final_sentiments = []
        
        # Process each date group
        for date, group in grouped:
            # Get positive, negative, and neutral groups
            pos_group = group[group['sentiment_class'] == 'positive']
            neg_group = group[group['sentiment_class'] == 'negative']
            neu_group = group[group['sentiment_class'] == 'neutral']
            
            # Calculate average sentiment scores for each class
            avg_pos_score = pos_group['sentiment_score'].mean() if not pos_group.empty else 0
            avg_neg_score = neg_group['sentiment_score'].mean() if not neg_group.empty else 0
            avg_neu_score = neu_group['sentiment_score'].mean() if not neu_group.empty else 0
            
            # Count occurrences of each sentiment class
            pos_count = len(pos_group)
            neg_count = len(neg_group)
            neu_count = len(neu_group)
            
            # Calculate weighted sentiment (positive sentiment is positive, negative is negative)
            pos_weighted = pos_count * avg_pos_score if not np.isnan(avg_pos_score) else 0
            neg_weighted = neg_count * (-avg_neg_score) if not np.isnan(avg_neg_score) else 0  # Negative sentiment gets negative weight
            neu_weighted = neu_count * 0  # Neutral has no impact
            
            # Calculate overall sentiment score for the day
            total_articles = pos_count + neg_count + neu_count
            if total_articles > 0:
                total_sentiment = (pos_weighted + neg_weighted + neu_weighted) / total_articles
            else:
                total_sentiment = 0
                
            # Determine final sentiment class (highest count wins)
            if pos_count >= neg_count and pos_count >= neu_count:
                final_sentiment = 'positive'
            elif neg_count >= pos_count and neg_count >= neu_count:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'
            
            # Append results
            dates.append(date)
            pos_scores.append(avg_pos_score)
            neg_scores.append(avg_neg_score)
            neu_scores.append(avg_neu_score)
            sentiment_scores.append(total_sentiment)
            final_sentiments.append(final_sentiment)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'date': dates,
            'positive_score': pos_scores,
            'negative_score': neg_scores,
            'neutral_score': neu_scores,
            'sentiment_score': sentiment_scores,  # Add overall sentiment score
            'sentiment': final_sentiments
        })
        
        # Sort result by date (oldest to newest)
        result_df = result_df.sort_values('date')
        
        return result_df
    
    def merge_price_and_sentiment(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge price data with sentiment data based on date.
        
        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            
        Returns:
            DataFrame with merged price and sentiment data
        """
        # Ensure date columns are datetime
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        # Merge dataframes on date
        merged_df = pd.merge(price_df, sentiment_df, on='date', how='left')
        
        return merged_df
