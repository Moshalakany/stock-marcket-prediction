import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class dataPreprocessor:
    def __init__(self, historical_data,sentiment_data,symbol):
        self.symbol = symbol
        self.historical_data = historical_data
        self.sentiment_data = sentiment_data
    
    def snetiment_data_preprocessor(self):
        data = pd.read_csv(self.sentiment_data)
        data['datetime'] = pd.to_datetime(data['datetime'], format='mixed')
        data['datetime'] = data['datetime'].dt.date
        data.set_index('datetime', inplace=True)
        # Group by date and sentiment class to calculate aggregate sentiment
        # First, create a group by date and sentiment class
        grouped = data.groupby([data.index, 'sentiment_class'])

        # Calculate the sum and count for each sentiment class per day
        class_scores = grouped['sentiment_score'].agg(['sum', 'count']).unstack(fill_value=0)

        # Get the totals for each sentiment class per day
        class_totals = class_scores['sum']
        class_counts = class_scores['count']

        # Find the class with the highest total score for each day
        winning_class = class_totals.idxmax(axis=1)

        # Calculate the average score for the winning class
        daily_sentiment = pd.DataFrame(index=class_totals.index)
        daily_sentiment['sentiment_class'] = winning_class
        daily_sentiment['count'] = [class_counts.loc[idx, cls] for idx, cls in zip(daily_sentiment.index, winning_class)]
        daily_sentiment['total_score'] = [class_totals.loc[idx, cls] for idx, cls in zip(daily_sentiment.index, winning_class)]
        daily_sentiment['sentiment_score'] = daily_sentiment['total_score'] / daily_sentiment['count']
        daily_sentiment['positive_count'] = class_counts['positive']
        daily_sentiment['negative_count'] = class_counts['negative']
        daily_sentiment['neutral_count'] = class_counts['neutral']

        # Calculate the average score for the winning class

        # Convert index to datetime for proper date handling
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        # replace the sentiment class with the winning class and the sentiment score with the weighted average
        data['sentiment_class'] = daily_sentiment['sentiment_class']
        # Create a complete date range from the first to last date in our data
        date_range = pd.date_range(start=daily_sentiment.index.min(), end=daily_sentiment.index.max(), freq='D')

        # Reindex the data to include all dates in the range
        daily_sentiment = daily_sentiment.reindex(date_range)

        # Forward fill missing days with the previous day's data
        daily_sentiment['sentiment_score'] = daily_sentiment['sentiment_score'].fillna(method='ffill')
        daily_sentiment['sentiment_class'] = daily_sentiment['sentiment_class'].fillna(method='ffill')
        # First, map sentiment classes to numeric values
        # Assuming sentiment_class contains values like 'positive', 'negative', 'neutral'
        class_map = {
            'positive': 1,
            'negative': -1,
            'neutral': 0.01
            # Add any other classes that might exist in your data
        }

        # Create a weighted sentiment that combines class and score
        daily_sentiment['weighted_sentiment'] = daily_sentiment['sentiment_class'].map(class_map) * daily_sentiment['sentiment_score']

        # Calculate indicators using the weighted sentiment
        # 1. Simple Moving Averages (5-day and 10-day)
        daily_sentiment['Weighted_SMA_5'] = daily_sentiment['weighted_sentiment'].rolling(window=5).mean()
        daily_sentiment['Weighted_SMA_10'] = daily_sentiment['weighted_sentiment'].rolling(window=10).mean()

        # 2. Momentum (difference between current value and 5 days ago)
        daily_sentiment['Weighted_Momentum'] = daily_sentiment['weighted_sentiment'] - daily_sentiment['weighted_sentiment'].shift(5)

        # 3. Volatility (10-day rolling standard deviation)
        daily_sentiment['Weighted_Volatility'] = daily_sentiment['weighted_sentiment'].rolling(window=10).std()

        # 4. Crossing signals (when 5-day SMA crosses 10-day SMA)
        daily_sentiment['Signal_Cross'] = np.where(
            daily_sentiment['Weighted_SMA_5'] > daily_sentiment['Weighted_SMA_10'], 1, 
            np.where(daily_sentiment['Weighted_SMA_5'] < daily_sentiment['Weighted_SMA_10'], -1, 0)
        )
        daily_sentiment['count'] = daily_sentiment['count'].fillna(0)
        daily_sentiment['total_score'] = daily_sentiment['total_score'].fillna(0)
        daily_sentiment['positive_count'] = daily_sentiment['positive_count'].fillna(0)
        daily_sentiment['negative_count'] = daily_sentiment['negative_count'].fillna(0)
        daily_sentiment['neutral_count'] = daily_sentiment['neutral_count'].fillna(0)
        daily_sentiment['date'] = daily_sentiment.index
        daily_sentiment.index=daily_sentiment['date']
        daily_sentiment=daily_sentiment.drop(columns='date')
        #drop the first 50 rows
        daily_sentiment = daily_sentiment[50:]
        return daily_sentiment
    def historical_data_preprocessor(self):
        historical_data = pd.read_csv(self.historical_data)
        historical_data['date'] = pd.to_datetime(historical_data['date'], format='mixed')
        # Then set it as index
        historical_data.set_index('date', inplace=True)
        date_range = pd.date_range(start=historical_data.index.min(), end=historical_data.index.max(), freq='D')
        historical_data = historical_data.reindex(date_range)
        historical_data = historical_data.fillna(method='ffill')
        historical_data.head(20)
        #reverse the sort of the data
        historical_data = historical_data.sort_index()
        historical_data['date'] = historical_data.index
        historical_data.index=historical_data['date']
        historical_data=historical_data.drop(columns='date')
        # Calculate technical indicators for historical data
        # Simple Moving Averages (SMA)
        historical_data['SMA_5'] = historical_data['close'].rolling(window=5).mean()
        historical_data['SMA_10'] = historical_data['close'].rolling(window=10).mean()
        # Exponential Moving Averages (EMA)
        historical_data['EMA_5'] = historical_data['close'].ewm(span=5, adjust=False).mean()
        historical_data['EMA_10'] = historical_data['close'].ewm(span=10, adjust=False).mean()

        # MACD (Moving Average Convergence Divergence)
        historical_data['EMA_12'] = historical_data['close'].ewm(span=12, adjust=False).mean()
        historical_data['EMA_26'] = historical_data['close'].ewm(span=26, adjust=False).mean()
        historical_data['MACD'] = historical_data['EMA_12'] - historical_data['EMA_26']
        historical_data['MACD_Signal'] = historical_data['MACD'].ewm(span=9, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = historical_data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        historical_data['RSI_14'] = 100 - (100 / (1 + rs))

        # Round the calculated indicators to 4 decimal places
        for col in ['SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'MACD', 'MACD_Signal', 'RSI_14']:
            historical_data[col] = historical_data[col].round(4)
        historical_data = historical_data[50:]
        return historical_data
    def merge_data(self):
        historical_data = self.historical_data_preprocessor()
        sentiment_data = self.snetiment_data_preprocessor()
        merged_data = pd.merge(historical_data, sentiment_data, on='date', how='inner')
        merged_data.to_csv(f'MergedData/{self.symbol}_merged.csv', index=True)
