import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os
from datetime import timedelta, datetime

class StockForecastingModel:
    def __init__(self, stock_symbol, data_path=None, output_dir=None, seq_length=30, pred_days=7):
        """
        Initialize the stock forecasting model
        
        Args:
            stock_symbol (str): Stock symbol (e.g., 'AAPL')
            data_path (str, optional): Path to the merged data CSV file
            output_dir (str, optional): Directory to save outputs
            seq_length (int): Number of days to use as input sequence
            pred_days (int): Number of days to predict ahead
        """
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.stock_symbol = stock_symbol
        
        # Set default data path if not provided
        if data_path is None:
            self.data_path = f'../MergedData/{stock_symbol}_merged.csv'
        else:
            self.data_path = data_path
        
        # Set output directory
        if output_dir is None:
            self.output_dir = f'output/{stock_symbol}'
        else:
            self.output_dir = f'{output_dir}/{stock_symbol}'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set model parameters
        self.seq_length = seq_length
        self.pred_days = pred_days
        
        # Initialize model attributes
        self.df = None
        self.X = None
        self.y = None
        self.scaler_X = None
        self.scaler_y = None
        self.model = None

    def load_data(self):
        """Load and preprocess the dataset"""
        df = pd.read_csv(self.data_path)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Sort by date (just to be sure)
        df.sort_index(inplace=True)
        start_date = '2020-01-01'
        df = df[df.index >= start_date]
        
        self.df = df
        print(f"Dataset loaded with shape: {df.shape}")
        return df
    
    def handle_missing_values(self):
        """Handle missing values in the dataframe"""
        df = self.df
        print(f"Missing values before handling:\n{df.isnull().sum()}")
        
        # Forward fill missing values (since it's time-series data)
        df.fillna(method='ffill', inplace=True)
        
        # If there are still missing values at the beginning, use backward fill
        df.fillna(method='bfill', inplace=True)
        
        print(f"Missing values after handling:\n{df.isnull().sum()}")
        
        self.df = df
        return df
    
    def encode_sentiment_class(self):
        """Encode sentiment class if available"""
        df = self.df
        print("Encoding sentiment_class...")
        
        if 'sentiment_class' in df.columns and 'sentiment_class_encoded' not in df.columns:
            sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
            df['sentiment_class_encoded'] = df['sentiment_class'].map(sentiment_mapping)
            print(f"Sentiment class encoding: {sentiment_mapping}")
        elif 'sentiment_class_encoded' in df.columns:
            print("sentiment_class_encoded already exists in the dataset")
        else:
            print("Warning: sentiment_class column not found")
        
        self.df = df
        return df
    
    def select_features(self):
        """Select the most important features for the model"""
        df = self.df
        
        # Based on technical and sentiment indicators, select the most relevant features
        selected_features = [
            'open','high','low', 'volume','SMA_5', 'EMA_5','SMA_10','EMA_10', 
            'sentiment_score', 'weighted_sentiment', 
            'Weighted_SMA_5', 'Signal_Cross', 'sentiment_class_encoded'
        ]
        
        # Ensure all selected features are in the DataFrame
        available_features = [f for f in selected_features if f in df.columns]
        
        if len(available_features) < len(selected_features):
            print(f"Warning: Some requested features are not in the dataset: {set(selected_features) - set(available_features)}")
        
        # Target variable is 'close'
        target = df['close']
        
        self.X = df[available_features]
        self.y = target
        
        print(f"Selected features: {available_features}")
        
        return df[available_features], target
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM model"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.seq_length - self.pred_days + 1):
            # Check if X is a DataFrame/Series or already a NumPy array
            if hasattr(X, 'values'):
                X_seq.append(X[i:i+self.seq_length].values)
            else:
                X_seq.append(X[i:i+self.seq_length])
            
            # Check if y is a Series/DataFrame or already a NumPy array
            if hasattr(y, 'values'):
                y_seq.append(y[i+self.seq_length:i+self.seq_length+self.pred_days].values)
            else:
                y_seq.append(y[i+self.seq_length:i+self.seq_length+self.pred_days])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape, output_steps):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape, 
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, activation='relu', return_sequences=False, 
                 kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            
            Dense(output_steps)  # Output layer for next 7 days prediction
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def plot_history(self, history):
        """Plot and save training history"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{self.stock_symbol} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend(['Train', 'Validation'])
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title(f'{self.stock_symbol} - Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_history.png')
        plt.close()
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        # Reshape predictions and actual values for inverse scaling
        pred_shape = predictions.shape
        predictions = predictions.reshape(-1, 1)
        actual = y_test.reshape(-1, 1)
        
        # Create dummy array for inverse transform (if needed)
        if hasattr(self.scaler_y, 'n_features_in_') and self.scaler_y.n_features_in_ > 1:
            dummy = np.zeros((predictions.shape[0], self.scaler_y.n_features_in_ - 1))
            predictions_rescaled = self.scaler_y.inverse_transform(np.hstack([predictions, dummy]))[:, 0]
            actual_rescaled = self.scaler_y.inverse_transform(np.hstack([actual, dummy]))[:, 0]
        else:
            predictions_rescaled = self.scaler_y.inverse_transform(predictions).flatten()
            actual_rescaled = self.scaler_y.inverse_transform(actual).flatten()
        
        # Reshape back
        predictions_rescaled = predictions_rescaled.reshape(pred_shape)
        actual_rescaled = actual_rescaled.reshape(pred_shape)
        
        # Calculate metrics
        mse = mean_squared_error(actual_rescaled, predictions_rescaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_rescaled, predictions_rescaled)
        r2 = r2_score(actual_rescaled.flatten(), predictions_rescaled.flatten())
        
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R² Score: {r2:.4f}')
        
        # Save evaluation metrics to file
        with open(f'{self.output_dir}/evaluation_metrics.txt', 'w') as f:
            f.write(f'Stock: {self.stock_symbol}\n')
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
            f.write(f'MAE: {mae:.4f}\n')
            f.write(f'R² Score: {r2:.4f}\n')
        
        return predictions_rescaled, actual_rescaled
    
    def plot_predictions(self, predictions, actual, last_date):
        """Plot and save model predictions"""
        # Create date range for x-axis
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=predictions.shape[0]*predictions.shape[1])
        
        # Flatten predictions and actual values
        predictions_flat = predictions.reshape(-1)
        actual_flat = actual.reshape(-1)
        
        plt.figure(figsize=(14, 7))
        plt.plot(dates[:len(actual_flat)], actual_flat, label='Actual', color='blue')
        plt.plot(dates[:len(predictions_flat)], predictions_flat, label='Predicted', color='red', alpha=0.7)
        
        plt.title(f'{self.stock_symbol} - {self.pred_days}-Day Ahead Stock Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/predictions_vs_actual.png')
        plt.close()
        
        # Plot the first few predictions as separate sequences
        num_examples = min(5, predictions.shape[0])
        plt.figure(figsize=(14, num_examples*3))
        
        for i in range(num_examples):
            plt.subplot(num_examples, 1, i+1)
            
            pred_dates = pd.date_range(start=last_date + timedelta(days=1) + timedelta(days=i), periods=self.pred_days)
            plt.plot(pred_dates, predictions[i], color='red', marker='o', label='Predicted')
            plt.plot(pred_dates, actual[i], color='blue', marker='x', label='Actual')
            
            plt.title(f'Prediction Starting at {pred_dates[0].date()}')
            plt.ylabel('Stock Price ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/prediction_sequences.png')
        plt.close()
    
    def predict_future(self, model, last_sequence):
        """Predict future stock prices from the latest data"""
        # Get the most recent sequence
        input_seq = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        # Predict next 7 days
        prediction = model.predict(input_seq)
        
        # Inverse transform to get actual prices
        if hasattr(self.scaler_y, 'n_features_in_') and self.scaler_y.n_features_in_ > 1:
            dummy = np.zeros((prediction.shape[1], self.scaler_y.n_features_in_ - 1))
            future_prices = self.scaler_y.inverse_transform(np.hstack([prediction.reshape(-1, 1), dummy]))[:, 0]
        else:
            future_prices = self.scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
        
        return future_prices
    
    def train_and_evaluate(self):
        """Main method to train and evaluate the model"""
        # Load data
        self.load_data()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Encode sentiment class
        self.encode_sentiment_class()
        
        # Select features
        X, y = self.select_features()
        
        # Scale features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_scaled = pd.DataFrame(self.scaler_X.fit_transform(X), columns=X.columns, index=X.index)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        print(f"Sequence shape: X={X_seq.shape}, y={y_seq.shape}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        
        # Build model
        self.model = self.build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]), output_steps=self.pred_days)
        self.model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            min_delta=0.0001,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            f'{self.output_dir}/{self.stock_symbol}_best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        # Plot training history
        self.plot_history(history)
        
        # Load best model
        self.model = load_model(f'{self.output_dir}/{self.stock_symbol}_best_model.keras')
        
        # Evaluate model
        predictions, actual = self.evaluate_model(self.model, X_test, y_test)
        
        # Plot predictions
        last_date = self.df.index[-1]
        self.plot_predictions(predictions, actual, last_date)
        
        # Predict future prices
        future_prices = self.predict_future(self.model, X_seq[-1])
        future_dates = pd.date_range(start=self.df.index[-1] + timedelta(days=1), periods=self.pred_days)
        
        print(f"\nNext {self.pred_days} days price predictions for {self.stock_symbol}:")
        for date, price in zip(future_dates, future_prices):
            print(f"{date.date()}: ${price:.2f}")
        
        # Save future predictions to CSV
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_prices
        })
        future_df.to_csv(f'{self.output_dir}/future_predictions.csv', index=False)
        
        # Plot future predictions
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index[-30:], self.df['close'].values[-30:], label='Historical Prices')
        plt.plot(future_dates, future_prices, 'ro-', label='Predicted Prices')
        plt.title(f'{self.stock_symbol} - Stock Price Prediction for Next {self.pred_days} Days')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/future_predictions.png')
        plt.close()
        
        return future_df
