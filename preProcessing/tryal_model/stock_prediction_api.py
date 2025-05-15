import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
MODEL_PATH = r'AAPL.keras'
model = load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# Constants for prediction
SEQ_LENGTH = 30  # Number of days used for input sequence
PRED_DAYS = 7   # Number of days to predict

# Cache for scalers and last sequence
cache = {
    'scaler_X': None,
    'scaler_y': None,
    'last_sequence': None,
    'last_date': None,
    'feature_columns': None
}

# Initialize data on startup
try:
    # Call the function that will be defined below
    def initialize_data():
        load_and_prepare_data()
        print("Data loaded and model ready for predictions")
        
    initialize_data()
except Exception as e:
    print(f"Error during initialization: {str(e)}")

def load_and_prepare_data(data_path='../MergedData/AAPL_merged.csv'):
    """Load and prepare the dataset for prediction"""
    df = pd.read_csv(data_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Sort by date (just to be sure)
    df.sort_index(inplace=True)
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Encode sentiment class if available
    if 'sentiment_class' in df.columns and 'sentiment_class_encoded' not in df.columns:
        sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['sentiment_class_encoded'] = df['sentiment_class'].map(sentiment_mapping)
    
    # Select features
    selected_features = [
        'open', 'high', 'low', 'volume', 'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10',
        'sentiment_score', 'weighted_sentiment', 
        'Weighted_SMA_5', 'Signal_Cross', 'sentiment_class_encoded'
    ]
    
    # Ensure all selected features are in the DataFrame
    available_features = [f for f in selected_features if f in df.columns]
    
    # Store the feature columns for future reference
    cache['feature_columns'] = available_features
    
    # Target variable is 'close'
    X = df[available_features]
    y = df['close']
    
    # Create and fit scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(X)
    scaler_y.fit(y.values.reshape(-1, 1))
    
    # Scale the data
    X_scaled = pd.DataFrame(scaler_X.transform(X), columns=X.columns, index=X.index)
    
    # Save scalers and last sequence for future predictions
    cache['scaler_X'] = scaler_X
    cache['scaler_y'] = scaler_y
    cache['last_sequence'] = X_scaled.values[-SEQ_LENGTH:]
    cache['last_date'] = df.index[-1]
    
    return X_scaled, y, df.index[-1]

def predict_future(last_sequence, days=PRED_DAYS):
    """Predict future stock prices"""
    # Reshape input sequence for the model
    input_seq = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    # Predict
    prediction = model.predict(input_seq)
    
    # Inverse transform to get actual prices
    future_prices = cache['scaler_y'].inverse_transform(prediction.reshape(-1, 1)).flatten()
    
    # Generate future dates
    last_date = cache['last_date']
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    return future_dates, future_prices

def generate_prediction_chart(historical_dates, historical_prices, future_dates, future_prices, filename='prediction_chart.png'):
    """Generate a chart showing historical data and predictions"""
    plt.figure(figsize=(12, 6))
    plt.plot(historical_dates, historical_prices, label='Historical Prices')
    plt.plot(future_dates, future_prices, 'ro-', label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """API endpoint to check if the service is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Stock prediction API is running',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/predict', methods=['GET'])
def predict():
    """API endpoint to predict future stock prices"""
    try:
        # Check if we need to initialize
        if cache['last_sequence'] is None:
            X_scaled, y, last_date = load_and_prepare_data()
            
        # Use fixed 7 days for prediction as requested
        days = 7  # Using fixed 7 days as requested
            
        # Make prediction
        future_dates, future_prices = predict_future(cache['last_sequence'], days)
        
        # Format dates for JSON response
        dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Create response
        response = {
            'dates': dates_str,
            'predicted_prices': future_prices.tolist(),
            'last_updated': cache['last_date'].strftime('%Y-%m-%d')
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# The following routes are commented out as per the new requirements
# Keeping the code for reference

"""
@app.route('/predict/chart', methods=['GET'])
def predict_with_chart():
    # ... existing code ...

@app.route('/update', methods=['POST'])
def update_data():
    # ... existing code ...
"""

@app.route('/', methods=['GET'])
def index():
    return """
    <h1>Stock Price Prediction API</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/healthcheck">/healthcheck</a> - Check if the API is running</li>
        <li><a href="/predict">/predict</a> - Get price predictions for the next 7 days</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(debug=True, port=3000)
