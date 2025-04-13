import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os
from datetime import timedelta, datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
def load_data(file_path=r'../MergedData/AAPL_merged.csv'):
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Sort by date (just to be sure)
    df.sort_index(inplace=True)
    
    return df

# Handle missing values
def handle_missing_values(df):
    print(f"Missing values before handling:\n{df.isnull().sum()}")
    
    # Forward fill missing values (since it's time-series data)
    df.fillna(method='ffill', inplace=True)
    
    # If there are still missing values at the beginning, use backward fill
    df.fillna(method='bfill', inplace=True)
    
    print(f"Missing values after handling:\n{df.isnull().sum()}")
    
    return df

# Encode sentiment class
def encode_sentiment_class(df):
    print("Encoding sentiment_class...")
    
    if 'sentiment_class' in df.columns and 'sentiment_class_encoded' not in df.columns:
        sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['sentiment_class_encoded'] = df['sentiment_class'].map(sentiment_mapping)
        print(f"Sentiment class encoding: {sentiment_mapping}")
    elif 'sentiment_class_encoded' in df.columns:
        print("sentiment_class_encoded already exists in the dataset")
    else:
        print("Warning: sentiment_class column not found")
    
    return df

# Select important features
def select_features(df):
    # Based on technical and sentiment indicators, select the most relevant features
    # This selection can be further refined using feature importance techniques
    
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
    
    # Return features and target
    return df[available_features], target

# Create sequences for LSTM
def create_sequences(X, y, seq_length=30, pred_days=7):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length - pred_days + 1):
        # Check if X is a DataFrame/Series or already a NumPy array
        if hasattr(X, 'values'):
            X_seq.append(X[i:i+seq_length].values)
        else:
            X_seq.append(X[i:i+seq_length])
        
        # Check if y is a Series/DataFrame or already a NumPy array
        if hasattr(y, 'values'):
            y_seq.append(y[i+seq_length:i+seq_length+pred_days].values)
        else:
            y_seq.append(y[i+seq_length:i+seq_length+pred_days])
    
    return np.array(X_seq), np.array(y_seq)

# Build LSTM model
def build_model(input_shape, output_steps):
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

# Plot loss history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Evaluate model
def evaluate_model(model, X_test, y_test, scaler_y):
    predictions = model.predict(X_test)
    
    # Reshape predictions and actual values for inverse scaling
    pred_shape = predictions.shape
    predictions = predictions.reshape(-1, 1)
    actual = y_test.reshape(-1, 1)
    
    # Create dummy array for inverse transform (if needed)
    if hasattr(scaler_y, 'n_features_in_') and scaler_y.n_features_in_ > 1:
        dummy = np.zeros((predictions.shape[0], scaler_y.n_features_in_ - 1))
        predictions_rescaled = scaler_y.inverse_transform(np.hstack([predictions, dummy]))[:, 0]
        actual_rescaled = scaler_y.inverse_transform(np.hstack([actual, dummy]))[:, 0]
    else:
        predictions_rescaled = scaler_y.inverse_transform(predictions).flatten()
        actual_rescaled = scaler_y.inverse_transform(actual).flatten()
    
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
    
    return predictions_rescaled, actual_rescaled

# Plot predictions
def plot_predictions(predictions, actual, last_date, pred_days=7):
    # Create date range for x-axis
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=predictions.shape[0]*predictions.shape[1])
    
    # Flatten predictions and actual values
    predictions_flat = predictions.reshape(-1)
    actual_flat = actual.reshape(-1)
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates[:len(actual_flat)], actual_flat, label='Actual', color='blue')
    plt.plot(dates[:len(predictions_flat)], predictions_flat, label='Predicted', color='red', alpha=0.7)
    
    plt.title('7-Day Ahead Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.show()
    
    # Plot the first few predictions as separate sequences
    num_examples = min(5, predictions.shape[0])
    plt.figure(figsize=(14, num_examples*3))
    
    for i in range(num_examples):
        plt.subplot(num_examples, 1, i+1)
        
        pred_dates = pd.date_range(start=last_date + timedelta(days=1) + timedelta(days=i), periods=pred_days)
        plt.plot(pred_dates, predictions[i], color='red', marker='o', label='Predicted')
        plt.plot(pred_dates, actual[i], color='blue', marker='x', label='Actual')
        
        plt.title(f'Prediction Starting at {pred_dates[0].date()}')
        plt.ylabel('Stock Price ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_sequences.png')
    plt.show()

# Predict next 7 days from the latest data
def predict_future(model, last_sequence, scaler_X, scaler_y):
    # Get the most recent sequence
    input_seq = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    # Predict next 7 days
    prediction = model.predict(input_seq)
    
    # Inverse transform to get actual prices
    if hasattr(scaler_y, 'n_features_in_') and scaler_y.n_features_in_ > 1:
        dummy = np.zeros((prediction.shape[1], scaler_y.n_features_in_ - 1))
        future_prices = scaler_y.inverse_transform(np.hstack([prediction.reshape(-1, 1), dummy]))[:, 0]
    else:
        future_prices = scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
    
    return future_prices

# Main function
def main():
    # Load data
    df = load_data()
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode sentiment class
    df = encode_sentiment_class(df)
    
    # Select features
    X, y = select_features(df)
    print(f"Selected features: {X.columns.tolist()}")
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    seq_length = 30  # Use 30 days of history
    pred_days = 7    # Predict next 7 days
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length, pred_days)
    
    print(f"Sequence shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )
    
    # Build model
    model = build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]), output_steps=pred_days)
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Reduced patience for earlier stopping
        min_delta=0.0001,  # Minimum change to qualify as improvement
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'AAPL.keras',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Reduce learning rate by 80% when plateau is detected
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    # Plot training history
    plot_history(history)
    
    # Evaluate model
    predictions, actual = evaluate_model(model, X_test, y_test, scaler_y)
    
    # Plot predictions
    last_date = df.index[-1]
    plot_predictions(predictions, actual, last_date, pred_days)
    
    # Predict future prices
    future_prices = predict_future(model, X_seq[-1], scaler_X, scaler_y)
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
    
    print("\nNext 7 days price predictions:")
    for date, price in zip(future_dates, future_prices):
        print(f"{date.date()}: ${price:.2f}")
    
    # Save future predictions to CSV
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })
    future_df.to_csv('future_predictions.csv', index=False)
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-30:], df['close'].values[-30:], label='Historical Prices')
    plt.plot(future_dates, future_prices, 'ro-', label='Predicted Prices')
    plt.title('Stock Price Prediction for Next 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('future_predictions.png')
    plt.show()

if __name__ == "__main__":
    main()
