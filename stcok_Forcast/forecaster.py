import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple
import os
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_processor import DataProcessor
from sentiment_analyzer import SentimentAnalyzer
from model_builder import ModelBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forecaster.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockForecaster:
    def __init__(self, 
                 data_dir: str = "data", 
                 models_dir: str = "models", 
                 results_dir: str = "results"):
        """Initialize the stock forecaster."""
        self.data_processor = DataProcessor(data_dir=data_dir)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.model_builder = ModelBuilder(models_dir=models_dir)
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Default parameters
        self.window_size = 20
        self.prediction_horizon = 5
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_stock_data(self, 
                          symbol: str, 
                          period: str = "5y", 
                          include_sentiment: bool = True) -> pd.DataFrame:
        """Fetch and prepare data for a stock symbol."""
        logger.info(f"Preparing data for {symbol}")
        
        # Get stock data with technical indicators
        stock_data = self.data_processor.fetch_historical_data(symbol, period)
        if stock_data.empty:
            logger.error(f"No data available for {symbol}")
            return pd.DataFrame()
            
        # Compute technical indicators
        stock_data = self.data_processor.compute_technical_indicators(stock_data)
        
        # Add sentiment data if requested
        if include_sentiment:
            try:
                sentiment_data = self.sentiment_analyzer.analyze_sentiment_for_stock(symbol, days_back=365)
                if not sentiment_data.empty:
                    stock_data = self.data_processor.compute_sentiment_indicators(stock_data, sentiment_data)
                    logger.info(f"Added sentiment indicators for {symbol}")
                else:
                    logger.warning(f"No sentiment data available for {symbol}")
            except Exception as e:
                logger.error(f"Error adding sentiment data for {symbol}: {e}")
                
        return stock_data
        
    def train_model(self, 
                   symbol: str, 
                   sector: str, 
                   period: str = "5y", 
                   window_size: int = None, 
                   prediction_horizon: int = None,
                   include_sentiment: bool = True,
                   epochs: int = None,
                   learning_rate: float = None) -> Dict:
        """Train a model for a stock symbol."""
        # Set parameters or use defaults
        window_size = window_size or self.window_size
        prediction_horizon = prediction_horizon or self.prediction_horizon
        epochs = epochs or self.epochs
        learning_rate = learning_rate or self.learning_rate
        
        logger.info(f"Training model for {symbol} (sector: {sector})")
        
        # Prepare data
        stock_data = self.prepare_stock_data(symbol, period, include_sentiment)
        if stock_data.empty:
            return {"success": False, "error": "No data available"}
            
        # Prepare datasets
        X, y = self.data_processor.prepare_dataset(
            stock_data, 
            window_size=window_size, 
            prediction_horizon=prediction_horizon
        )
        
        if X is None or len(X) == 0:
            return {"success": False, "error": "Failed to prepare dataset"}
            
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        
        X_train_scaled_2d = scaler.fit_transform(X_train_2d)
        X_val_scaled_2d = scaler.transform(X_val_2d)
        
        # Reshape back
        X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
        X_val_scaled = X_val_scaled_2d.reshape(X_val.shape)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Create model
        input_dim = X_train.shape[2]  # Number of features
        model = self.model_builder.create_model(sector, input_dim)
        model.to(self.device)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 10
        counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
        # Save the model
        model_path, scaler_path = self.model_builder.save_model(model, scaler, symbol, sector)
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            predictions = model(X_val_tensor.to(self.device)).cpu().numpy()
            
        # Calculate metrics
        mse = mean_squared_error(y_val, predictions)
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        # Save training history and results
        self._save_training_results(symbol, sector, train_losses, val_losses, 
                                   y_val, predictions, mse, mae, r2)
        
        return {
            "success": True,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metrics": {
                "mse": mse,
                "mae": mae,
                "r2": r2
            },
            "training_epochs": len(train_losses)
        }
    
    def predict(self, 
               symbol: str, 
               sector: str, 
               days_ahead: int = 5,
               period: str = "1y") -> Dict:
        """Make predictions for a stock symbol."""
        logger.info(f"Making predictions for {symbol} (sector: {sector})")
        
        try:
            # Prepare recent data
            stock_data = self.prepare_stock_data(symbol, period, include_sentiment=True)
            if stock_data.empty:
                return {"success": False, "error": "No data available"}
                
            # Load model and scaler
            input_dim = len(stock_data.columns) - 1  # All columns except Date
            model, scaler = self.model_builder.load_model(symbol, sector, input_dim)
            model.to(self.device)
            model.eval()
            
            # Use the most recent window_size data points
            recent_data = stock_data.sort_values('Date').tail(self.window_size)
            
            # Drop Date column and convert to numpy array
            features = [col for col in recent_data.columns if col != 'Date']
            X = recent_data[features].values
            
            # Reshape, scale, and convert to tensor
            X_reshaped = X.reshape(1, X.shape[0], X.shape[1])  # Add batch dimension
            X_2d = X_reshaped.reshape(1, -1)
            X_scaled_2d = scaler.transform(X_2d)
            X_scaled = X_scaled_2d.reshape(X_reshaped.shape)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                price_change_pct = model(X_tensor).cpu().numpy()[0][0]
                
            # Get the last closing price
            last_price = recent_data['Close'].iloc[-1]
            last_date = recent_data['Date'].iloc[-1]
            
            # Calculate predicted price
            predicted_price = last_price * (1 + price_change_pct)
            
            # Create prediction result
            prediction_date = pd.to_datetime(last_date) + timedelta(days=days_ahead)
            
            result = {
                "success": True,
                "symbol": symbol,
                "sector": sector,
                "last_date": last_date.strftime('%Y-%m-%d') if isinstance(last_date, pd.Timestamp) else last_date,
                "last_price": float(last_price),
                "prediction_date": prediction_date.strftime('%Y-%m-%d'),
                "predicted_price": float(predicted_price),
                "predicted_change_pct": float(price_change_pct) * 100
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions for {symbol}: {e}")
            return {"success": False, "error": str(e)}
            
    def _save_training_results(self, 
                              symbol: str, 
                              sector: str, 
                              train_losses: List[float], 
                              val_losses: List[float],
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              mse: float,
                              mae: float,
                              r2: float):
        """Save training results and plots."""
        # Create directories
        sector_dir = os.path.join(self.results_dir, sector.replace(" ", "_"))
        os.makedirs(sector_dir, exist_ok=True)
        symbol_dir = os.path.join(sector_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics = {
            "timestamp": timestamp,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "training_epochs": len(train_losses)
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(symbol_dir, f"metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{symbol} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(symbol_dir, f"loss_plot_{timestamp}.png")
        plt.savefig(loss_plot_path)
        plt.close()
        
        # Save prediction plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.title(f'{symbol} True vs Predicted Price Change %')
        plt.xlabel('True Price Change %')
        plt.ylabel('Predicted Price Change %')
        plt.grid(True)
        pred_plot_path = os.path.join(symbol_dir, f"prediction_plot_{timestamp}.png")
        plt.savefig(pred_plot_path)
        plt.close()

    def train_sector_models(self, 
                           sector: str, 
                           limit: int = None,
                           period: str = "5y") -> Dict:
        """Train models for all stocks in a sector."""
        stocks = self.data_processor.sector_stocks.get(sector, [])
        if limit:
            stocks = stocks[:limit]
            
        results = {}
        for symbol in stocks:
            logger.info(f"Training model for {symbol} in sector {sector}")
            result = self.train_model(symbol, sector, period=period)
            results[symbol] = result
            
        return results
    
    def predict_sector(self, 
                      sector: str, 
                      limit: int = None,
                      days_ahead: int = 5) -> Dict:
        """Make predictions for all stocks in a sector."""
        stocks = self.data_processor.sector_stocks.get(sector, [])
        if limit:
            stocks = stocks[:limit]
            
        results = {}
        for symbol in stocks:
            logger.info(f"Predicting for {symbol} in sector {sector}")
            result = self.predict(symbol, sector, days_ahead=days_ahead)
            results[symbol] = result
            
        return results

    def get_top_predictions(self, 
                           sector_predictions: Dict[str, Dict], 
                           top_n: int = 5, 
                           metric: str = "predicted_change_pct") -> List[Dict]:
        """Get top N predictions based on specified metric."""
        valid_predictions = [
            pred for symbol, pred in sector_predictions.items() 
            if pred.get("success", False)
        ]
        
        # Sort by the specified metric in descending order
        sorted_predictions = sorted(
            valid_predictions, 
            key=lambda x: x.get(metric, 0), 
            reverse=True
        )
        
        return sorted_predictions[:top_n]
        
    def backtest(self, 
                symbol: str, 
                sector: str,
                test_period: str = "1y",
                window_size: int = None,
                prediction_horizon: int = None) -> Dict:
        """Backtest a model on historical data."""
        window_size = window_size or self.window_size
        prediction_horizon = prediction_horizon or self.prediction_horizon
        
        logger.info(f"Backtesting model for {symbol}")
        
        try:
            # Get historical data
            stock_data = self.prepare_stock_data(symbol, period=test_period, include_sentiment=True)
            if stock_data.empty:
                return {"success": False, "error": "No data available for backtesting"}
                
            # Load model and scaler
            input_dim = len(stock_data.columns) - 1  # All columns except Date
            model, scaler = self.model_builder.load_model(symbol, sector, input_dim)
            model.to(self.device)
            model.eval()
            
            # Prepare datasets for backtesting
            X, y = self.data_processor.prepare_dataset(
                stock_data, 
                window_size=window_size, 
                prediction_horizon=prediction_horizon
            )
            
            if X is None or len(X) == 0:
                return {"success": False, "error": "Failed to prepare backtest dataset"}
                
            # Scale features
            X_2d = X.reshape(X.shape[0], -1)
            X_scaled_2d = scaler.transform(X_2d)
            X_scaled = X_scaled_2d.reshape(X.shape)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                predictions = model(X_tensor).cpu().numpy().flatten()
                
            # Calculate metrics
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            # Create result dictionary with dates
            dates = stock_data['Date'].iloc[window_size:len(y) + window_size].values
            backtest_results = {
                "success": True,
                "symbol": symbol,
                "sector": sector,
                "metrics": {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2
                },
                "predictions": [
                    {
                        "date": dates[i].strftime('%Y-%m-%d') if isinstance(dates[i], pd.Timestamp) else dates[i],
                        "actual": float(y[i]),
                        "predicted": float(predictions[i])
                    }
                    for i in range(len(y))
                ]
            }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtesting for {symbol}: {e}")
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    forecaster = StockForecaster()
    
    # Example 1: Train a model for a single stock
    # result = forecaster.train_model("AAPL", "Technology")
    # print(f"Training result: {result}")
    
    # Example 2: Make predictions for a stock
    # prediction = forecaster.predict("AAPL", "Technology")
    # print(f"Prediction: {prediction}")
    
    # Example 3: Train models for a sector
    # tech_results = forecaster.train_sector_models("Technology", limit=5)
    # print(f"Trained {len(tech_results)} models for Technology sector")
    
    # Example 4: Get predictions for a sector and find top performers
    # tech_predictions = forecaster.predict_sector("Technology", limit=10)
    # top_picks = forecaster.get_top_predictions(tech_predictions, top_n=3)
    # print(f"Top picks: {top_picks}")
    
    # Example 5: Backtest a model
    # backtest_result = forecaster.backtest("AAPL", "Technology")
    # print(f"Backtest metrics: {backtest_result['metrics']}")
