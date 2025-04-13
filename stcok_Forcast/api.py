from flask import Flask, request, jsonify
from forecaster import StockForecaster
import logging
import os
from datetime import datetime
import threading
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
forecaster = StockForecaster()

# Cache for results to avoid redundant processing
prediction_cache = {}
training_status = {}

def clear_old_cache_entries():
    """Clear cache entries older than 1 hour"""
    current_time = datetime.now()
    to_delete = []
    
    for key, value in prediction_cache.items():
        if (current_time - value['timestamp']).total_seconds() > 3600:  # 1 hour
            to_delete.append(key)
            
    for key in to_delete:
        del prediction_cache[key]
        
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/sectors', methods=['GET'])
def get_sectors():
    """Get all available sectors"""
    sectors = forecaster.data_processor.sectors
    sector_data = {}
    
    for sector in sectors:
        stocks = forecaster.data_processor.sector_stocks[sector]
        sector_data[sector] = {
            "name": sector,
            "stock_count": len(stocks),
            "stocks": stocks[:10] + ['...'] if len(stocks) > 10 else stocks  # Only show first 10 stocks
        }
    
    return jsonify({"sectors": sector_data})

@app.route('/stock/<symbol>', methods=['GET'])
def get_stock_info(symbol):
    """Get information about a stock"""
    symbol = symbol.upper()
    
    # Find sector for the symbol
    symbol_data = forecaster.data_processor.sp500_data
    symbol_info = symbol_data[symbol_data['Symbol'] == symbol]
    
    if symbol_info.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    sector = symbol_info['Sector'].iloc[0]
    name = symbol_info['Name'].iloc[0]
    
    # Get recent price data
    try:
        price_data = forecaster.data_processor.fetch_historical_data(symbol, period="1mo")
        if price_data.empty:
            return jsonify({"error": f"No price data available for {symbol}"}), 404
        
        # Format the price data
        price_history = price_data.sort_values('Date').tail(10).to_dict(orient='records')
        for record in price_history:
            if isinstance(record['Date'], pd.Timestamp):
                record['Date'] = record['Date'].strftime('%Y-%m-%d')
                
        return jsonify({
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "recent_prices": price_history
        })
    except Exception as e:
        logger.error(f"Error retrieving stock info for {symbol}: {e}")
        return jsonify({"error": f"Error retrieving data: {str(e)}"}), 500

@app.route('/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    """Make a prediction for a stock"""
    symbol = symbol.upper()
    days_ahead = int(request.args.get('days_ahead', 5))
    period = request.args.get('period', '1y')
    
    # Check cache first
    cache_key = f"{symbol}_{days_ahead}_{period}"
    if cache_key in prediction_cache:
        cached = prediction_cache[cache_key]
        # If cached prediction is less than 30 minutes old, return it
        if (datetime.now() - cached['timestamp']).total_seconds() < 1800:
            return jsonify(cached['result'])
    
    # Find sector for the symbol
    symbol_data = forecaster.data_processor.sp500_data
    symbol_info = symbol_data[symbol_data['Symbol'] == symbol]
    
    if symbol_info.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    sector = symbol_info['Sector'].iloc[0]
    
    try:
        result = forecaster.predict(symbol, sector, days_ahead=days_ahead, period=period)
        
        # Cache the result
        prediction_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error predicting {symbol}: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/train/<symbol>', methods=['POST'])
def train_model(symbol):
    """Train a model for a stock"""
    symbol = symbol.upper()
    
    # Check if already training
    if symbol in training_status and training_status[symbol]['status'] == 'in_progress':
        return jsonify({
            "message": f"Training for {symbol} already in progress", 
            "started_at": training_status[symbol]['started_at']
        })
    
    # Get parameters from request
    data = request.json or {}
    period = data.get('period', '5y')
    window_size = data.get('window_size', forecaster.window_size)
    prediction_horizon = data.get('prediction_horizon', forecaster.prediction_horizon)
    include_sentiment = data.get('include_sentiment', True)
    epochs = data.get('epochs', forecaster.epochs)
    
    # Find sector for the symbol
    symbol_data = forecaster.data_processor.sp500_data
    symbol_info = symbol_data[symbol_data['Symbol'] == symbol]
    
    if symbol_info.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    sector = symbol_info['Sector'].iloc[0]
    
    # Start training in a background thread
    def train_background():
        try:
            training_status[symbol] = {
                'status': 'in_progress',
                'started_at': datetime.now().isoformat(),
                'parameters': {
                    'period': period,
                    'window_size': window_size,
                    'prediction_horizon': prediction_horizon,
                    'include_sentiment': include_sentiment,
                    'epochs': epochs
                }
            }
            
            result = forecaster.train_model(
                symbol, 
                sector, 
                period=period, 
                window_size=window_size,
                prediction_horizon=prediction_horizon,
                include_sentiment=include_sentiment,
                epochs=epochs
            )
            
            training_status[symbol] = {
                'status': 'completed',
                'started_at': training_status[symbol]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'result': result
            }
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            training_status[symbol] = {
                'status': 'failed',
                'started_at': training_status[symbol]['started_at'],
                'failed_at': datetime.now().isoformat(),
                'error': str(e)
            }
    
    # Start the training thread
    train_thread = threading.Thread(target=train_background)
    train_thread.start()
    
    return jsonify({
        "message": f"Training started for {symbol}",
        "status": "in_progress",
        "started_at": training_status[symbol]['started_at']
    })

@app.route('/training-status/<symbol>', methods=['GET'])
def get_training_status(symbol):
    """Get the status of a training job"""
    symbol = symbol.upper()
    
    if symbol in training_status:
        return jsonify(training_status[symbol])
    else:
        return jsonify({"status": "not_found", "message": f"No training job found for {symbol}"}), 404

@app.route('/sector-predict/<sector>', methods=['GET'])
def predict_sector(sector):
    """Make predictions for all stocks in a sector"""
    limit = request.args.get('limit', None)
    if limit is not None:
        limit = int(limit)
    days_ahead = int(request.args.get('days_ahead', 5))
    top_n = int(request.args.get('top', 5))
    
    if sector not in forecaster.data_processor.sectors:
        return jsonify({"error": f"Sector {sector} not found"}), 404
    
    try:
        predictions = forecaster.predict_sector(sector, limit=limit, days_ahead=days_ahead)
        top_predictions = forecaster.get_top_predictions(predictions, top_n=top_n)
        
        return jsonify({
            "sector": sector,
            "days_ahead": days_ahead,
            "predictions_count": len(predictions),
            "successful_predictions": sum(1 for p in predictions.values() if p.get('success', False)),
            "top_predictions": top_predictions
        })
    except Exception as e:
        logger.error(f"Error predicting sector {sector}: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/backtest/<symbol>', methods=['GET'])
def backtest_model(symbol):
    """Backtest a model on historical data"""
    symbol = symbol.upper()
    test_period = request.args.get('period', '1y')
    window_size = int(request.args.get('window', forecaster.window_size))
    prediction_horizon = int(request.args.get('horizon', forecaster.prediction_horizon))
    
    # Find sector for the symbol
    symbol_data = forecaster.data_processor.sp500_data
    symbol_info = symbol_data[symbol_data['Symbol'] == symbol]
    
    if symbol_info.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    sector = symbol_info['Sector'].iloc[0]
    
    try:
        result = forecaster.backtest(
            symbol, 
            sector, 
            test_period=test_period,
            window_size=window_size,
            prediction_horizon=prediction_horizon
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error backtesting {symbol}: {e}")
        return jsonify({"error": f"Backtest error: {str(e)}"}), 500

# Create a periodic task to clean the cache
def start_cache_cleaner():
    """Start a thread to periodically clean the cache"""
    def cleaner():
        while True:
            time.sleep(600)  # Run every 10 minutes
            clear_old_cache_entries()
    
    thread = threading.Thread(target=cleaner)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    import time
    start_cache_cleaner()
    app.run(debug=True, host='0.0.0.0', port=5000)
