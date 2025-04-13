# Stock Market Prediction System

This system provides tools for predicting stock market movements using machine learning models trained on historical data and sentiment analysis.

## Features

- Historical stock data retrieval and preprocessing
- Technical indicator calculation
- Sentiment analysis from financial news
- Custom machine learning model architectures for different sectors
- Model training, evaluation, and prediction
- Backtesting capabilities
- Command-line interface
- REST API for web access

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-prediction.git
   cd stock-market-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The system provides a command-line interface for various operations:

#### List Available Sectors

```bash
python main.py sectors
```

#### Train a Model

For a single stock:
```bash
python main.py train --symbol AAPL --period 5y --sentiment
```

For an entire sector (with optional limit):
```bash
python main.py train --sector Technology --limit 5
```

#### Make Predictions

For a single stock:
```bash
python main.py predict --symbol AAPL --days-ahead 5
```

For a sector (showing top performers):
```bash
python main.py predict --sector Technology --limit 10 --top 5
```

#### Backtest a Model

```bash
python main.py backtest --symbol AAPL --period 1y
```

### Web API

The system also provides a REST API for web access:

1. Start the API server:
   ```bash
   python api.py
   ```

2. Access the API at `http://localhost:5000`

#### API Endpoints

- `GET /health`: Health check
- `GET /sectors`: List all available sectors
- `GET /stock/<symbol>`: Get information about a stock
- `GET /predict/<symbol>`: Make predictions for a stock
- `POST /train/<symbol>`: Train a model for a stock
- `GET /training-status/<symbol>`: Check the status of a training job
- `GET /sector-predict/<sector>`: Make predictions for all stocks in a sector
- `GET /backtest/<symbol>`: Backtest a model on historical data

## Project Structure

- `data_processor.py`: Handles data retrieval and preprocessing
- `sentiment_analyzer.py`: Analyzes sentiment from financial news
- `model_builder.py`: Creates and manages machine learning models
- `forecaster.py`: Main forecasting system that integrates all components
- `main.py`: Command-line interface
- `api.py`: REST API server

## Dependencies

- pandas, numpy: Data processing
- yfinance: Stock data retrieval
- scikit-learn, torch: Machine learning
- transformers: NLP models for sentiment analysis
- flask: Web API
- matplotlib: Visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.