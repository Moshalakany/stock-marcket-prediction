# Sentiment Analysis Server

This project is a Flask-based sentiment analysis server that utilizes a trained machine learning model to analyze the sentiment of stock news articles. The server provides endpoints for analyzing sentiment for both single and multiple texts.

## Project Structure

```
sentiment-model-server
├── Dockerfile
├── requirements.txt
├── model_files
│   ├── best_model.keras
│   └── tokenizer.pkl
├── app
│   ├── __init__.py
│   ├── flask_sentiment_api.py
│   ├── enhanced_sentiment_model.py
│   └── utils
│       └── __init__.py
├── start_flask_sentiment_api.py
└── README.md
```

## Setup Instructions

### Option 1: Local Development

1. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```
   python start_flask_sentiment_api.py
   ```
   
   The server will start on http://0.0.0.0:8000

### Option 2: Using Docker

1. **Build the Docker image**:
   ```
   docker build -t sentiment-analysis-server .
   ```

2. **Run the Docker container**:
   ```
   docker run -p 8000:8000 sentiment-analysis-server
   ```

## Using the API

The API will be available at `http://localhost:8000`. You can interact with the API using tools like Postman, curl, or any HTTP client.

### Example API Calls:

1. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Analyze Single Text**:
   ```bash
   curl -X POST http://localhost:8000/sentiment/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Stock prices are expected to rise significantly next quarter due to increased earnings."}'
   ```

3. **Analyze Multiple Texts**:
   ```bash
   curl -X POST http://localhost:8000/sentiment/batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Stock prices are rising", "Market is facing a downturn"]}'
   ```

## API Endpoints

- **GET /**: Returns API information.
- **GET /health**: Health check endpoint to verify if the API and model are operational.
- **POST /sentiment/analyze**: Analyze sentiment for a single text.
- **POST /sentiment/batch**: Analyze sentiment for multiple texts.

## Requirements

The project requires the following Python packages, which are listed in `requirements.txt`. These will be installed automatically when building the Docker image.

## Model Files

The trained sentiment analysis model and tokenizer are located in the `model_files` directory. Ensure that these files are present for the server to function correctly.

## Logging

The server logs important events and errors to `sentiment_api.log`. Check this file for debugging and monitoring purposes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.