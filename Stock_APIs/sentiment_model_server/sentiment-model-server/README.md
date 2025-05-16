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


1. **Build the Docker image**:
   ```
   docker build -t sentiment-analysis-server .
   ```

2. **Run the Docker container**:
   ```
   docker run -p 8000:8000 sentiment-analysis-server
   ```

3. **Access the API**:
   The API will be available at `http://localhost:8000`. You can use tools like Postman or curl to interact with the endpoints.

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