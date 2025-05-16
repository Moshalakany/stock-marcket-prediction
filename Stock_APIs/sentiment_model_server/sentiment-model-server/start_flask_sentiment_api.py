from app.flask_sentiment_api import app

if __name__ == "__main__":
    print("Starting Sentiment Analysis Server on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000)
