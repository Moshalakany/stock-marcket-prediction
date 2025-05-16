from app.flask_sentiment_api import app
import os

if __name__ == "__main__":
    print("Starting Sentiment Analysis Server on http://0.0.0.0:8000")
    
    # Get environment variables with defaults
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 8000))
    
    # Print environment info
    print(f"Model directory: {os.environ.get('MODEL_DIR', 'model_files')}")
    print(f"Data directory: {os.environ.get('DATA_DIR', 'model_files')}")
    print(f"Debug mode: {debug_mode}")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
