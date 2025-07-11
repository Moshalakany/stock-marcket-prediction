from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import statistics as st
from enhanced_sentiment_model import EnhancedStockSentimentAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Allow CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable for the analyzer
analyzer = None

def get_analyzer():
    """Get or initialize the sentiment analyzer."""
    global analyzer
    if analyzer is None:
        # Get model directory from environment variable or use default
        model_dir = os.environ.get('MODEL_DIR', r'model_files')
        data_dir = os.environ.get('DATA_DIR', r'model_files')
        
        logger.info(f"Initializing sentiment analyzer with model dir: {model_dir}")
        analyzer = EnhancedStockSentimentAnalyzer(data_dir, model_dir)
        
        # Try to load the model
        load_model_files(analyzer, model_dir)
    
    return analyzer

def load_model_files(analyzer, model_dir):
    """Explicitly load the model and tokenizer files"""
    try:
        model_path = os.path.join(model_dir, 'best_model.keras')
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            # Force model loading by making a test prediction
            analyzer.predict_sentiment("Test loading model")
            logger.info("Sentiment model loaded successfully")
            return True
        else:
            logger.warning(f"Model files not found at {model_dir}. Model will load on first prediction.")
            logger.warning(f"Looking for: {model_path} and {tokenizer_path}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Validation functions to replace Pydantic validators
def validate_text(text):
    if not text or text.strip() == "":
        return False, "Text cannot be empty"
    if len(text) > 10000:
        return False, "Text cannot exceed 10000 characters"
    return True, text

def validate_texts(texts):
    if not texts or len(texts) == 0:
        return False, "Texts list cannot be empty"
    if len(texts) > 100:
        return False, "Cannot process more than 100 texts at once"
    
    for text in texts:
        if not text or text.strip() == "":
            return False, "Texts cannot contain empty strings"
    return True, texts

# Initialize model at startup
# Replace @app.before_first_request with application context
def startup():
    """Initialize and load the model when the API starts"""
    logger.info("API starting up - initializing sentiment analyzer...")
    analyzer = get_analyzer()
    if analyzer and hasattr(analyzer, 'model') and analyzer.model is not None:
        logger.info("✓ Sentiment model successfully loaded and ready")
    else:
        logger.warning("! Sentiment model not loaded at startup. Will attempt to load on first request.")

# Call startup function right after creating the app
with app.app_context():
    startup()

@app.route("/", methods=["GET"])
def read_root():
    """Root endpoint with API information."""
    return jsonify({
        "api": "Stock News Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/sentiment/analyze": "Analyze sentiment for a single text",
            "/sentiment/batch": "Analyze sentiment for multiple texts"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify if the API and model are operational."""
    model_loaded = False
    
    try:
        analyzer = get_analyzer()
        model_loaded = hasattr(analyzer, 'model') and analyzer.model is not None
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"error": f"API is experiencing issues: {str(e)}"}), 500
    
    return jsonify({
        "status": "healthy", 
        "model_loaded": model_loaded
    })

def is_model_ready(analyzer):
    """Check if the analyzer and its model are properly loaded and ready for predictions"""
    return analyzer and hasattr(analyzer, 'model') and analyzer.model is not None

@app.route("/sentiment/analyze", methods=["POST"])
def analyze_sentiment():
    """
    Analyze sentiment for a single text.
    
    Returns sentiment classification (positive, negative, neutral) along with confidence scores.
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required field: text"}), 400
        
        valid, result = validate_text(data['text'])
        if not valid:
            return jsonify({"error": result}), 400
        
        analyzer = get_analyzer()
        if not is_model_ready(analyzer):
            logger.error("Sentiment model not loaded or initialized properly")
            return jsonify({"error": "Sentiment model not available. Please try again later."}), 503
        
        result = analyzer.predict_sentiment(data['text'], return_scores=True)
        
        return jsonify({
            "text": data['text'],
            "sentiment": result['sentiment'],
            "confidence": result['confidence'],
            "scores": result['scores']
        })
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return jsonify({"error": f"Error analyzing sentiment: {str(e)}"}), 500

@app.route("/sentiment/batch", methods=["POST"])
def batch_analyze_sentiment():
    """
    Analyze sentiment for multiple texts.
    
    Returns sentiment classifications and confidence scores for each text.
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing required field: texts"}), 400
        
        valid, result = validate_texts(data['texts'])
        if not valid:
            return jsonify({"error": result}), 400
        
        analyzer = get_analyzer()
        if not is_model_ready(analyzer):
            logger.error("Sentiment model not loaded or initialized properly")
            return jsonify({"error": "Sentiment model not available. Please try again later."}), 503
        
        results = analyzer.predict_sentiment(data['texts'], return_scores=True)
        sentiment_arr = []
        
        responses = []
        for result in results:
            sentiment_arr.append(result['sentiment'])
            responses.append({
                "text": result['text'],
                "sentiment": result['sentiment'],
                "confidence": result['confidence'],
                "scores": result['scores']
            })
        
        return jsonify({
            "results": responses,
            "count": len(results),
            "sentiment_mod": st.mode(sentiment_arr)
        })
    except Exception as e:
        logger.error(f"Error batch analyzing sentiment: {str(e)}")
        return jsonify({"error": f"Error batch analyzing sentiment: {str(e)}"}), 500

if __name__ == "__main__":
    # Run the Flask app when script is executed directly
    app.run(host="0.0.0.0", port=8000, debug=True)