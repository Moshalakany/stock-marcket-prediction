from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import os
import logging
import statistics as st
from enhanced_sentiment_model import EnhancedStockSentimentAnalyzer
import uvicorn

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

# Initialize FastAPI app
app = FastAPI(
    title="Stock News Sentiment API",
    description="API for analyzing sentiment in stock market news and texts",
    version="1.0.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define Pydantic models for request and response
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze sentiment")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if v.strip() == "":
            raise ValueError('text cannot be empty')
        return v

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze sentiment")
    
    @validator('texts')
    def texts_must_not_be_empty(cls, texts):
        for text in texts:
            if not text or text.strip() == "":
                raise ValueError('texts cannot contain empty strings')
        return texts

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: Optional[float] = None
    scores: Optional[Dict[str, float]] = None

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    count: int
    sentiment_mod: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Global variable for the analyzer
analyzer = None

def get_analyzer():
    """Get or initialize the sentiment analyzer."""
    global analyzer
    if analyzer is None:
        # Get model directory from environment variable or use default
        model_dir = os.environ.get("MODEL_DIR", r'model_files')
        data_dir = os.environ.get("DATA_DIR", r'model_files')
        
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
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize and load the model when the API starts"""
    logger.info("API starting up - initializing sentiment analyzer...")
    analyzer = get_analyzer()
    if analyzer and hasattr(analyzer, 'model') and analyzer.model is not None:
        logger.info("✓ Sentiment model successfully loaded and ready")
    else:
        logger.warning("! Sentiment model not loaded at startup. Will attempt to load on first request.")

@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint with API information."""
    return {
        "api": "Stock News Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/sentiment/analyze": "Analyze sentiment for a single text",
            "/sentiment/batch": "Analyze sentiment for multiple texts"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to verify if the API and model are operational."""
    model_loaded = False
    
    try:
        analyzer = get_analyzer()
        model_loaded = hasattr(analyzer, 'model') and analyzer.model is not None
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API is experiencing issues: {str(e)}")
    
    return HealthResponse(
        status="healthy", 
        model_loaded=model_loaded
    )

@app.post("/sentiment/analyze", response_model=SentimentResponse, tags=["Sentiment Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment for a single text.
    
    Returns sentiment classification (positive, negative, neutral) along with confidence scores.
    """
    try:
        analyzer = get_analyzer()
        result = analyzer.predict_sentiment(request.text, return_scores=True)
        
        return SentimentResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            scores=result['scores']
        )
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.post("/sentiment/batch", response_model=BatchSentimentResponse, tags=["Sentiment Analysis"])
def batch_analyze_sentiment(request: BatchSentimentRequest):
    """
    Analyze sentiment for multiple texts.
    
    Returns sentiment classifications and confidence scores for each text.
    """
    try:
        analyzer = get_analyzer()
        results = analyzer.predict_sentiment(request.texts, return_scores=True)
        sentiment_arr = []
        for result in results:
            sentiment_arr.append(result['sentiment'])
        return BatchSentimentResponse(
            results=[
                SentimentResponse(
                    text=result['text'],
                    sentiment=result['sentiment'],
                    confidence=result['confidence'],
                    scores=result['scores']
                ) for result in results
            ],
            count=len(results),
            sentiment_mod =st.mode(sentiment_arr)            
        )
    except Exception as e:
        logger.error(f"Error batch analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error batch analyzing sentiment: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when script is executed directly
    uvicorn.run("sentiment_api:app", host="0.0.0.0", port=8000, reload=True)
