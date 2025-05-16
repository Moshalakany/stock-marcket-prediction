import os
import numpy as np
import re
import logging
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    # Download NLTK resources if not already available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedStockSentimentAnalyzer:
    def __init__(self, data_dir, model_dir):
        """
        Initialize the Enhanced Stock Sentiment Analyzer
        
        Parameters:
        data_dir (str): Directory containing data files
        model_dir (str): Directory containing model files
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = 150  # Default, will be adjusted based on model
        
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
        
        # Try to load model files on initialization
        try:
            self.model = self.load_model()
            self.tokenizer = self.load_tokenizer()
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model on initialization. Will try again when needed: {str(e)}")

    def load_model(self):
        """Load the trained sentiment model"""
        model_path = os.path.join(self.model_dir, 'best_model.keras')
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            return None
            
        try:
            return load_model(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def load_tokenizer(self):
        """Load the trained tokenizer"""
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            logger.warning(f"Tokenizer file not found at {tokenizer_path}")
            return None
            
        try:
            with open(tokenizer_path, 'rb') as handle:
                return pickle.load(handle)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            return None

    def clean_text(self, text):
        """
        Clean and normalize text data
        
        Parameters:
        text (str): Text to clean
        
        Returns:
        str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning without NLTK if not available
        if not NLTK_AVAILABLE:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            # Lowercase
            return text.lower()
        
        # Enhanced cleaning with NLTK
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        text = text.lower()
        
        # Tokenize
        word_tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        words = [self.lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
        
        return ' '.join(words)

    def predict_sentiment(self, text, return_scores=False):
        """
        Predict sentiment for new text
        
        Parameters:
        text (str or list): Text or list of texts to analyze
        return_scores (bool): Whether to return confidence scores
        
        Returns:
        dict or list: Sentiment predictions
        """
        # Make sure model and tokenizer are loaded
        if self.model is None:
            self.model = self.load_model()
            if self.model is None:
                return {"error": "Model not available"}
                
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
            if self.tokenizer is None:
                return {"error": "Tokenizer not available"}
        
        # Handle single text or list of texts
        if isinstance(text, list):
            # Handle batch prediction
            results = []
            for single_text in text:
                result = self._predict_single_text(single_text, return_scores)
                results.append(result)
            return results
        else:
            # Handle single text prediction
            return self._predict_single_text(text, return_scores)

    def _predict_single_text(self, text, return_scores=False):
        """Process a single text and return prediction"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post', truncating='post')
        
        # Make prediction
        prediction = self.model.predict(padded_sequence, verbose=0)
        
        # Get sentiment class
        sentiment_idx = np.argmax(prediction, axis=1)[0]
        sentiment_classes = ['negative', 'neutral', 'positive']
        sentiment = sentiment_classes[sentiment_idx]
        
        # Calculate confidence
        confidence = float(np.max(prediction, axis=1)[0])
        
        # Prepare result
        result = {'text': text, 'sentiment': sentiment}
        
        if return_scores:
            result['confidence'] = confidence
            result['scores'] = {sentiment_classes[i]: float(score) for i, score in enumerate(prediction[0])}
        
        return result