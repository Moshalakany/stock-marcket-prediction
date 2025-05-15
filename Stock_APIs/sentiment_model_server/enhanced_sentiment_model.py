import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional
from tensorflow.keras.layers import GlobalMaxPooling1D, Dense, Dropout, Concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedStockSentimentAnalyzer:
    def __init__(self, data_dir, output_dir):
        """
        Initialize the Enhanced Stock Sentiment Analyzer
        
        Parameters:
        data_dir (str): Directory containing CSV files with news data
        output_dir (str): Directory to save processed data and models
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.combined_data = None
        self.model = None
        self.tokenizer = None
        self.max_len = 150
        self.lemmatizer = WordNetLemmatizer()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Download NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
    
    def load_and_combine_data(self):
        """
        Load all CSV files from data_dir and combine them into one DataFrame
        """
        logger.info(f"Loading data from {self.data_dir}")
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not all_files:
            logger.error(f"No CSV files found in {self.data_dir}")
            return False
            
        dataframes = []
        
        for file in all_files:
            try:
                file_path = os.path.join(self.data_dir, file)
                df = pd.read_csv(file_path)
                
                # Skip empty files
                if df.empty:
                    logger.warning(f"Skipping empty file: {file}")
                    continue
                
                # Check if required columns are present
                required_columns = ['title', 'sentiment_class', 'sentiment_score']
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"Skipping file {file} due to missing required columns")
                    continue
                
                # Add ticker symbol (filename without extension)
                ticker = os.path.splitext(file)[0]
                df['ticker'] = ticker
                
                dataframes.append(df)
                logger.info(f"Loaded {file} with {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                
        if not dataframes:
            logger.error("No dataframes were successfully loaded")
            return False
            
        self.combined_data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(dataframes)} files into dataset with {len(self.combined_data)} records")
        
        # Save combined data
        combined_path = os.path.join(self.output_dir, 'combined_stock_news.csv')
        self.combined_data.to_csv(combined_path, index=False)
        logger.info(f"Saved combined data to {combined_path}")
        
        return True
    
    def preprocess_data(self):
        """
        Preprocess the text data and prepare for modeling
        """
        if self.combined_data is None or len(self.combined_data) == 0:
            logger.error("No data to preprocess. Run load_and_combine_data first.")
            return False
        
        logger.info("Starting data preprocessing...")
        
        # Drop any rows with missing titles
        self.combined_data = self.combined_data.dropna(subset=['title'])
        
        # Ensure sentiment_class is properly formatted
        self.combined_data['sentiment_class'] = self.combined_data['sentiment_class'].str.lower()
        
        # Map sentiment classes to ensure consistent labeling
        sentiment_map = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'pos': 'positive',
            'neg': 'negative',
            'neu': 'neutral'
        }
        
        self.combined_data['sentiment_class'] = self.combined_data['sentiment_class'].map(
            lambda x: sentiment_map.get(x, 'neutral') if isinstance(x, str) else 'neutral'
        )
        
        # Clean text
        logger.info("Cleaning text data...")
        self.combined_data['cleaned_title'] = self.combined_data['title'].apply(self.clean_text)
        
        # Encode sentiment classes
        logger.info("Encoding sentiment classes...")
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        self.combined_data['sentiment_encoded'] = label_encoder.fit_transform(self.combined_data['sentiment_class'])
        
        # Save the label encoder for later use
        with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Store class mappings
        self.class_mappings = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        logger.info(f"Class mappings: {self.class_mappings}")
        
        # Add sentiment polarity from score where available
        if 'sentiment_score' in self.combined_data.columns:
            self.combined_data['sentiment_polarity'] = self.combined_data['sentiment_score'].apply(
                lambda x: (x - 0.5) * 2 if not pd.isna(x) else 0
            )
        
        # Save preprocessed data
        preprocessed_path = os.path.join(self.output_dir, 'preprocessed_stock_news.csv')
        self.combined_data.to_csv(preprocessed_path, index=False)
        logger.info(f"Saved preprocessed data to {preprocessed_path}")
        
        # Display data summary
        logger.info(f"Data summary after preprocessing:")
        logger.info(f"Total records: {len(self.combined_data)}")
        logger.info(f"Sentiment distribution: {self.combined_data['sentiment_class'].value_counts()}")
        
        return True
    
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
    
    def prepare_training_data(self, max_words=15000, test_size=0.2, val_size=0.1):
        """
        Prepare the data for training the model
        
        Parameters:
        max_words (int): Maximum number of words for the tokenizer
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        
        Returns:
        tuple: Training, validation, and test data splits and metadata
        """
        if self.combined_data is None:
            logger.error("No data to prepare. Run preprocess_data first.")
            return None
            
        logger.info("Preparing training data...")
        
        # Tokenize text
        logger.info(f"Tokenizing with vocabulary size of {max_words}...")
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.combined_data['cleaned_title'])
        
        # Save tokenizer
        with open(os.path.join(self.output_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
        
        # Convert text to sequences
        sequences = tokenizer.texts_to_sequences(self.combined_data['cleaned_title'])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Get targets
        targets = self.combined_data['sentiment_encoded']
        
        # Convert to one-hot encoding for multi-class classification
        targets_categorical = tf.keras.utils.to_categorical(targets)
        
        # Split data
        logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}...")
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, targets_categorical, test_size=test_size, random_state=42, stratify=targets_categorical
        )
        
        # Further split training data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train
        )
        
        # Save tokenizer and other metadata for later use
        self.tokenizer = tokenizer
        self.vocab_size = min(max_words, len(tokenizer.word_index) + 1)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, model_type='hybrid', embedding_dim=300):
        """
        Build an enhanced deep learning model for sentiment analysis
        
        Parameters:
        model_type (str): Type of model architecture ('lstm', 'cnn', 'hybrid')
        embedding_dim (int): Dimensionality of embeddings
        
        Returns:
        model: Compiled Keras model
        """
        logger.info(f"Building {model_type.upper()} model...")
        
        input_layer = Input(shape=(self.max_len,))
        
        # Embedding layer
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=embedding_dim,
            input_length=self.max_len
        )(input_layer)
        
        if model_type == 'lstm':
            # LSTM-based model
            x = Dropout(0.2)(embedding_layer)
            x = Bidirectional(LSTM(128, return_sequences=True))(x)
            x = Dropout(0.2)(x)
            x = Bidirectional(LSTM(64))(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            
        elif model_type == 'cnn':
            # CNN-based model
            conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding_layer)
            pool1 = GlobalMaxPooling1D()(conv1)
            
            conv2 = Conv1D(filters=128, kernel_size=4, activation='relu')(embedding_layer)
            pool2 = GlobalMaxPooling1D()(conv2)
            
            conv3 = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
            pool3 = GlobalMaxPooling1D()(conv3)
            
            x = Concatenate()([pool1, pool2, pool3])
            x = Dropout(0.2)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            
        else:  # hybrid model (default)
            # LSTM branch
            lstm_branch = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
            lstm_branch = Dropout(0.2)(lstm_branch)
            lstm_branch = Bidirectional(LSTM(64))(lstm_branch)
            
            # CNN branch - multiple filter sizes
            conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding_layer)
            pool1 = GlobalMaxPooling1D()(conv1)
            
            conv2 = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
            pool2 = GlobalMaxPooling1D()(conv2)
            
            # Combine branches
            x = Concatenate()([lstm_branch, pool1, pool2])
            x = Dropout(0.2)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
        
        # Output layer - 3 classes (positive, negative, neutral)
        output_layer = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary(print_fn=logger.info)
        
        return model

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=15):
        """
        Train the sentiment analysis model
        
        Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size (int): Batch size
        epochs (int): Maximum number of epochs
        
        Returns:
        history: Training history
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            return None
            
        logger.info(f"Training model with batch_size={batch_size}, epochs={epochs}...")
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_model.keras'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and training history
        self.model.save(os.path.join(self.output_dir, 'final_model.keras'))
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data
        
        Parameters:
        X_test: Test features
        y_test: Test targets
        
        Returns:
        dict: Evaluation metrics
        """
        # Try to load best model if available
        best_model_path = os.path.join(self.output_dir, 'best_model.keras')
        if os.path.exists(best_model_path):
            logger.info("Loading best model for evaluation...")
            self.model = load_model(best_model_path)
        
        if self.model is None:
            logger.error("No model available for evaluation")
            return None
        
        logger.info("Evaluating model on test data...")
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test loss: {loss:.4f}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Generate predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        # Log classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, ['negative', 'neutral', 'positive'])
        
        # Create evaluation metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_training_history(self, history):
        """
        Plot and save training history
        
        Parameters:
        history: Model training history
        """
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot learning rate if available
        if 'lr' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        logger.info(f"Training history plot saved to {self.output_dir}/training_history.png")
        
    def plot_confusion_matrix(self, cm, class_names):
        """
        Plot and save confusion matrix
        
        Parameters:
        cm: Confusion matrix
        class_names: List of class names
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names, 
            yticklabels=class_names
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        logger.info(f"Confusion matrix saved to {self.output_dir}/confusion_matrix.png")
    
    def predict_sentiment(self, text, return_scores=False):
        """
        Predict sentiment for new text
        
        Parameters:
        text (str or list): Text or list of texts to analyze
        return_scores (bool): Whether to return confidence scores
        
        Returns:
        list or dict: Sentiment predictions
        """
        if self.model is None:
            # Try to load model
            model_path = os.path.join(self.output_dir, 'best_model.keras')
            if os.path.exists(model_path):
                self.model = load_model(model_path)
            else:
                logger.error("No model available for prediction")
                return None
        
        if self.tokenizer is None:
            # Try to load tokenizer
            tokenizer_path = os.path.join(self.output_dir, 'tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
            else:
                logger.error("No tokenizer available for prediction")
                return None
        
        # Handle single text or list of texts
        single_text = False
        if isinstance(text, str):
            text = [text]
            single_text = True
        
        # Clean and preprocess texts
        cleaned_texts = [self.clean_text(t) for t in text]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Make predictions
        predictions = self.model.predict(padded_sequences)
        
        # Process results
        results = []
        sentiment_classes = ['negative', 'neutral', 'positive']
        
        for i, pred in enumerate(predictions):
            sentiment_idx = np.argmax(pred)
            sentiment = sentiment_classes[sentiment_idx]
            
            result = {'text': text[i], 'sentiment': sentiment}
            
            if return_scores:
                result['confidence'] = float(pred[sentiment_idx])
                result['scores'] = {sentiment_classes[j]: float(score) for j, score in enumerate(pred)}
            
            results.append(result)
        
        # Return single result if input was a single text
        if single_text:
            return results[0]
        
        return results
    
    def run_pipeline(self, model_type='hybrid', train_model=True, epochs=15):
        """
        Run the complete pipeline from data loading to model evaluation
        
        Parameters:
        model_type (str): Type of model architecture ('lstm', 'cnn', 'hybrid')
        train_model (bool): Whether to train the model or just prepare data
        epochs (int): Number of epochs for training
        
        Returns:
        dict: Evaluation metrics if model is trained and evaluated
        """
        # 1. Load and combine data
        if not self.load_and_combine_data():
            logger.error("Failed to load and combine data")
            return None
        
        # 2. Preprocess data
        if not self.preprocess_data():
            logger.error("Failed to preprocess data")
            return None
        
        # 3. Prepare training data
        data = self.prepare_training_data()
        if data is None:
            logger.error("Failed to prepare training data")
            return None
        
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        if not train_model:
            logger.info("Data preparation complete. Skipping model training as requested.")
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        
        # 4. Build model
        self.model = self.build_model(model_type=model_type)
        
        # 5. Train model
        history = self.train_model(X_train, y_train, X_val, y_val, epochs=epochs)
        if history is None:
            logger.error("Failed to train model")
            return None
        
        # 6. Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        return metrics


# Example usage when run as a script
if __name__ == "__main__":
    # Set paths
    data_directory = "/e:/Stock Market GP/Reddit/news/sentiment_data"
    output_directory = "/e:/Stock Market GP/Reddit/news/enhanced_sentiment_model_output"
    
    # Initialize analyzer
    analyzer = EnhancedStockSentimentAnalyzer(data_directory, output_directory)
    
    # Run the full pipeline with hybrid model (CNN + LSTM)
    metrics = analyzer.run_pipeline(model_type='hybrid', train_model=True, epochs=15)
    
    if metrics:
        logger.info(f"Model training complete. Final accuracy: {metrics['accuracy']:.4f}")
        
        # Example prediction
        sample_texts = [
            "Company reports record profits exceeding expectations",
            "Stock plummets after poor quarterly results",
            "Company announces new CEO appointment"
        ]
        
        for text in sample_texts:
            prediction = analyzer.predict_sentiment(text, return_scores=True)
            logger.info(f"Text: {text}")
            logger.info(f"Sentiment: {prediction['sentiment']} (confidence: {prediction['confidence']:.4f})")
