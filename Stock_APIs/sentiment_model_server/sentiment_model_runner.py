import os
import argparse
import logging
from enhanced_sentiment_model import EnhancedStockSentimentAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_model.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate enhanced stock news sentiment model')
    
    parser.add_argument('--data_dir', type=str, default=r'E:\Stock Market GP\Reddit\news\sentiment_data',
                        help='Directory containing sentiment data CSV files')
    
    parser.add_argument('--output_dir', type=str, default=r'E:\Stock Market GP\Reddit\news\sentiment_model_output',
                        help='Directory to save models and outputs')
    
    parser.add_argument('--architecture', type=str, choices=['lstm', 'cnn', 'hybrid'], default='hybrid',
                        help='Model architecture for enhanced model')
    
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training and use saved model for predictions')
    
    parser.add_argument('--predict', type=str, nargs='*',
                        help='Predict sentiment for given text')
    
    args = parser.parse_args()
    
    # Create output directory if doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up model directory for the enhanced model
    model_dir = os.path.join(args.output_dir, f'enhanced_{args.architecture}_model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Create the enhanced analyzer
    analyzer = EnhancedStockSentimentAnalyzer(args.data_dir, model_dir)
    
    # If prediction mode
    if args.predict:
        for text in args.predict:
            prediction = analyzer.predict_sentiment(text, return_scores=True)
            print(f"\nText: {text}")
            print(f"Sentiment: {prediction['sentiment']}")
            
            if 'confidence' in prediction:
                print(f"Confidence: {prediction['confidence']:.4f}")
            if 'scores' in prediction:
                for sentiment, score in prediction['scores'].items():
                    print(f"  {sentiment}: {score:.4f}")
    else:
        # Train or eval mode
        analyzer.run_pipeline(model_type=args.architecture, train_model=not args.skip_training, epochs=args.epochs)

if __name__ == "__main__":
    main()
