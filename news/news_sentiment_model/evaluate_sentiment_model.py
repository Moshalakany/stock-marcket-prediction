
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import logging
from enhanced_sentiment_model import EnhancedStockSentimentAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    Load the combined news dataset
    
    Parameters:
    file_path (str): Path to the combined CSV file
    
    Returns:
    pd.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading dataset from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def take_random_sample(df, sample_size=1000, random_seed=42):
    """
    Take a random sample from the dataset
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    sample_size (int): Size of the sample to take
    random_seed (int): Random seed for reproducibility
    
    Returns:
    pd.DataFrame: Sample dataframe
    """
    if df is None or len(df) == 0:
        logger.error("No data to sample")
        return None
    
    # Adjust sample size if larger than dataset
    if sample_size > len(df):
        logger.warning(f"Sample size {sample_size} is larger than dataset size {len(df)}. Using full dataset.")
        return df
    
    logger.info(f"Taking random sample of {sample_size} records")
    
    # Take random sample
    sample = df.sample(n=sample_size, random_state=random_seed)
    
    return sample

def evaluate_model_on_sample(analyzer, sample_df, output_dir):
    """
    Evaluate the sentiment model on a sample dataset
    
    Parameters:
    analyzer (EnhancedStockSentimentAnalyzer): Initialized sentiment analyzer
    sample_df (pd.DataFrame): Sample dataframe
    output_dir (str): Directory to save results
    
    Returns:
    tuple: Results dataframe and metrics dictionary
    """
    if sample_df is None or len(sample_df) == 0:
        logger.error("No sample data to evaluate")
        return None, None
    
    # Check if 'title' column exists
    if 'title' not in sample_df.columns:
        logger.error("Dataset doesn't contain 'title' column")
        return None, None
    
    # Check if we have ground truth for evaluation
    has_ground_truth = 'sentiment_class' in sample_df.columns
    
    logger.info(f"Evaluating model on {len(sample_df)} samples (ground truth available: {has_ground_truth})")
    
    # Get list of texts to classify
    texts = sample_df['title'].tolist()
    
    # Predict sentiment for each text
    predictions = []
    logger.info("Classifying texts... This may take a while.")
    
    # Process in batches to show progress
    batch_size = 100
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        logger.info(f"Processing batch {i+1}/{num_batches} ({start_idx+1}-{end_idx} of {len(texts)})")
        batch_predictions = [analyzer.predict_sentiment(text, return_scores=True) for text in batch_texts]
        predictions.extend(batch_predictions)
    
    # Create results dataframe
    results_df = sample_df.copy()
    results_df['predicted_sentiment'] = [p['sentiment'] for p in predictions]
    results_df['confidence'] = [p['confidence'] for p in predictions]
    
    # Add score columns for each sentiment
    sentiment_classes = ['negative', 'neutral', 'positive']
    for sentiment in sentiment_classes:
        results_df[f'score_{sentiment}'] = [p['scores'].get(sentiment, 0) for p in predictions]
    
    # Save results
    results_path = os.path.join(output_dir, 'sentiment_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Evaluate if ground truth is available
    metrics = {}
    if has_ground_truth:
        # Map sentiment classes to ensure consistent format
        sentiment_map = {
            'positive': 'positive',
            'negative': 'negative', 
            'neutral': 'neutral',
            'pos': 'positive',
            'neg': 'negative',
            'neu': 'neutral'
        }
        
        # Normalize ground truth labels
        results_df['sentiment_class'] = results_df['sentiment_class'].str.lower()
        results_df['sentiment_class'] = results_df['sentiment_class'].map(
            lambda x: sentiment_map.get(x, 'neutral') if isinstance(x, str) else 'neutral'
        )
        
        # Calculate accuracy
        accuracy = accuracy_score(results_df['sentiment_class'], results_df['predicted_sentiment'])
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(
            results_df['sentiment_class'],
            results_df['predicted_sentiment'],
            target_names=sentiment_classes,
            output_dict=True
        )
        
        # Log classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            results_df['sentiment_class'],
            results_df['predicted_sentiment'],
            target_names=sentiment_classes
        ))
        
        # Generate confusion matrix
        cm = confusion_matrix(
            results_df['sentiment_class'], 
            results_df['predicted_sentiment'],
            labels=sentiment_classes
        )
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    return results_df, metrics

def plot_results(results_df, metrics, output_dir):
    """
    Plot evaluation results
    
    Parameters:
    results_df (pd.DataFrame): Results dataframe
    metrics (dict): Evaluation metrics
    output_dir (str): Directory to save plots
    """
    if results_df is None:
        logger.error("No results to plot")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 10))
    
    # 1. Plot distribution of predicted sentiments
    plt.subplot(2, 2, 1)
    sns.countplot(x='predicted_sentiment', data=results_df, order=['negative', 'neutral', 'positive'])
    plt.title('Distribution of Predicted Sentiments')
    plt.xticks(rotation=45)
    
    # 2. Plot confidence distribution
    plt.subplot(2, 2, 2)
    sns.histplot(results_df['confidence'], bins=20, kde=True)
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence Score')
    
    # 3. Plot ground truth vs predicted if available
    has_ground_truth = 'sentiment_class' in results_df.columns
    
    if has_ground_truth:
        plt.subplot(2, 2, 3)
        ground_truth_counts = results_df['sentiment_class'].value_counts().reindex(['negative', 'neutral', 'positive'], fill_value=0)
        predicted_counts = results_df['predicted_sentiment'].value_counts().reindex(['negative', 'neutral', 'positive'], fill_value=0)
        
        width = 0.35
        x = np.arange(len(ground_truth_counts.index))
        
        plt.bar(x - width/2, ground_truth_counts.values, width, label='Ground Truth')
        plt.bar(x + width/2, predicted_counts.values, width, label='Predicted')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Ground Truth vs Predicted Sentiment')
        plt.xticks(x, ground_truth_counts.index, rotation=45)
        plt.legend()
        
        # 4. Plot confusion matrix
        if 'confusion_matrix' in metrics:
            plt.subplot(2, 2, 4)
            cm = metrics['confusion_matrix']
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive']
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
    else:
        # If no ground truth, plot distribution of scores
        plt.subplot(2, 2, 3)
        avg_scores = {
            'negative': results_df['score_negative'].mean(),
            'neutral': results_df['score_neutral'].mean(),
            'positive': results_df['score_positive'].mean()
        }
        sns.barplot(x=list(avg_scores.keys()), y=list(avg_scores.values()))
        plt.title('Average Sentiment Scores')
        plt.xticks(rotation=45)
        
        # 4. Plot top 10 most confident predictions
        plt.subplot(2, 2, 4)
        top10 = results_df.sort_values('confidence', ascending=False).head(10)
        sns.barplot(x='confidence', y='predicted_sentiment', data=top10, orient='h')
        plt.title('Top 10 Most Confident Predictions')
        plt.tight_layout()
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'sentiment_evaluation_plots.png'))
    logger.info(f"Plots saved to {output_dir}/sentiment_evaluation_plots.png")
    
    # If we have ground truth, also save the classification report as an image
    if has_ground_truth and 'classification_report' in metrics:
        plt.figure(figsize=(10, 6))
        report = metrics['classification_report']
        
        # Create table-like visualization
        plt.axis('off')
        
        header = ['Sentiment', 'Precision', 'Recall', 'F1-score', 'Support']
        cell_text = []
        
        for sentiment in ['negative', 'neutral', 'positive']:
            row = [
                sentiment,
                f"{report[sentiment]['precision']:.2f}",
                f"{report[sentiment]['recall']:.2f}",
                f"{report[sentiment]['f1-score']:.2f}",
                f"{report[sentiment]['support']}"
            ]
            cell_text.append(row)
        
        # Add accuracy row
        accuracy = report['accuracy']
        cell_text.append(['accuracy', '', '', f"{accuracy:.2f}", f"{report['macro avg']['support']}"])
        
        table = plt.table(
            cellText=cell_text,
            colLabels=header,
            loc='center',
            cellLoc='center',
            colWidths=[0.15, 0.15, 0.15, 0.15, 0.15]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        
        plt.title('Classification Report', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save the classification report
        plt.savefig(os.path.join(output_dir, 'classification_report.png'))
        logger.info(f"Classification report saved to {output_dir}/classification_report.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluate stock news sentiment model on a random sample')
    
    parser.add_argument('--data_path', type=str, 
                        default=r'E:\Stock Market GP\Reddit\news\sentiment_model_output\enhanced_hybrid_model\combined_stock_news.csv',
                        help='Path to the combined stock news CSV file')
    
    parser.add_argument('--model_dir', type=str, 
                        default=r'E:\Stock Market GP\Reddit\news\sentiment_model_output\enhanced_hybrid_model',
                        help='Directory containing the trained sentiment model')
    
    parser.add_argument('--output_dir', type=str, 
                        default=r'E:\Stock Market GP\Reddit\news\sentiment_evaluation_results',
                        help='Directory to save evaluation results')
    
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of random news samples to evaluate')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 1. Load the dataset
    df = load_dataset(args.data_path)
    if df is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # 2. Take a random sample
    sample_df = take_random_sample(df, args.sample_size, args.random_seed)
    if sample_df is None:
        logger.error("Failed to create sample. Exiting.")
        return
    
    # 3. Initialize the analyzer
    analyzer = EnhancedStockSentimentAnalyzer(
        data_dir=os.path.dirname(args.data_path),
        output_dir=args.model_dir
    )
    
    # 4. Evaluate the model on the sample
    results_df, metrics = evaluate_model_on_sample(analyzer, sample_df, args.output_dir)
    if results_df is None:
        logger.error("Evaluation failed. Exiting.")
        return
    
    # 5. Plot and save results
    plot_results(results_df, metrics, args.output_dir)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
