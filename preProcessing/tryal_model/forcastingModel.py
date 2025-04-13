import os
import pandas as pd
import matplotlib.pyplot as plt
from general import StockForecastingModel
import time

class StockForecaster:
    def __init__(self, data_dir='../MergedData', output_dir='output', stock_list=None):
        """
        Initialize the stock forecaster
        
        Args:
            data_dir (str): Directory containing merged stock data files
            output_dir (str): Directory to save model outputs
            stock_list (list): List of stock symbols to process. If None, all CSV files in data_dir will be processed
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.stock_list = stock_list
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of stocks to process if stock_list is None
        if self.stock_list is None:
            self.stock_list = self._find_stock_files()

    def _find_stock_files(self):
        """Find all stock data files in the data directory"""
        stock_files = []
        
        for file in os.listdir(self.data_dir):
            if file.endswith('_merged.csv'):
                # Extract stock symbol from filename (assuming format is SYMBOL_merged.csv)
                stock_symbol = file.split('_merged.csv')[0]
                stock_files.append(stock_symbol)
        
        return stock_files

    def train_all_models(self, seq_length=30, pred_days=7):
        """
        Train models for all stocks in the stock list
        
        Args:
            seq_length (int): Number of days to use as input sequence
            pred_days (int): Number of days to predict ahead
        """
        # Dictionary to store predictions for all stocks
        all_predictions = {}
        
        # Track training time
        start_time = time.time()
        
        print(f"Starting training for {len(self.stock_list)} stocks...")
        
        for idx, stock in enumerate(self.stock_list):
            print(f"\n[{idx+1}/{len(self.stock_list)}] Training model for {stock}...")
            
            # File path for this stock
            file_path = os.path.join(self.data_dir, f"{stock}_merged.csv")
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found for {stock} at {file_path}. Skipping...")
                continue
            
            try:
                # Create and train model for this stock
                model = StockForecastingModel(
                    stock_symbol=stock,
                    data_path=file_path,
                    output_dir=self.output_dir,
                    seq_length=seq_length,
                    pred_days=pred_days
                )
                
                # Train and get predictions
                future_predictions = model.train_and_evaluate()
                
                # Store predictions
                all_predictions[stock] = future_predictions
                
            except Exception as e:
                print(f"Error training model for {stock}: {str(e)}")
                continue
        
        # Calculate total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining completed for all stocks.")
        print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Create summary plot of all predictions
        self._create_summary_plot(all_predictions)
        
        return all_predictions
    
    def _create_summary_plot(self, all_predictions):
        """Create a summary plot of predictions for all stocks"""
        if not all_predictions:
            print("No predictions available for summary plot.")
            return
        
        # Create a plot with predictions for all stocks
        plt.figure(figsize=(15, 10))
        
        for stock, predictions in all_predictions.items():
            # Normalize predictions to percentage change from first day for better comparison
            prices = predictions['Predicted_Close'].values
            normalized = (prices / prices[0] - 1) * 100
            plt.plot(predictions['Date'], normalized, label=stock)
        
        plt.title('Predicted Price Movement (% change) - All Stocks')
        plt.xlabel('Date')
        plt.ylabel('Price Change (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_stocks_predictions.png')
        
        # Save summary to CSV
        summary_data = []
        for stock, predictions in all_predictions.items():
            first_price = predictions['Predicted_Close'].iloc[0]
            last_price = predictions['Predicted_Close'].iloc[-1]
            change = (last_price - first_price) / first_price * 100
            
            summary_data.append({
                'Stock': stock,
                'Start_Price': first_price,
                'End_Price': last_price,
                'Change_Percent': change,
                'Trend': 'Up' if change > 0 else 'Down'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.sort_values('Change_Percent', ascending=False, inplace=True)
        summary_df.to_csv(f'{self.output_dir}/prediction_summary.csv', index=False)
        
        print(f"Summary plot and data saved to {self.output_dir}")


# Example usage
if __name__ == "__main__":
    # Define stocks to process (or leave as None to process all available stocks)
    stocks_to_process = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  
    
    # Initialize forecaster
    forecaster = StockForecaster(
        data_dir='../MergedData',
        output_dir='output',
        stock_list=stocks_to_process
    )
    
    # Train models for all stocks
    predictions = forecaster.train_all_models(seq_length=30, pred_days=7)