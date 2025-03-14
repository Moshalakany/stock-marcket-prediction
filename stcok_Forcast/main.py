import argparse
import json
import os
import pandas as pd
from datetime import datetime
from forecaster import StockForecaster
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def save_json(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Stock Market Forecasting System')
    
    # Main actions
    parser.add_argument('action', choices=['train', 'predict', 'backtest', 'sectors'], 
                        help='Action to perform')
                        
    # Stock or sector arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--symbol', type=str, help='Stock symbol')
    group.add_argument('--sector', type=str, help='Market sector')
    
    # Common arguments
    parser.add_argument('--period', type=str, default='5y', 
                        help='Data period (e.g., 1y, 5y, max)')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file for results')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--window', type=int, default=20,
                        help='Window size for sequence data')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Prediction horizon in days')
    parser.add_argument('--sentiment', action='store_true',
                        help='Include sentiment analysis')
    
    # Prediction/backtest arguments
    parser.add_argument('--days-ahead', type=int, default=5,
                        help='Days ahead for prediction')
    
    # Sector arguments
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of stocks per sector')
    parser.add_argument('--top', type=int, default=5,
                        help='Show top N predictions')
    
    args = parser.parse_args()
    
    try:
        forecaster = StockForecaster()
        forecaster.window_size = args.window
        forecaster.prediction_horizon = args.horizon
        forecaster.epochs = args.epochs
        
        if args.action == 'sectors':
            # List available sectors
            print("Available sectors:")
            for sector in forecaster.data_processor.sectors:
                stock_count = len(forecaster.data_processor.sector_stocks[sector])
                print(f"- {sector} ({stock_count} stocks)")
            return
            
        # Require either symbol or sector for other actions
        if args.symbol is None and args.sector is None:
            print("Error: Either --symbol or --sector is required.")
            return
            
        # Confirm sector is valid
        if args.sector and args.sector not in forecaster.data_processor.sectors:
            print(f"Error: '{args.sector}' is not a valid sector.")
            print("Use 'sectors' action to see available sectors.")
            return
            
        # Get sector for a symbol if not provided
        if args.symbol and not args.sector:
            symbol_data = forecaster.data_processor.sp500_data
            symbol_info = symbol_data[symbol_data['Symbol'] == args.symbol]
            if not symbol_info.empty:
                args.sector = symbol_info['Sector'].iloc[0]
                print(f"Using sector '{args.sector}' for {args.symbol}")
            else:
                print(f"Error: Could not determine sector for {args.symbol}")
                return
                
        # Execute requested action
        if args.action == 'train':
            if args.symbol:
                # Train single stock
                result = forecaster.train_model(
                    args.symbol, 
                    args.sector, 
                    period=args.period, 
                    window_size=args.window,
                    prediction_horizon=args.horizon,
                    include_sentiment=args.sentiment,
                    epochs=args.epochs
                )
                print(f"Training result for {args.symbol}:")
                print(f"  Success: {result['success']}")
                if result['success']:
                    print(f"  MSE: {result['metrics']['mse']:.6f}")
                    print(f"  MAE: {result['metrics']['mae']:.6f}")
                    print(f"  R²: {result['metrics']['r2']:.6f}")
                    print(f"  Model saved to: {result['model_path']}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
            else:
                # Train sector
                results = forecaster.train_sector_models(
                    args.sector, 
                    limit=args.limit, 
                    period=args.period
                )
                successful = sum(1 for r in results.values() if r['success'])
                print(f"Trained {successful}/{len(results)} models for {args.sector} sector")
                
            # Save results if requested
            if args.output:
                output_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "action": "train",
                    "parameters": vars(args),
                    "results": result if args.symbol else results
                }
                save_json(output_data, args.output)
                print(f"Results saved to {args.output}")
                
        elif args.action == 'predict':
            if args.symbol:
                # Predict single stock
                result = forecaster.predict(
                    args.symbol, 
                    args.sector, 
                    days_ahead=args.days_ahead,
                    period=args.period
                )
                print(f"Prediction for {args.symbol}:")
                if result['success']:
                    print(f"  Current price ({result['last_date']}): ${result['last_price']:.2f}")
                    print(f"  Predicted ({result['prediction_date']}): ${result['predicted_price']:.2f}")
                    print(f"  Expected change: {result['predicted_change_pct']:.2f}%")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
            else:
                # Predict sector
                results = forecaster.predict_sector(
                    args.sector, 
                    limit=args.limit, 
                    days_ahead=args.days_ahead
                )
                successful = sum(1 for r in results.values() if r['success'])
                print(f"Generated {successful}/{len(results)} predictions for {args.sector} sector")
                
                # Show top predictions
                top_predictions = forecaster.get_top_predictions(results, args.top)
                print(f"\nTop {len(top_predictions)} predictions:")
                for i, pred in enumerate(top_predictions, 1):
                    print(f"{i}. {pred['symbol']}: {pred['predicted_change_pct']:.2f}% (${pred['predicted_price']:.2f})")
                
            # Save results if requested
            if args.output:
                output_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "action": "predict",
                    "parameters": vars(args),
                    "results": result if args.symbol else results,
                }
                if not args.symbol:
                    output_data["top_predictions"] = top_predictions
                save_json(output_data, args.output)
                print(f"Results saved to {args.output}")
                
        elif args.action == 'backtest':
            if args.symbol:
                # Backtest single stock
                result = forecaster.backtest(
                    args.symbol, 
                    args.sector, 
                    test_period=args.period,
                    window_size=args.window,
                    prediction_horizon=args.horizon
                )
                print(f"Backtest result for {args.symbol}:")
                if result['success']:
                    print(f"  MSE: {result['metrics']['mse']:.6f}")
                    print(f"  MAE: {result['metrics']['mae']:.6f}")
                    print(f"  R²: {result['metrics']['r2']:.6f}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
                    
                # Save results if requested
                if args.output:
                    output_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "backtest",
                        "parameters": vars(args),
                        "results": result
                    }
                    save_json(output_data, args.output)
                    print(f"Results saved to {args.output}")
            else:
                print("Sector backtesting is not implemented. Use --symbol for backtesting.")
    
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
