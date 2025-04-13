import os
import pandas as pd
import logging
import traceback
from data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_data_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_stock_data(ticker):
    """
    Process data for a single stock symbol.
    
    Args:
        ticker (str): Stock symbol (e.g., 'AAPL')
    
    Returns:
        pd.DataFrame: Processed data with technical and sentiment indicators
    """
    # Create an instance of the DataProcessor class
    processor = DataProcessor()
    
    logger.info(f"Starting data processing for {ticker}")
    
    # Define the paths to the data files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    historical_data_path = os.path.join(base_dir, 'data', 'historical_prices', f'{ticker}.csv')
    
    # Load historical price data
    try:
        historical_data = pd.read_csv(historical_data_path)
        logger.info(f"Loaded historical data with {len(historical_data)} rows")
        
        # Ensure column names match expected format for technical indicators
        if 'Date' not in historical_data.columns and 'date' in historical_data.columns:
            historical_data.rename(columns={'date': 'Date'}, inplace=True)
        if 'Close' not in historical_data.columns and 'close' in historical_data.columns:
            historical_data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
        
        # Ensure date column is datetime
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
        # Sort by date (oldest to newest)
        historical_data = historical_data.sort_values('Date')
            
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        historical_data_with_indicators = processor.compute_technical_indicators(historical_data)
        logger.info(f"Added {len(historical_data_with_indicators.columns) - len(historical_data.columns)} technical indicators")
        
    except Exception as e:
        logger.error(f"Error loading historical data or calculating indicators: {e}")
        logger.error(f"Tried to load from: {historical_data_path}")
        logger.error(traceback.format_exc())
        return None
    
    # Process sentiment data
    try:
        # Process sentiment data for the ticker
        processed_sentiment = processor.process_sentiment_data(ticker)
        logger.info(f"Processed sentiment data with {len(processed_sentiment)} rows")
        
    except Exception as e:
        logger.error(f"Error processing sentiment data: {e}")
        logger.error(traceback.format_exc())
        
        # Even if sentiment processing fails, we may still want to return the technical indicators
        # Convert Date to date for consistency if sentiment merging will be skipped
        if 'Date' in historical_data_with_indicators.columns:
            historical_data_with_indicators = historical_data_with_indicators.rename(columns={'Date': 'date'})
        
        # Save what we have so far
        output_dir = os.path.join(base_dir, 'data', 'final')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{ticker}_final_with_indicators.csv')
        historical_data_with_indicators.to_csv(output_path, index=False)
        logger.info(f"Saved data with technical indicators (no sentiment) to {output_path}")
        
        return historical_data_with_indicators
    
    # If both datasets are loaded successfully, merge them on date
    try:
        # Make sure date columns are in the same format
        if 'date' in historical_data_with_indicators.columns:
            historical_data_with_indicators['date'] = pd.to_datetime(historical_data_with_indicators['date'])
        elif 'Date' in historical_data_with_indicators.columns:
            historical_data_with_indicators = historical_data_with_indicators.rename(columns={'Date': 'date'})
            historical_data_with_indicators['date'] = pd.to_datetime(historical_data_with_indicators['date'])
        
        # Calculate sentiment indicators and merge with price data
        merged_data = processor.compute_sentiment_indicators(historical_data_with_indicators, processed_sentiment)
        
        # Ensure data is sorted chronologically
        merged_data = merged_data.sort_values('date')
        
        logger.info(f"Generated final data with {len(merged_data)} rows and {len(merged_data.columns)} columns")
        
        # Save the merged data as the final training data
        output_dir = os.path.join(base_dir, 'data', 'final')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{ticker}_final_with_indicators.csv')
        merged_data.to_csv(output_path, index=False)
        logger.info(f"Saved final data with technical and sentiment indicators to {output_path}")
        
        return merged_data
        
    except Exception as e:
        logger.error(f"Error merging price and sentiment data: {e}")
        logger.error(traceback.format_exc())
        return historical_data_with_indicators

def process_sector_data(sector):
    """
    Process data for all stocks in a given sector.
    
    Args:
        sector (str): Sector name (e.g., 'Technology')
    
    Returns:
        dict: Dictionary mapping stock symbols to their processed dataframes
    """
    # Create an instance of the DataProcessor class
    processor = DataProcessor()
    
    logger.info(f"Starting data processing for sector: {sector}")
    
    # Load SP500 companies data to get stocks in the sector
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sp500_path = os.path.join(base_dir, 'sp500_companies.csv')
        sp500_data = pd.read_csv(sp500_path)
        sector_stocks = sp500_data[sp500_data['Sector'] == sector]['Symbol'].tolist()
        logger.info(f"Found {len(sector_stocks)} stocks in {sector} sector")
    except Exception as e:
        logger.error(f"Error loading SP500 data: {e}")
        logger.error(traceback.format_exc())
        return {}
    
    # Process data for each stock in the sector
    sector_data = {}
    for ticker in sector_stocks:
        try:
            stock_data = process_stock_data(ticker)
            if stock_data is not None and not stock_data.empty:
                sector_data[ticker] = stock_data
                logger.info(f"Successfully processed data for {ticker}")
            else:
                logger.warning(f"No data returned for {ticker}")
        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {e}")
    
    logger.info(f"Completed processing {len(sector_data)} stocks in {sector} sector")
    
    # Create a combined dataframe with sector data
    if sector_data:
        # Create a directory for sector data
        sector_dir = os.path.join(base_dir, 'data', 'sectors')
        os.makedirs(sector_dir, exist_ok=True)
        
        # Save individual stock data to the sector directory
        for ticker, df in sector_data.items():
            df.to_csv(os.path.join(sector_dir, f'{ticker}_sector_{sector}.csv'), index=False)
        
        # Create a combined file with sector name
        sector_filename = os.path.join(sector_dir, f'{sector}_combined.csv')
        
        # Add stock symbol column to each dataframe and combine them
        combined_data = []
        for ticker, df in sector_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = ticker
            combined_data.append(df_copy)
        
        if combined_data:
            sector_combined_df = pd.concat(combined_data, ignore_index=True)
            sector_combined_df.to_csv(sector_filename, index=False)
            logger.info(f"Saved combined sector data to {sector_filename}")
    
    return sector_data

if __name__ == "__main__":
    # Example usage
    # For processing a single stock
    # df = process_stock_data('AAPL')
    
    # For processing a sector
    # sector_data = process_sector_data('Technology')
    pass
