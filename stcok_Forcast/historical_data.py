import requests as re 
import pandas as pd
import os
import time
from tqdm import tqdm  # For progress bar, install with pip if not available

def get_historical_data(symbol):
    url = f"https://api.nasdaq.com/api/quote/{symbol}/historical?assetclass=stocks&fromdate=2015-03-14&limit=9999&todate=2025-03-14&random=30"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = re.get(url, headers=headers).json()
        
        # Check if the API request was successful
        if response and 'data' in response and 'tradesTable' in response['data'] and 'rows' in response['data']['tradesTable']:
            # Extract the historical data rows
            historical_data = response['data']['tradesTable']['rows']
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(historical_data)
            
            # Convert date column to datetime format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Convert numeric columns to float
            numeric_columns = ['close', 'volume', 'open', 'high', 'low']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].str.replace('$', '', regex=False)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        else:
            print(f"Failed to retrieve data for {symbol}. Response structure is different than expected.")
            return None
    except Exception as e:
        print(f"Error retrieving historical data for {symbol}: {str(e)}")
        return None

def get_all_sp500_historical_data():
    """
    Get historical data for all S&P 500 companies and save to CSV files.
    """
    # Path to the S&P 500 companies CSV file
    sp500_file = r"stcok_Forcast\sp500_companies.csv"
    
    # Create directory for historical data if it doesn't exist
    output_dir = r"stcok_Forcast\data\historical_prices"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Read S&P 500 companies
    try:
        sp500_df = pd.read_csv(sp500_file)
        if 'Symbol' not in sp500_df.columns:
            print("Error: 'Symbol' column not found in the S&P 500 companies CSV file.")
            return
        
        symbols = sp500_df['Symbol'].tolist()
        print(f"Found {len(symbols)} symbols in the S&P 500 list.")
        
        # Track successful and failed downloads
        successful = 0
        failed = []
        
        # Process each symbol
        for symbol in tqdm(symbols, desc="Fetching data"):
            try:
                # Get historical data
                df = get_historical_data(symbol)
                
                if df is not None and not df.empty:
                    # Save to CSV
                    output_path = os.path.join(output_dir, f"{symbol}.csv")
                    df.to_csv(output_path, index=False)
                    successful += 1
                    
                    # Add a delay to avoid hitting rate limits
                    time.sleep(1)
                else:
                    failed.append(symbol)
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                failed.append(symbol)
        
        # Print summary
        print(f"\nDownload completed. Successfully downloaded data for {successful} stocks.")
        if failed:
            print(f"Failed to download data for {len(failed)} stocks: {', '.join(failed)}")
        
    except Exception as e:
        print(f"Error reading S&P 500 companies file: {str(e)}")

# Example usage
if __name__ == "__main__":
    # To get data for a single stock
    # symbol = "AAPL"
    # data = get_historical_data(symbol)
    # if data is not None:
    #     print(f"Retrieved {len(data)} records for {symbol}")
    #     print(data.head())
    
    # To get data for all S&P 500 stocks
    get_all_sp500_historical_data()