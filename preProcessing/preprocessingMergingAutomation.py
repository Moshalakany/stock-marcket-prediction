import dataPreprocessor as dp
import pandas as pd
stocks =pd.read_csv('sp500_companies.csv')['Symbol'].tolist()
print (stocks)
for stock in stocks:
    historical_path = f'historical_prices\{stock}.csv'
    sentiment_path = f'sentiment_data\{stock}.csv'
    try:
        dp.dataPreprocessor(historical_data=historical_path,sentiment_data=sentiment_path,symbol=stock).merge_data()  
    except:
        print(f'files of {stock}not found')
        continue
#dp.dataPreprocessor('MSFT_historical.csv','NVDA.csv','NVDA').merge_data()