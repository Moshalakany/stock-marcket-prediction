import pandas as pd 
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
def tokenize(headline):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model=AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    calssifier=pipeline('text-classification',model=model,tokenizer=tokenizer)
    res=calssifier(headline)
    return res[0]
file_path = r'sp500_companies.csv'
data1 = pd.read_csv(file_path)
print(data1.head())
symbols_array = data1['Symbol'].tolist()

def get_sentiment(text):
    res=tokenize(text)    
    sentiment_class = res['label']
    sentiment_score = res['score']
    return sentiment_class, sentiment_score
completedSympols=["AAPL","AVGO","BRK-B","GOOG","GOOGL","LLY","MSFT","NVDA"]
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_stock(symbol):
    print(f"Processing {symbol}")
    filename = f'news_data/{symbol}.csv'
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        return

    rows = []
    for _, row in df.iterrows():
        headline = row['title']
        sentiment_class, sentiment_score = get_sentiment(headline)

        rows.append({
            'datetime': row['datetime'],
            'title': row['title'],
            'source': row['source'],
            'link': row['link'],
            'sentiment_class': sentiment_class,
            'sentiment_score': sentiment_score
        })

    new_df = pd.DataFrame(rows)
    output_filename = f'sentiment_data/{symbol}.csv'
    new_df.to_csv(output_filename, index=False)
    print(f"Finished {symbol}")

def generate_sentiment():
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, symbol) for symbol in symbols_array[:180] if symbol not in completedSympols]
        
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a stock: {e}")

    print("All stocks processed.")
generate_sentiment()