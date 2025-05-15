from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_page(url, retry=0):
    """Get and parse articles from a page"""
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'lxml')
    articles = soup.find_all('div', class_='latest-news__story')
    
    if len(articles) == 0 and retry < 2:
        retry += 1
        print(f'No articles found, retrying {retry} time')
        return get_page(url, retry)
    
    return articles

def parse_articles(articles):
    """Extract information from article elements and parse datetime"""
    news_items = []
    for article in articles:
        datetime_str = article.find('time', class_='latest-news__date').get('datetime')
        title = article.find('a', class_='news-link').text
        source = article.find('span', class_='latest-news__source').text
        link = article.find('a', class_='news-link').get('href')
        
        # Parse datetime string to datetime object
        parsed_datetime = None
        try:
            # Try parsing in format "3/13/2025 6:23:33 AM"
            parsed_datetime = datetime.strptime(datetime_str, '%m/%d/%Y %I:%M:%S %p')
        except ValueError:
            try:
                # Fallback to ISO format if the above fails
                parsed_datetime = datetime.fromisoformat(datetime_str)
            except ValueError:
                print(f"Could not parse date: {datetime_str}")
                # Keep the original string if parsing fails
                parsed_datetime = datetime_str
        
        news_items.append({
            'datetime': parsed_datetime,
            'datetime_str': datetime_str,  # Keep original string for reference
            'title': title,
            'source': source,
            'link': link
        })
    
    return news_items

def get_latest_news(symbol):
    """Get the latest news for a stock symbol"""
    # Get the latest page and return top 10 news items
    news_items = get_latest_page_news(symbol)
    
    # Sort by datetime descending and return top 10
    return sorted(news_items, key=lambda x: x['datetime'], reverse=True)[:10]

def get_news_by_date_range(symbol, start_date, end_date):
    """Get news for a stock symbol within a date range
    
    Args:
        symbol: Stock symbol
        start_date: Start date string in format 'm/d/y'
        end_date: End date string in format 'm/d/y'
    """
    
    # Convert string dates to datetime objects using correct format
    start_dt = datetime.strptime(start_date, '%m/%d/%Y')
    end_dt = datetime.strptime(end_date, '%m/%d/%Y')
    end_dt= end_dt + timedelta(days=1)
    
    url = f'https://markets.businessinsider.com/news/{symbol.lower()}-stock?miRedirects=1&p=1'
    filtered_news = []
    
    # Get first page to find max pages
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'lxml')
    pagination_items = soup.find_all(class_='pagination__item')
    max_page = 1
    
    for item in pagination_items:
        page_number = item.get('data-pagination-page')
        if page_number:
            max_page = max(max_page, int(page_number))
    
    print(f'{max_page} pages of news for {symbol} found')
    
    # Gather news from all pages
    should_break = False
    for page in range(1, max_page+1):
        url = f'https://markets.businessinsider.com/news/{symbol.lower()}-stock?miRedirects=1&p={page}'
        articles = get_page(url)
        if not articles:
            continue
            
        page_news = parse_articles(articles)
        
        # Check each article's datetime and break if outside date range
        for item in page_news:
            article_dt = item['datetime']
            
            # If datetime parsing failed and we have a string, try to parse it again
            if isinstance(article_dt, str):
                try:
                    # Parse datetime in format "3/13/2025 6:23:33 AM"
                    article_dt = datetime.strptime(article_dt, '%m/%d/%Y %I:%M:%S %p')
                except ValueError:
                    # Fallback to ISO format if the above fails
                    try:
                        article_dt = datetime.fromisoformat(article_dt)
                    except ValueError:
                        print(f"Could not parse date: {article_dt}")
                        continue
            
            if article_dt <= end_dt and article_dt >= start_dt:
                filtered_news.append(item)
            elif article_dt < start_dt:
                # We've gone past our date range, no need to check more pages
                should_break = True
                
        if should_break:
            print(f'Found news older than {start_date}, stopping at page {page}')
            break
        
        print(f'page {page} done')
    
    return filtered_news

def get_latest_page_news(symbol):
    """Get only the latest page of news for a stock symbol"""
    url = f'https://markets.businessinsider.com/news/{symbol.lower()}-stock?miRedirects=1&p=1'
    
    try:
        articles = get_page(url)
        if not articles:
            return []
            
        news_items = parse_articles(articles)
        return news_items
    except Exception as e:
        print(f"Error scraping news for {symbol}: {e}")
        return []
