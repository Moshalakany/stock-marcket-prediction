from flask import Flask, jsonify, request
from flask_cors import CORS
from scraper import get_latest_news, get_news_by_date_range, get_latest_page_news
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure news_data directory exists
if not os.path.exists('news_data'):
    os.makedirs('news_data')

@app.route('/api/news/latest/<symbol>', methods=['GET'])
def latest_news(symbol):
    """Get the latest news for a stock symbol"""
    try:
        news = get_latest_news(symbol.upper())
        return jsonify(news)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/news/range/<symbol>', methods=['GET'])
def news_by_date_range(symbol):
    """Get news for a stock symbol within a date range"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not start_date or not end_date:
            return jsonify({"error": "Please provide start_date and end_date parameters"}), 400
        
        news = get_news_by_date_range(symbol.upper(), start_date, end_date)
        return jsonify(news)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/news/latest-page/<symbol>', methods=['GET'])
def latest_page_news(symbol):
    """Get only the latest page of news for a stock symbol"""
    try:
        news = get_latest_page_news(symbol.upper())
        return jsonify(news)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
