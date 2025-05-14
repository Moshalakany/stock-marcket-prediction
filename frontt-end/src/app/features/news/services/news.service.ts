import { Injectable } from '@angular/core';
import { ApiService } from '../../../core/services/api.service';
import { NewsArticle, TickerInfo } from '../../../core/models/news.model';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

interface ApiNewsResponse {
  datetime: string;
  link: string;
  title: string;
  source: string;
  // Add any additional fields from your API response here
}

@Injectable({
  providedIn: 'root'
})
export class NewsService {
  private endpoint = 'news/latest';

  constructor(private apiService: ApiService) { }

  public getNewsListForTicker(ticker: string): Observable<NewsArticle[]> {
    return this.apiService.get<ApiNewsResponse[]>(`${this.endpoint}/${ticker}`).pipe(
      map(apiResponse => this.transformNewsResponse(apiResponse, ticker))
    );
  }

  private transformNewsResponse(apiArticles: ApiNewsResponse[],ticker: string): NewsArticle[] {
    return apiArticles.map(apiArticle => ({
      id: this.generateArticleId(apiArticle.link),
      title: apiArticle.title,
      source: apiArticle.source,
      publishedTime: new Date(apiArticle.datetime).toISOString(),
      url: apiArticle.link,
      relatedTickers: [{ 
        symbol: ticker,
        companyName: '',
        changePercent: 0,
        isFavorite: true
      }]
    }));
  }

  private generateArticleId(link: string): string {
    // Create ID from URL hash
    return btoa(link).slice(-12);
  }

  private extractTickers(title: string): TickerInfo[] {
    // Optional: Implement ticker extraction logic from title if needed
    const tickerRegex = /\b[A-Z]{2,4}\b/g;
    const matches = title.match(tickerRegex) || [];
    return matches.map(ticker => ({ 
      symbol: ticker, 
      companyName: '',
      changePercent: 0,
      isFavorite: false
    }));
  }
  
  public getSentimentAnalysis(headline: string): Observable<{
    sentiment: string,
    confidence: number,
    scores: { negative: number, neutral: number, positive: number }
  }> {
    return this.apiService.post<{
      sentiment: string,
      confidence: number,
      scores: { negative: number, neutral: number, positive: number }
    }>('sentiment/analyze/', { text: headline });
  }
}