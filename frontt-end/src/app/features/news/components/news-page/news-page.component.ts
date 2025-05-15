// news-page.component.ts
import { Component, OnInit } from '@angular/core';
import { NewsArticle, TickerInfo } from '../../../../core/models/news.model';
import { NewsService } from '../../services/news.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { catchError, finalize, map, switchMap } from 'rxjs/operators';
import { forkJoin, Observable, of } from 'rxjs';
// Sample S&P 500 tickers - in practice, you'd want to get this from a service or file
const SP500_TICKERS = [
  'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META','NVDA','CSCO','NFLX','ADBE','CMCSA','ABT','TMO','NKE','AVGO','INTC',
  'TSLA'
];

@Component({
  selector: 'app-news-page',
  templateUrl: './news-page.component.html',
  styleUrls: ['./news-page.component.scss'],
  standalone: false
})
export class NewsPageComponent implements OnInit {
  newsArticles: NewsArticle[] = [];
  isLoading = true;
  errorMessage: string | null = null;
  darkMode = false;
  categories = [
    { id: 'all', name: 'All News' },
    { id: 'business', name: 'Business' },
    { id: 'markets', name: 'Markets' },
    { id: 'technology', name: 'Technology' },
    { id: 'economy', name: 'Economy' },
    { id: 'personal-finance', name: 'Personal Finance' }
  ];

  activeCategory = 'all';
  searchQuery = '';
  tickers: string[] = SP500_TICKERS; 

  constructor(private newsService: NewsService) { }

  ngOnInit(): void {
    this.loadNewsData();
  }

  private loadNewsData(): void {
    this.isLoading = true;
    this.errorMessage = null;
    this.newsArticles = [];
  
    // First get news for all tickers
    const newsRequests = this.tickers.map(ticker => 
      this.newsService.getNewsListForTicker(ticker)
    );
  
    forkJoin(newsRequests).pipe(
      switchMap((results: NewsArticle[][]) => {
        // Flatten all articles from all tickers
        const allArticles = results.flat();
        
        // Create sentiment analysis requests for all articles
        const analysisRequests = allArticles.map(article => 
          this.newsService.getSentimentAnalysis(article.title).pipe(
            map(response => ({
              ...article,
              sentiment: response.sentiment,
              confidence: response.confidence,
              scores: response.scores
            })),
            catchError(error => {
              console.error('Error fetching sentiment analysis:', error);
              return of(article); // Return original article if analysis fails
            })
          )
        );
  
        return forkJoin(analysisRequests);
      }),
      finalize(() => this.isLoading = false)
    ).subscribe({
      next: (analyzedArticles) => {
        this.newsArticles = analyzedArticles;
        // Optional: Remove duplicates if needed
        // this.newsArticles = this.removeDuplicates(analyzedArticles);
      },
      error: (err) => {
        this.errorMessage = 'Failed to load news articles. Please try again later.';
        console.error('News load error:', err);
      }
    });
  }

  // Helper method to remove duplicate articles (optional)
  private removeDuplicates(articles: NewsArticle[]): NewsArticle[] {
    const uniqueArticles = new Map<string, NewsArticle>();
    articles.forEach(article => {
      // Assuming each article has a unique identifier like 'id' or 'url'
      uniqueArticles.set(article.id || article.url, article);
    });
    return Array.from(uniqueArticles.values());
  }

  setCategory(categoryId: string): void {
    this.activeCategory = categoryId;
    // Implement category-based filtering if needed
  }

  get filteredArticles(): NewsArticle[] {
    if (!this.searchQuery) {
      return this.newsArticles;
    }
    
    const query = this.searchQuery.toLowerCase();
    return this.newsArticles.filter(article => 
      article.title.toLowerCase().includes(query) || 
      article.source.toLowerCase().includes(query)
    );
  }

  refreshNews(): void {
    this.loadNewsData();
  }
}