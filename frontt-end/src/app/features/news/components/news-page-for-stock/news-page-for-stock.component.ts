import { Component, OnInit, Input, SimpleChanges } from '@angular/core';
import { NewsArticle } from '../../../../core/models/news.model';
import { NewsService } from '../../services/news.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs/operators';
import { Observable } from 'rxjs';
import { forkJoin, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
@Component({
  selector: 'app-news-page-for-stock',
  standalone: false,
  templateUrl: './news-page-for-stock.component.html',
  styleUrl: './news-page-for-stock.component.css'
})
export class NewsPageForStockComponent implements OnInit {
  @Input() symbol: string = ''; // Input property to receive the company symbol

  newsArticles: NewsArticle[] = [];
  isLoading = true;
  errorMessage: string | null = null;

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
  darkMode = false;
  constructor(private newsService: NewsService) { }

  ngOnInit(): void {
    if (this.symbol) {
      this.loadNews();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['symbol'] && changes['symbol'].currentValue) {
      this.loadNews();
    }
  }

  private loadNewsData(): void {
    this.isLoading = true;
    this.errorMessage = null;
    this.newsArticles = []; // Reset articles array

    this.newsService.getNewsListForTicker(this.symbol)
      .pipe(
        finalize(() => this.isLoading = false)
      )
      .subscribe({
        next: (articles: NewsArticle[]) => {
          this.newsArticles = articles;
        },
        error: (err) => {
          this.errorMessage = 'Failed to load news articles. Please try again later.';
          console.error('News load error:', err);
        }
      });
  }


    loadNews() {
      this.isLoading = true;
      this.errorMessage = null;
      this.newsArticles = []; // Reset articles array
       console.log(`Fetching news for: ${this.symbol}`);
       this.newsService.getNewsListForTicker(this.symbol).subscribe(articles => {
         const analysisRequests = articles.map(article => 
           this.newsService.getSentimentAnalysis(article.title).pipe(
            
             map(response => {
               const result = {
                 ...article,
                 sentiment: response.sentiment,
                 confidence: response.confidence,
                 scores: response.scores
               };
               this.isLoading = false;
               return result;
             }),
             catchError(error => {
               console.error('Error fetching sentiment analysis:', error);
               return of(article); // Return original article if analysis fails
             })
           )
         );
         
         forkJoin(analysisRequests).subscribe(updatedArticles => {
           this.newsArticles = updatedArticles;
         });
       });
     }








  setCategory(categoryId: string): void {
    this.activeCategory = categoryId;
    // Add category-based filtering logic here if needed
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
    this.loadNews();
  }
}