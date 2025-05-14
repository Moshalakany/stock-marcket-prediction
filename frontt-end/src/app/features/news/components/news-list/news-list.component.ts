import { Component, Input } from '@angular/core';
import { NewsArticle } from '../../../../core/models/news.model';
import {  TickerInfo} from '../../../../core/models/news.model';
import { CompanyService } from '../../../../shared/services/company.service';
import { NewsService } from '../../services/news.service';
import { Observable } from 'rxjs';
import { forkJoin, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
@Component({
  selector: 'app-news-list',
  templateUrl: './news-list.component.html',
  styleUrls: ['./news-list.component.css'],
  standalone: false
})
export class NewsListComponent {
  @Input() limit: number = 3;
  selectedCompany: string = '';
  newsArticles: NewsArticle[] = [
    
  ];

  constructor(private companyService: CompanyService,private newsService: NewsService) {}


  ngOnInit() {
    this.companyService.selectedCompany$.subscribe(company => {
      this.selectedCompany = company;
      this.loadNews();
    });
  }

  loadNews() {
    console.log(`Fetching news for: ${this.selectedCompany}`);
    this.newsService.getNewsListForTicker(this.selectedCompany).subscribe(articles => {
      const analysisRequests = articles.map(article => 
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
      
      forkJoin(analysisRequests).subscribe(updatedArticles => {
        this.newsArticles = updatedArticles;
      });
    });
  }
}