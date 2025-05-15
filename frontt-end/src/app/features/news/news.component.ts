import { Component } from '@angular/core';
import { CompanyService } from '../../shared/services/company.service';
import { NewsService } from './services/news.service';
@Component({
  selector: 'app-news',
  standalone: false,
  templateUrl: './news.component.html',
  styleUrl: './news.component.css'
})
export class NewsComponent {
  currentTime: Date = new Date();
  watchlistEmpty: boolean = true;
  selectedCompany: string = '';
  constructor(private companyService: CompanyService,private newsService: NewsService) {}

  ngOnInit(): void {
    // Update time every minute
    setInterval(() => {
      this.currentTime = new Date();
    }, 60000);
    this.companyService.selectedCompany$.subscribe(company => {
      this.selectedCompany = company;
      this.loadNews();
    });
  }
  loadNews() {
    console.log(`Fetching news for: ${this.selectedCompany}`);
    // Call your API service to fetch news for the selected company
  }
}
