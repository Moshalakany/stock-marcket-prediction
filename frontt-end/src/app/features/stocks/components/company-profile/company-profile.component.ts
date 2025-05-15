import { Component, OnInit, Input, OnChanges, SimpleChanges } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { finalize, catchError } from 'rxjs/operators';
import { of } from 'rxjs';
import { StockService } from '../../services/stock.service';
import { CompanyService } from '../../../../shared/services/company.service';
import { NewsService } from '../../../../features/news/services/news.service';
import {ChartForStockComponent} from '../chart-for-stock/chart-for-stock.component';
import { PredictionChartComponent} from '../prediction-chart/prediction-chart.component';
import { StocksModule } from '../../stocks.module';
import { NewsModule } from '../../../news/news.module';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-company-profile',
  templateUrl: './company-profile.component.html',
  styleUrls: ['./company-profile.component.css'],
  standalone: true,
  providers: [StockService, CompanyService, NewsService],
  imports: [ StocksModule,NewsModule,MatIconModule,CommonModule,FormsModule]
})
export class CompanyProfileComponent implements OnInit, OnChanges {
  @Input() companySymbol: string = '';
  
  company: any = null;
  financialData: any = null;
  keyMetrics: any = null;
  isLoading = true;
  error: string | null = null;
  darkMode = false; // You can connect this to your theme service
  ticker: string | null = '';
  // Time periods for chart
  timePeriods = ['1D', '1W', '1M', '3M', '6M', '1Y', '5Y'];
  selectedPeriod = '1M';

  constructor(
    private stockService: StockService,
    private companyService: CompanyService,
    private newsService: NewsService,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    // Check if we have a route parameter
    this.route.params.subscribe(params => {
      if (params['ticker']) {  // <-- Change 'symbol' to 'ticker'
        this.companySymbol = params['ticker'];
        this.loadCompanyData();
      }
    });
  
    this.route.paramMap.subscribe(params => {
      this.ticker = params.get('ticker'); // This should now work correctly
      console.log('Ticker received:', this.ticker);
    });
  
    if (!this.companySymbol) {
      this.error = 'No company symbol provided';
      this.isLoading = false;
    }
  }
  
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['companySymbol'] && !changes['companySymbol'].firstChange) {
      this.loadCompanyData();
    }
  }
  
  loadCompanyData(): void {
    this.isLoading = true;
    this.error = null;
    
    this.stockService.getCompanyProfile(this.companySymbol)
      .pipe(
        finalize(() => this.isLoading = false),
        catchError(err => {
          this.error = `Error loading company data: ${err.message}`;
          return of(null);
        })
      )
      .subscribe(data => {
        if (data) {
          this.company = data;
          
          // Set the selected company in the service
          // This will trigger updates in other components like news
          this.companyService.setSelectedCompany(this.companySymbol);
          
          // Load financial data
          this.loadFinancialData();
        }
      });
  }
  
  loadFinancialData(): void {
    this.stockService.getFinancialMetrics(this.companySymbol)
      .pipe(
        catchError(err => {
          console.error('Error loading financial data', err);
          return of(null);
        })
      )
      .subscribe(data => {
        this.financialData = data;
        
        // Calculate key metrics
        if (this.financialData && this.company) {
          this.calculateKeyMetrics();
        }
      });
  }
  
  calculateKeyMetrics(): void {
    // Calculate or extract important financial metrics
    this.keyMetrics = {
      marketCap: this.company.marketCap || 'N/A',
      pe: this.company.pe || 'N/A',
      eps: this.financialData?.eps || 'N/A',
      revenue: this.financialData?.revenue || 'N/A',
      revenueGrowth: this.financialData?.revenueGrowth || 'N/A',
      profitMargin: this.financialData?.profitMargin || 'N/A',
      dividendYield: this.company.dividend || 'N/A',
      beta: this.financialData?.beta || 'N/A',
      yearHigh: this.financialData?.yearHigh || 'N/A',
      yearLow: this.financialData?.yearLow || 'N/A'
    };
  }
  
  changePeriod(period: string): void {
    this.selectedPeriod = period;
    // Additional logic to update charts based on selected period
  }
  
  getChangeClass(change: number): string {
    return change > 0 ? 'text-green-500' : 
           change < 0 ? 'text-red-500' : 
           'text-gray-500';
  }
  
  getChangeIcon(change: number): string {
    return change > 0 ? 'trending_up' : 
           change < 0 ? 'trending_down' : 
           'remove';
  }
  
  formatLargeNumber(num: number): string {
    if (!num) return 'N/A';
    
    if (num >= 1e12) {
      return (num / 1e12).toFixed(2) + 'T';
    } else if (num >= 1e9) {
      return (num / 1e9).toFixed(2) + 'B';
    } else if (num >= 1e6) {
      return (num / 1e6).toFixed(2) + 'M';
    } else if (num >= 1e3) {
      return (num / 1e3).toFixed(2) + 'K';
    }
    
    return num.toString();
  }
}