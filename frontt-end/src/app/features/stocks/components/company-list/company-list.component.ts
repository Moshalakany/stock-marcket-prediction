import { Component, OnInit, OnDestroy, Input } from '@angular/core';
import { WebsocketService } from '../../../../core/services/websocket.service';
import { StockService } from '../../services/stock.service';
import { Company } from '../../../../core/models/company.model';
import { Subscription } from 'rxjs';
import { MarketDataService } from '../../services/market-data.service';
import { FormControl } from '@angular/forms';
import { Observable, of, debounceTime, from, concatMap, reduce } from 'rxjs';
import { Stock } from '../../../stocks/stocks.component';
// Static S&P 500 list (simplified; in reality, fetch from a JSON file or API)
const SP500_STOCKS: Partial<Company>[] = [
  { ticker: 'AAPL', name: 'Apple Inc.', price: 0, change: 0 },
  { ticker: 'MSFT', name: 'Microsoft Corp.', price: 0, change: 0 },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', price: 0, change: 0 },
  // Add all 500 S&P 500 stocks here (e.g., from a JSON file)
];

@Component({
  selector: 'app-company-list',
  templateUrl: './company-list.component.html',
  styleUrls: ['./company-list.component.css'],
  standalone: false
})
export class CompanyListComponent implements OnInit, OnDestroy {
  @Input() limit: number | null = 5;
  //companies: Company[] = [];
  darkMode = false;
  stocks: Stock[] = [];
    filteredStocks: Stock[] = [];
    sectors: string[] = [];
   
    // Loading and error states
    isLoading = true;
    error: string | null = null;
  constructor(private stocksService: StockService, private marketService:MarketDataService) { }

  ngOnInit(): void {
    // Check theme preference
    const savedTheme = localStorage.getItem('theme');
    this.darkMode = savedTheme === 'dark' || 
      (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    
    // Subscribe to theme changes
    window.addEventListener('storage', this.handleStorageChange.bind(this));

    // Load stock data
    this.loadStocks();
  }

  ngOnDestroy(): void {
    window.removeEventListener('storage', this.handleStorageChange.bind(this));
  }

  handleStorageChange(event: StorageEvent): void {
    if (event.key === 'theme') {
      this.darkMode = event.newValue === 'dark';
    }
  }

 loadStocks(): void {
     this.isLoading = true;
     // List of S&P 500 symbols (you'll need to maintain this list)
     const sp500Symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
     'CSCO','NFLX'
     ];
     
     // Process in batches to avoid rate limiting
     const batchSize = 50;
     const batches = [];
     for (let i = 0; i < sp500Symbols.length; i += batchSize) {
       batches.push(sp500Symbols.slice(i, i + batchSize));
     }
 
     from(batches).pipe(
       concatMap(batch => this.marketService.getMultipleStockQuotes(batch)),
       reduce<Stock[], Stock[]>((acc, batchResult) => [...acc, ...batchResult], [] as Stock[])
     ).subscribe({
       next: (data) => {
        // `https://coincodex.com/stonks_api/get_logo/AAPL/`
         this.stocks = data;
         for (let i = 0; i < this.stocks.length; i++) {
          // let cleanName = this.stocks[i].symbol
          //   .toLowerCase()
          //   .replace(/\s+/g, '') // Removes spaces
          //   .replace(/[^a-z0-9]/g, ''); // Removes special characters
  
          this.stocks[i].logoUrl = `https://coincodex.com/stonks_api/get_logo/${this.stocks[i].symbol}/`;
        }
  
         this.isLoading = false;
       },
       error: (err) => {
         this.error = 'Failed to load stocks data: ' + err.message;
         this.isLoading = false;
       }
     });
     
   }

   
}