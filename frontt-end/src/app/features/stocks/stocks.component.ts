import { Component, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { Observable, of, debounceTime, from, concatMap, reduce } from 'rxjs';
import { StockService } from './services/stock.service';
import { MarketDataService } from './services/market-data.service';
import {IndexesSummaryService, StockIndex} from './services/indexes-summary.service';
export interface Stock {
  symbol: string;
  companyName: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  sector: string;
  pe: number;
  dividend: number;
  logoUrl?: string;
}

@Component({
  selector: 'app-stocks',
  templateUrl: './stocks.component.html',
  styleUrl: './stocks.component.css',
  standalone: false
})
export class StocksComponent implements OnInit {
  stocks: Stock[] = [];
  filteredStocks: Stock[] = [];
  sectors: string[] = [];
  darkMode = false;
  // Loading and error states
  isLoading = true;
  error: string | null = null;
  loading = true;
  stockIndices: StockIndex[] = [];
  // Array of stock index symbols
  private readonly indexSymbols = [
    '^GSPC', // S&P 500
  ];
  // Pagination
  currentPage = 1;
  pageSize = 20;
  totalItems = 0;
  
  // Sorting and filtering
  sortField = 'marketCap';
  sortDirection = 'desc';
  searchControl = new FormControl('');
  selectedSector = 'All';
  
  // Watchlist functionality
  watchlist: Set<string> = new Set();
  
  constructor(private stockService: StockService,private marketService: MarketDataService,private stockMarketService: IndexesSummaryService) {}
  
  ngOnInit(): void {
    this.loadStocks();
    this.searchControl.valueChanges
      .pipe(debounceTime(300))
      .subscribe(value => this.filterStocks());
  }

  loadStocks(): void {
    this.isLoading = true;
    // List of S&P 500 symbols (you'll need to maintain this list)
    const sp500Symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'V',
    'WMT', 'XOM', 'UNH', 'MA', 'PG', 'JNJ', 'HD', 'COST', 'DIS', 'BAC',
    'KO', 'PEP', 'CSCO', 'CVX', 'ABBV', 'MRK', 'PFE', 'TMO', 'ACN', 'WFC',
    'MCD', 'INTC', 'VZ', 'CMCSA', 'ADBE', 'NKE', 'BMY', 'UPS', 'SCHW', 'T',
    'AMD', 'PM', 'UNP', 'HON', 'COP', 'IBM', 'BA', 'AMGN', 'GE', 'SBUX',
    'MMM', 'GS', 'CAT', 'LOW', 'SPGI', 'BKNG', 'BLK', 'AXP', 'DE', 'MDT',
    'GILD', 'CVS', 'CI', 'ELV', 'LMT', 'SYK', 'TJX', 'PNC', 'FCX', 'MO',
    'F', 'GM', 'DHR', 'AIG', 'C', 'HCA', 'USB', 'TGT', 'FDX', 'DOW',
    'NEE', 'SO', 'DUK', 'CSX', 'ITW', 'SHW', 'CL', 'ZTS', 'EQIX', 'ICE',
    'WM', 'CME', 'BDX', 'EOG', 'SLB', 'APD', 'MAR', 'AON', 'PSX', 'ORLY'];
    
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
        this.stocks = data;

        this.extractSectors();
        this.filterStocks();
        this.isLoading = false;
      },
      error: (err) => {
        this.error = 'Failed to load stocks data: ' + err.message;
        this.isLoading = false;
      }
    });
     this.stockMarketService.getStockIndices(this.indexSymbols).subscribe({
      next: (data) => {
        this.stockIndices = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load stock market data';
        this.loading = false;
        console.error('Error loading stock indices:', err);
      }
    });
  }
  extractSectors(): void {
    const sectorSet = new Set(this.stocks.map(stock => stock.sector));
    this.sectors = Array.from(sectorSet);
  }
  
  filterStocks(): void {
    let result = [...this.stocks];
    
    // Apply search filter
    const searchTerm = this.searchControl.value?.toLowerCase();
    if (searchTerm) {
      result = result.filter(stock => 
        stock.symbol.toLowerCase().includes(searchTerm) || 
        stock.companyName.toLowerCase().includes(searchTerm)
      );
    }
    
    // Apply sector filter
    if (this.selectedSector !== 'All') {
      result = result.filter(stock => stock.sector === this.selectedSector);
    }
    
    // Apply sorting
    result.sort((a, b) => {
      const aValue = a[this.sortField as keyof Stock];
      const bValue = b[this.sortField as keyof Stock];
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return this.sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
      } else {
        const aStr = String(aValue);
        const bStr = String(bValue);
        return this.sortDirection === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
      }
    });
    
    // Update pagination
    this.totalItems = result.length;
    this.filteredStocks = this.paginateStocks(result);
  }
  
  paginateStocks(stocks: Stock[]): Stock[] {
    const startIndex = (this.currentPage - 1) * this.pageSize;
    return stocks.slice(startIndex, startIndex + this.pageSize);
  }
  
  onPageChange(page: number): void {
    this.currentPage = page;
    this.filterStocks();
  }
  
  onSortChange(field: string): void {
    if (this.sortField === field) {
      this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortField = field;
      this.sortDirection = 'desc';
    }
    this.filterStocks();
  }
  
  onSectorChange(sector: string): void {
    this.selectedSector = sector;
    this.currentPage = 1;
    this.filterStocks();
  }
  
  toggleWatchlist(symbol: string): void {
    if (this.watchlist.has(symbol)) {
      this.watchlist.delete(symbol);
    } else {
      this.watchlist.add(symbol);
    }
  }
  
  isInWatchlist(symbol: string): boolean {
    return this.watchlist.has(symbol);
  }
  
  getChangeClass(change: number): string {
    return change > 0 ? 'positive-change' : (change < 0 ? 'negative-change' : 'neutral-change');
  }
  
}