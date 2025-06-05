import { Component, OnInit } from '@angular/core';
import {IndexesSummaryService, StockIndex} from '../../services/indexes-summary.service';
@Component({
  selector: 'indexes-summary',
  standalone: false,
  templateUrl: './indexes-summary.component.html',
  styleUrl: './indexes-summary.component.css'
})
export class IndexesSummaryComponent implements OnInit {
stockIndices: StockIndex[] = [];
  loading = true;
  error: string | null = null;

  // Array of stock index symbols
  private readonly indexSymbols = [
    '^GSPC', // S&P 500
    '^IXIC', // NASDAQ
    '^DJI',  // Dow Jones
    '^RUT' ,  // Russell 2000
    
  ];

  constructor(private stockMarketService: IndexesSummaryService) {}

  ngOnInit(): void {
    this.loadStockIndices();
  }

  loadStockIndices(): void {
    this.loading = true;
    this.error = null;

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

  // Helper method to determine color class based on change
  getChangeColorClass(change: number): string {
    return change >= 0 ? 'text-green-600' : 'text-red-600';
  }

  // Helper method to determine border color class based on change
  getBorderColorClass(change: number): string {
    return change >= 0 ? 'border-green-500' : 'border-red-500';
  }

  // Helper method to format percentage with sign
  formatPercentage(percentage: number): string {
    const sign = percentage >= 0 ? '+' : '';
    return `${sign}${percentage.toFixed(2)}%`;
  }

  // Helper method to format change with sign
  formatChange(change: number): string {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}`;
  }
  // Helper method to get display name for symbols
  getDisplayName(symbol: string): string {
    const nameMap: { [key: string]: string } = {
      '^GSPC': 'S&P 500',
      '^IXIC': 'NASDAQ',
      '^DJI': 'DOW JONES',
      '^RUT': 'RUSSELL 2000',
      '^GDAXI': 'DAX',
      '^FTSE': 'FTSE 100',
    };
    return nameMap[symbol] || symbol;
  }
}
