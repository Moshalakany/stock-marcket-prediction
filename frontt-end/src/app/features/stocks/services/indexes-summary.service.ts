import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, forkJoin } from 'rxjs';
import { map } from 'rxjs/operators';

export interface StockIndex {
  symbol: string;
  name: string;
  price: number;
  changePercentage: number;
  change: number;
  volume: number;
  dayLow: number;
  dayHigh: number;
  yearHigh: number;
  yearLow: number;
  marketCap: number;
  priceAvg50: number;
  priceAvg200: number;
  exchange: string;
  open: number;
  previousClose: number;
  timestamp: number;
}

@Injectable({
  providedIn: 'root'
})
export class IndexesSummaryService {
 private readonly baseUrl = 'https://financialmodelingprep.com/stable/quote';
  private readonly apiKey = '91pbMXkifHeU5nXzW7XLrK3a59jujVL7';

  constructor(private http: HttpClient) {}

  getStockIndices(symbols: string[]): Observable<StockIndex[]> {
    const requests = symbols.map(symbol => 
      this.http.get<StockIndex[]>(`${this.baseUrl}?symbol=${symbol}&apikey=${this.apiKey}`)
    );
    
    return forkJoin(requests).pipe(
      map((responses: StockIndex[][]) => responses.flat())
    );
  }
}
