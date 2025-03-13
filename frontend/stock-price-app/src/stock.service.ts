import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class StockService {

  constructor() { }

  getStockPrice(symbol: string): Observable<any> {
    // Mock current price
    const mockPriceData = {
      symbol: symbol,
      price: 240
    };
    return of(mockPriceData); // 'of()' emits a single value and completes
  }

  getStockHistory(symbol: string): Observable<any> {
    // Mock history data
    const mockHistoryData = {
      symbol: symbol,
      history: [
        { date: '2023-01-01', price: 32 },
        { date: '2023-01-02', price: 54 },
        { date: '2023-01-03', price: 38 },
        { date: '2023-01-04', price: 20 },
        { date: '2023-01-05', price: 30 }
      ]
    };
    return of(mockHistoryData);
  }
}
