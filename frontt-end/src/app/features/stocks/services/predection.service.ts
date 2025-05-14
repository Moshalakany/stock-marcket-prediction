import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { map, catchError } from 'rxjs/operators';

interface StockDataPoint {
  x: number; // Timestamp
  y: number; //close price
}

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  private apiUrl = 'http://localhost:3000/predict';

  constructor(private http: HttpClient) { }

  getStockData(symbol: string, timeRange: '1D' | '1W' | '1M' | '1Y'): Observable<StockDataPoint[]> {
    const url = `${this.apiUrl}`;
    return this.http.get<any>(url).pipe(
      map(response => this.transformStockData(response)),
      catchError(this.handleError)
    );
  }

  private transformStockData(response: any): StockDataPoint[] {
    if (!response || !response.dates || !response.predicted_prices ||
        !Array.isArray(response.dates) || !Array.isArray(response.predicted_prices) ||
        response.dates.length !== response.predicted_prices.length) {
      console.error("Invalid API Response:", response);
      return [];
    }

    const stockData: StockDataPoint[] = [];
    for (let i = 0; i < response.dates.length; i++) {
      stockData.push({
        x: new Date(response.dates[i]).getTime(),
        y: response.predicted_prices[i]
      });
    }

    return stockData;
  }

  private handleError(error: HttpErrorResponse) {
    console.error('API Error:', error);
    let errorMessage = 'An unknown error occurred.';
    if (error.error instanceof ErrorEvent) {
      errorMessage = `Error: ${error.error.message}`;
    } else {
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }
    return throwError(() => errorMessage);
  }
}
