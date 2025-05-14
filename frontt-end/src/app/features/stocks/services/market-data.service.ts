import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError, of, forkJoin } from 'rxjs';
import {  map, tap, switchMap,catchError, delay } from 'rxjs/operators';
import { Company } from '../../../core/models/company.model';
import { ApiService } from '../../../core/services/api.service';
import { Stock } from '../../stocks/stocks.component';
@Injectable({
  providedIn: 'root'
})
export class MarketDataService {
  private coinCodexApiUrl = 'https://coincodex.com/stonks_api/get_quote/';
  private fmpApiKey = 'CI2l342r7A2BGZWYx6u6qN4qrzqXKbyW'; // Replace with your free FMP API key
  private fmpBaseUrl = 'https://financialmodelingprep.com/api/v3/sp500_constituents';
  constructor(private http: HttpClient) { }



  getSP500Symbols(): Observable<string[]> {
    return this.http.get<any>(`${this.fmpBaseUrl}?apikey=${this.fmpApiKey}`).pipe(
      map(response => response.map((item: any) => item.symbol)),
      catchError(error => {
        console.error('Error fetching S&P 500 symbols:', error);
        return of([]); // Return empty array on error
      })
    );
  }

  // Get data for a single stock
  getStockQuote(symbol: string): Observable<any> {
    return this.http.get(`${this.coinCodexApiUrl}${symbol}`);
  }

  // Get data for multiple stocks with rate limiting
  getMultipleStockQuotes(symbols: string[]): Observable<Stock[]> {
    const requests = symbols.map(symbol => 
      this.getStockQuote(symbol).pipe(
        catchError(error => {
          console.error(`Error fetching ${symbol}:`, error);
          return of(null); // Return null for failed requests
        })
      )
    );

    return forkJoin(requests).pipe(
      map(responses => responses
        .filter(response => response !== null)
        .map(response => this.mapApiResponseToStock(response))
      ),
      // Add delay between batches to avoid rate limiting
      delay(1000)
    );
  }

  private mapApiResponseToStock(apiResponse: any): Stock {
    return {
      symbol: apiResponse.symbol,
      companyName: apiResponse.companyName,
      price: apiResponse.latestPrice,
      change: apiResponse._change,
      changePercent: apiResponse.changePercent,
      volume: apiResponse.latestVolume,
      marketCap: apiResponse.marketCap,
      sector: apiResponse.sector || 'Unknown', // Note: API doesn't provide sector
      pe: apiResponse.peRatio,
      dividend: apiResponse.dividendYield
    };
  }
}