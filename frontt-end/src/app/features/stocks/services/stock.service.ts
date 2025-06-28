

//q49xrWblJSHJBUfYbUqpM6BhmbX1osF_


import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError, of, forkJoin } from 'rxjs';
import {  map, tap, switchMap,catchError, delay } from 'rxjs/operators';
import { Company } from '../../../core/models/company.model';
import { ApiService } from '../../../core/services/api.service';


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
}

// Define interfaces for API responses
interface FinnhubConstituentsResponse {
  constituents: string[];
  symbol: string;
}

interface FinnhubQuoteResponse {
  c: number;  // Current price
  d: number;  // Change
  dp: number; // Percent change
  h: number;  // High price of the day
  l: number;  // Low price of the day
  o: number;  // Open price of the day
  pc: number; // Previous close price
  t: number;  // Timestamp
  v: number;  // Volume
}

interface FinnhubProfileResponse {
  name: string;
  marketCapitalization: number;
  finnhubIndustry: string;
  peRatio?: number;
  dividendYield?: number;
}

interface FinnhubCandleResponse {
  c: number[];  // Close prices
  h: number[];  // High prices
  l: number[];  // Low prices
  o: number[];  // Open prices
  s: string;    // Status
  t: number[];  // Timestamps
  v: number[];  // Volumes
}

interface FinnhubSearchResponse {
  count: number;
  result: Array<{
    description: string;
    displaySymbol: string;
    symbol: string;
    type: string;
  }>;
}



@Injectable({
  providedIn: 'root'
})
export class StockService {
  private apiKey = '07BWIDLHE458SOXH'; 
  private apiUrl = 'https://www.alphavantage.co/query';
  private useMockData = false; 

  private finnhubApiKey = 'cv9ut4hr01qpd9s9mc1gcv9ut4hr01qpd9s9mc20'; 
  private mockMode = false; // Set to false to use real API

  
  private finnhubApiUrl = 'https://finnhub.io/api/v1'; // Finnhub base URL

  private coinCodexApiUrl = 'https://coincodex.com/stonks_api/get_quote/'
  constructor(private http: HttpClient,private apiService: ApiService) {}


  private mockCompanies: Company[] = [
    { 
      ticker: 'AAPL', 
      name: 'Apple Inc.',
      price: 213.49,
      change: 3.81,
      percentChange: 1.82,
      logoUrl: 'https://logo.clearbit.com/apple.com',
      marketCap: '3.21T',
    },
    { 
      ticker: 'NVDA', 
      name: 'NVIDIA Corporation',
      price: 121.67,
      change: 6.09,
      percentChange: 5.27,
      logoUrl: 'https://logo.clearbit.com/nvidia.com',
      marketCap: '2.97T',
    },
    { 
      ticker: 'META', 
      name: 'Facebook, Inc.',
      price: 128.75,
      change: 101.25,
      percentChange: 1.83,
      logoUrl: 'https://logo.clearbit.com/Facebook.com',
      marketCap: '1.54T	',
      
    },
    { 
      ticker: 'MSFT', 
      name: 'Microsoft Corp',
      price: 417.15,
      change: 3.26,
      percentChange: 0.79,
      logoUrl: 'https://logo.clearbit.com/Microsoft.com',
      marketCap: '2.89T',
    },
    { 
      ticker: 'AMZN', 
      name: 'Amazon.com, Inc.',
      price: 187.23,
      change: 1.45,
      percentChange: 0.78,
      logoUrl: 'https://logo.clearbit.com/Amazon.com',
      marketCap: '2.10T',
    }
  ];


  getFinancialMetrics(symbol: string): Observable<any> {
    if (this.useMockData) {
      return of(this.getMockFinancialMetrics(symbol)).pipe(delay(700));
    }
    
    // Using Alpha Vantage INCOME_STATEMENT endpoint
    return this.http.get(`${this.apiUrl}`, {
      params: {
        function: 'INCOME_STATEMENT',
        symbol: symbol,
        apikey: this.apiKey
      }
    }).pipe(
      map(response => this.transformFinancialData(response)),
      catchError(error => {
        console.error('Error fetching financial metrics:', error);
        return of(this.getMockFinancialMetrics(symbol));
      })
    );
  }

  getHistoricalDataNew(symbol: string, period: string): Observable<any[]> {
    if (this.useMockData) {
      return of(this.getMockHistoricalData(symbol, period)).pipe(delay(600));
    }
    
    let interval = '60min';
    let outputSize = 'compact';
    
    // Adjust interval based on time period
    switch(period) {
      case '1D':
        interval = '5min';
        break;
      case '1W':
        interval = '60min';
        break;
      case '1M':
      case '3M':
        interval = 'daily';
        outputSize = 'full';
        break;
      case '6M':
      case '1Y':
      case '5Y':
        interval = 'weekly';
        outputSize = 'full';
        break;
    }
    
    // Function to use based on interval
    let timeSeriesFunction = 'TIME_SERIES_INTRADAY';
    if (interval === 'daily') timeSeriesFunction = 'TIME_SERIES_DAILY';
    if (interval === 'weekly') timeSeriesFunction = 'TIME_SERIES_WEEKLY';
    
    return this.http.get(`${this.apiUrl}`, {
      params: {
        function: timeSeriesFunction,
        symbol: symbol,
        interval: interval !== 'daily' && interval !== 'weekly' ? interval : '',
        outputsize: outputSize,
        apikey: this.apiKey
      }
    }).pipe(
      map(response => this.transformHistoricalData(response, interval)),
      catchError(error => {
        console.error('Error fetching historical data:', error);
        return of(this.getMockHistoricalData(symbol, period));
      })
    );
  }

  /**
   * Get mock company profile data
   */
  private getMockCompanyProfile(symbol: string): any {
    // Mock company profiles
    interface CompanyProfile {
      symbol: string;
      companyName: string;
      price: number;
      change: number;
      changePercent: number;
      marketCap: number;
      pe: number;
      industry: string;
      sector: string;
      exchange: string;
      description: string;
      employees: number;
      country: string;
      ceo: string;
      website: string;
      logoUrl: string;
      dividend: number;
      lastUpdated: Date;
    }
    
    const mockProfiles: Record<string, CompanyProfile> = {
      'AAPL': {
        symbol: 'AAPL',
        companyName: 'Apple Inc.',
        price: 182.37,
        change: 1.25,
        changePercent: 0.69,
        marketCap: 2850000000000,
        pe: 30.25,
        industry: 'Consumer Electronics',
        sector: 'Technology',
        exchange: 'NASDAQ',
        description: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories.',
        employees: 154000,
        country: 'United States',
        ceo: 'Mr. Timothy D. Cook',
        website: 'https://www.apple.com',
        logoUrl: 'https://logo.clearbit.com/apple.com',
        dividend: 0.0059,
        lastUpdated: new Date()
      },
      'MSFT': {
        symbol: 'MSFT',
        companyName: 'Microsoft Corporation',
        price: 417.15,
        change: 3.26,
        changePercent: 0.79,
        marketCap: 3100000000000,
        pe: 35.92,
        industry: 'Software—Infrastructure',
        sector: 'Technology',
        exchange: 'NASDAQ',
        description: 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide. The company operates in three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.',
        employees: 181000,
        country: 'United States',
        ceo: 'Satya Nadella',
        website: 'https://www.microsoft.com',
        logoUrl: 'https://logo.clearbit.com/microsoft.com',
        dividend: 0.0074,
        lastUpdated: new Date()
      },
      'TSLA': {
        symbol: 'TSLA',
        companyName: 'Tesla, Inc.',
        price: 175.21,
        change: -2.55,
        changePercent: -1.43,
        marketCap: 557000000000,
        pe: 50.7,
        industry: 'Auto Manufacturers',
        sector: 'Consumer Cyclical',
        exchange: 'NASDAQ',
        description: 'Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems. The company operates in two segments, Automotive, and Energy Generation and Storage.',
        employees: 127855,
        country: 'United States',
        ceo: 'Elon Musk',
        website: 'https://www.tesla.com',
        logoUrl: 'https://logo.clearbit.com/tesla.com',
        dividend: 0,
        lastUpdated: new Date()
      }
    };
    
    // Return the requested profile or generate a default one
    return mockProfiles[symbol] || {
      symbol: symbol,
      companyName: `${symbol} Corporation`,
      price: 100 + Math.random() * 300,
      change: Math.random() * 10 - 5,
      changePercent: Math.random() * 5 - 2.5,
      marketCap: Math.random() * 500000000000,
      pe: 10 + Math.random() * 40,
      industry: 'Various',
      sector: 'Technology',
      exchange: 'NASDAQ',
      description: `${symbol} Corporation is a fictional company created for demonstration purposes.`,
      employees: Math.floor(1000 + Math.random() * 200000),
      country: 'United States',
      ceo: 'John Doe',
      website: `https://www.${symbol.toLowerCase()}.com`,
      logoUrl: `https://coincodex.com/stonks_api/get_logo/${symbol.toUpperCase()}/`, 
      dividend: Math.random() * 0.03,
      lastUpdated: new Date()
    };
  }

  /**
   * Get mock financial data
   */
  private getMockFinancialMetrics(symbol: string): any {
    // Generate realistic mock financial metrics
    return {
      eps: (Math.random() * 10 + 1).toFixed(2),
      revenue: (Math.random() * 50 + 10) * 1000000000,
      revenueGrowth: (Math.random() * 0.3 - 0.05).toFixed(2),
      profitMargin: (Math.random() * 0.4).toFixed(2),
      beta: (0.5 + Math.random() * 2).toFixed(2),
      yearHigh: (100 + Math.random() * 300).toFixed(2),
      yearLow: (50 + Math.random() * 100).toFixed(2),
      grossProfit: (Math.random() * 30 + 5) * 1000000000,
      operatingIncome: (Math.random() * 20 + 2) * 1000000000,
      netIncome: (Math.random() * 15 + 1) * 1000000000,
      debtToEquity: (Math.random() * 2).toFixed(2),
      quickRatio: (0.5 + Math.random() * 2).toFixed(2),
      returnOnAssets: (Math.random() * 0.2).toFixed(2),
      returnOnEquity: (Math.random() * 0.35).toFixed(2)
    };
  }

  /**
   * Generate mock historical stock data
   */
  private getMockHistoricalData(symbol: string, period: string): any[] {
    const data = [];
    const today = new Date();
    const startPrice = 100 + Math.random() * 200;
    let currentPrice = startPrice;
    let volatility = 0.02; // 2% daily volatility
    let trend = 0.0005; // slight upward trend
    let days = 30;
    
    // Adjust parameters based on period
    switch(period) {
      case '1D':
        days = 1;
        volatility = 0.005;
        // Generate hourly data
        for (let i = 0; i < 8; i++) {
          const date = new Date();
          date.setHours(9 + i);
          date.setMinutes(30);
          const change = (Math.random() - 0.5) * 2 * volatility * currentPrice + trend * currentPrice;
          currentPrice = Math.max(0.1, currentPrice + change);
          data.push({
            date: date.getTime(),
            value: parseFloat(currentPrice.toFixed(2))
          });
        }
        return data;
      case '1W':
        days = 7;
        volatility = 0.01;
        break;
      case '1M':
        days = 30;
        break;
      case '3M':
        days = 90;
        trend = 0.0002;
        break;
      case '6M':
        days = 180;
        trend = 0.0001;
        break;
      case '1Y':
        days = 365;
        volatility = 0.025;
        trend = 0.0001;
        break;
      case '5Y':
        days = 365 * 5;
        volatility = 0.03;
        trend = 0.00008;
        break;
    }
    
    // Generate daily data
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(today.getDate() - i);
      
      // Add some randomness but with a trend
      const change = (Math.random() - 0.5) * 2 * volatility * currentPrice + trend * currentPrice;
      currentPrice = Math.max(0.1, currentPrice + change);
      
      // Don't add weekend data
      const dayOfWeek = date.getDay();
      if (dayOfWeek !== 0 && dayOfWeek !== 6) {
        data.push({
          date: date.getTime(),
          value: parseFloat(currentPrice.toFixed(2)),
          volume: Math.floor(1000000 + Math.random() * 10000000)
        });
      }
    }
    
    return data;
  }

  /**
   * Transform Alpha Vantage company profile response to our format
   */
  private transformCompanyProfile(response: any): any {
    if (!response) {
      return null;
    }
    
    return {
      symbol: response.Symbol || '',
      companyName: response.Name || '',
      price: parseFloat(response['50DayMovingAverage']) || 0,
      change: 0, // Not provided directly, would need additional API call
      changePercent: 0, // Not provided directly
      marketCap: parseFloat(response.MarketCapitalization) || 0,
      pe: parseFloat(response.PERatio) || 0,
      industry: response.Industry || '',
      sector: response.Sector || '',
      exchange: response.Exchange || '',
      description: response.Description || '',
      employees: parseInt(response.FullTimeEmployees, 10) || 0,
      country: response.Country || '',
      ceo: response.CEO || '',
      website: response.Website || '',
      logoUrl: `https://coincodex.com/stonks_api/get_logo/${response.symbol.toUpperCase()}/`,//`https://coincodex.com/stonks_api/get_logo/${response.symbol}/`
      dividend: parseFloat(response.DividendYield) || 0,
      lastUpdated: new Date()
    };
  }

  /**
   * Transform Alpha Vantage financial data to our format
   */
  private transformFinancialData(response: any): any {
    if (!response || !response.annualReports || response.annualReports.length === 0) {
      return null;
    }
    
    const latestReport = response.annualReports[0];
    const previousReport = response.annualReports.length > 1 ? response.annualReports[1] : null;
    
    const revenue = parseFloat(latestReport.totalRevenue) || 0;
    const prevRevenue = previousReport ? parseFloat(previousReport.totalRevenue) || 0 : 0;
    
    // Calculate growth
    const revenueGrowth = prevRevenue > 0 ? (revenue - prevRevenue) / prevRevenue : 0;
    
    return {
      eps: latestReport.reportedEPS || 0,
      revenue: revenue,
      revenueGrowth: revenueGrowth,
      profitMargin: revenue > 0 ? (parseFloat(latestReport.netIncome) || 0) / revenue : 0,
      grossProfit: parseFloat(latestReport.grossProfit) || 0,
      operatingIncome: parseFloat(latestReport.operatingIncome) || 0,
      netIncome: parseFloat(latestReport.netIncome) || 0,
      // These would typically come from balance sheet, so we'd mock them
      beta: 1 + Math.random() * 1.5,
      yearHigh: 0, // Would need additional API call
      yearLow: 0 // Would need additional API call
    };
  }

  /**
   * Transform historical data from Alpha Vantage
   */
  private transformHistoricalData(response: any, interval: string): any[] {
    interface HistoricalDataPoint {
      date: number;
      value: number;
      volume: number;
    }
    
    const result: HistoricalDataPoint[] = [];
    let timeSeriesKey = '';
    
    // Determine the time series key based on the interval
    if (interval === 'daily') {
      timeSeriesKey = 'Time Series (Daily)';
    } else if (interval === 'weekly') {
      timeSeriesKey = 'Weekly Time Series';
    } else {
      timeSeriesKey = `Time Series (${interval})`;
    }
    
    const timeSeries = response[timeSeriesKey];
    if (!timeSeries) {
      return [];
    }
    
    // Convert the object to an array of data points
    Object.keys(timeSeries).forEach(date => {
      const dataPoint = timeSeries[date];
      result.push({
        date: new Date(date).getTime(),
        value: parseFloat(dataPoint['4. close']),
        volume: parseInt(dataPoint['5. volume'], 10)
      });
    });
    
    // Sort by date ascending
    return result.sort((a, b) => a.date - b.date);
  }

  getCompanyProfileN(symbol: string): Observable<FinnhubProfileResponse> {
    if (this.useMockData) {
      return of(this.getMockCompanyProfile(symbol)).pipe(delay(500));
    }

    // Using Finnhub companyProfile2 endpoint
    return this.http.get(`${this.finnhubApiUrl}/stock/profile2`, {
      params: {
        symbol: symbol,
        token: this.finnhubApiKey
      }
    }).pipe(
      map((response: any) => this.transformFinnhubProfile(response)),
      catchError(error => {
        console.error('Error fetching company profile from Finnhub:', error);
        return of(this.getMockCompanyProfile1(symbol));
      })
    );
  }

  private transformFinnhubProfile(response: any): FinnhubProfileResponse {
    return {
      name: response.name || 'Unknown',
      marketCapitalization: response.marketCapitalization || 0,
      finnhubIndustry: response.finnhubIndustry || 'Unknown',
      peRatio: undefined, // Not available in companyProfile2; fetch separately if needed
      dividendYield: undefined // Not available in companyProfile2; fetch separately if needed
    };
  }

  private getMockCompanyProfile1(symbol: string): FinnhubProfileResponse {

    // Mock data for fallback
    return {
      name: `${symbol} Inc`,
      marketCapitalization: 1000000,
      finnhubIndustry: 'Technology',
      peRatio: 15.5,
      dividendYield: 0.02
    };
  }
 




  getCompanyProfile(symbol: string): Observable<any> {
    if (this.useMockData) {
      return of(this.getMockCompanyProfile(symbol)).pipe(delay(500));
    }
    
    // Using Alpha Vantage OVERVIEW endpoint
    return this.http.get(`${this.apiUrl}`, {
      params: {
        function: 'OVERVIEW',
        symbol: symbol,
        apikey: this.apiKey
      }
    }).pipe(
      map(response => this.transformCompanyProfile(response)),
      catchError(error => {
        console.error('Error fetching company profile:', error);
        return of(this.getMockCompanyProfile(symbol));
      })
    );
  }




  getStocks(): Observable<Stock[]> {
    if (this.mockMode) {
      return of(this.generateMockStocks());
    }

    // First get the list of all S&P 500 symbols
    return this.apiService.get1<FinnhubConstituentsResponse>('https://finnhub.io/api/v1/index/constituents', {
      symbol: 'SPX',
      token: this.finnhubApiKey
    }).pipe(
      map(response => response.constituents),
      switchMap(symbols => {
        // Take a reasonable batch size (API limits)
        const batchSize = 100;
        const batchedSymbols = symbols.slice(0, batchSize);
        
        // Create an array of observables for each stock
        const stockRequests = batchedSymbols.map(symbol => this.getStockDetails(symbol));
        
        // Combine all requests
        return forkJoin(stockRequests);
      }),
      catchError(error => {
        console.error('Error fetching S&P 500 stocks:', error);
        return of(this.generateMockStocks()); // Fallback to mock data
      })
    );
  }

  /**
   * Get detailed information for a specific stock
   */
  getStockDetails(symbol: string): Observable<Stock> {
    // Create multiple requests for different data points
    const quoteRequest = this.apiService.get1<FinnhubQuoteResponse>('https://finnhub.io/api/v1/quote', {
      symbol: symbol,
      token: this.finnhubApiKey
    });
    
    const profileRequest = this.apiService.get1<FinnhubProfileResponse>('https://finnhub.io/api/v1/stock/profile2', {
      symbol: symbol,
      token: this.finnhubApiKey
    });

    // Combine the responses
    return forkJoin([quoteRequest, profileRequest]).pipe(
      map(([quote, profile]) => {
        return {
          symbol: symbol,
          companyName: profile.name,
          price: quote.c, // Current price
          change: quote.d, // Price change
          changePercent: quote.dp, // Percent change
          volume: quote.v, // Volume
          marketCap: profile.marketCapitalization * 1000000, // Convert from millions
          sector: profile.finnhubIndustry,
          pe: profile.peRatio || 0,
          dividend: profile.dividendYield || 0
        };
      }),
      catchError(error => {
        console.error(`Error fetching details for ${symbol}:`, error);
        // Return a placeholder object in case of error
        return of({
          symbol: symbol,
          companyName: symbol,
          price: 0,
          change: 0,
          changePercent: 0,
          volume: 0,
          marketCap: 0,
          sector: 'Unknown',
          pe: 0,
          dividend: 0
        });
      })
    );
  }

  /**
   * Get historical data for a stock
   */
  getHistoricalData(symbol: string, resolution: string = 'D', from: number, to: number): Observable<any[]> {
    if (this.mockMode) {
      return of(this.generateMockHistoricalData(symbol));
    }

    return this.apiService.get1<FinnhubCandleResponse>('https://finnhub.io/api/v1/stock/candle', {
      symbol: symbol,
      resolution: resolution, // 'D' for day, 'W' for week
      from: from,
      to: to,
      token: this.finnhubApiKey
    }).pipe(
      map(response => {
        // Transform the candle data to a usable format for charts
        const { c, t, v } = response; // close prices, timestamps, volumes
        return t.map((timestamp: number, index: number) => ({
          timestamp: timestamp * 1000, // Convert to milliseconds
          close: c[index],
          volume: v[index]
        }));
      }),
      catchError(error => {
        console.error(`Error fetching historical data for ${symbol}:`, error);
        return of(this.generateMockHistoricalData(symbol));
      })
    );
  }

  /**
   * Search for stocks by keyword
   */
  searchStocks(keyword: string): Observable<Stock[]> {
    if (this.mockMode) {
      return of(this.generateMockStocks().filter(
        stock => stock.symbol.toLowerCase().includes(keyword.toLowerCase()) || 
                stock.companyName.toLowerCase().includes(keyword.toLowerCase())
      ));
    }

    return this.apiService.get1<FinnhubSearchResponse>('https://finnhub.io/api/v1/search', {
      q: keyword,
      token: this.finnhubApiKey
    }).pipe(
      switchMap(response => {
        // Filter for just stocks (not other securities)
        const stockSymbols = response.result
          .filter((item) => item.type === 'Common Stock')
          .map((item) => item.symbol)
          .slice(0, 10); // Limit results
        
        if (stockSymbols.length === 0) {
          return of([]);
        }
        
        // Get details for each symbol
        const stockRequests = stockSymbols.map((symbol: string) => this.getStockDetails(symbol));
        return forkJoin(stockRequests);
      }),
      catchError(error => {
        console.error('Error searching stocks:', error);
        return of([]); 
      })
    );
  }


  /**
   * Generate mock S&P 500 stocks data
   */
  private generateMockStocks(): Stock[] {
    // List of major sectors
    const sectors = [
      'Technology', 'Financial Services', 'Healthcare', 'Communication Services',
      'Consumer Cyclical', 'Industrials', 'Consumer Defensive', 'Energy',
      'Utilities', 'Real Estate', 'Basic Materials'
    ];
    
    // List of top S&P 500 companies to include in mock data
    const companies = [
      { symbol: "AAPL", name: "Apple Inc.", sector: "Technology" },
  { symbol: "MSFT", name: "Microsoft Corporation", sector: "Technology" },
  { symbol: "AMZN", name: "Amazon.com Inc.", sector: "Consumer Cyclical" },
  { symbol: "NVDA", name: "NVIDIA Corporation", sector: "Technology" },
  { symbol: "GOOGL", name: "Alphabet Inc. Class A", sector: "Communication Services" },
  { symbol: "META", name: "Meta Platforms Inc.", sector: "Communication Services" },
  { symbol: "TSLA", name: "Tesla Inc.", sector: "Consumer Cyclical" },
  { symbol: "BRK.B", name: "Berkshire Hathaway Inc. Class B", sector: "Financial Services" },
  { symbol: "UNH", name: "UnitedHealth Group Inc.", sector: "Healthcare" },
  { symbol: "JNJ", name: "Johnson & Johnson", sector: "Healthcare" },
  { symbol: "JPM", name: "JPMorgan Chase & Co.", sector: "Financial Services" },
  { symbol: "V", name: "Visa Inc. Class A", sector: "Financial Services" },
  { symbol: "PG", name: "Procter & Gamble Co.", sector: "Consumer Defensive" },
  { symbol: "MA", name: "Mastercard Inc. Class A", sector: "Financial Services" },
  { symbol: "HD", name: "Home Depot Inc.", sector: "Consumer Cyclical" },
  { symbol: "CVX", name: "Chevron Corporation", sector: "Energy" },
  { symbol: "AVGO", name: "Broadcom Inc.", sector: "Technology" },
  { symbol: "MRK", name: "Merck & Co. Inc.", sector: "Healthcare" },
  { symbol: "ABBV", name: "AbbVie Inc.", sector: "Healthcare" },
  { symbol: "PEP", name: "PepsiCo Inc.", sector: "Consumer Defensive" },
  { symbol: "COST", name: "Costco Wholesale Corporation", sector: "Consumer Defensive" },
  { symbol: "WMT", name: "Walmart Inc.", sector: "Consumer Defensive" },
  { symbol: "XOM", name: "Exxon Mobil Corporation", sector: "Energy" },
  { symbol: "KO", name: "Coca-Cola Company", sector: "Consumer Defensive" },
  { symbol: "DIS", name: "Walt Disney Company", sector: "Communication Services" },
  { symbol: "CSCO", name: "Cisco Systems Inc.", sector: "Technology" },
  { symbol: "PFE", name: "Pfizer Inc.", sector: "Healthcare" },
  { symbol: "BAC", name: "Bank of America Corporation", sector: "Financial Services" },
  { symbol: "INTC", name: "Intel Corporation", sector: "Technology" },
  { symbol: "CMCSA", name: "Comcast Corporation Class A", sector: "Communication Services" },
  { symbol: "WFC", name: "Wells Fargo & Company", sector: "Financial Services" },
  { symbol: "T", name: "AT&T Inc.", sector: "Communication Services" },
  { symbol: "MCD", name: "McDonald's Corporation", sector: "Consumer Cyclical" },
  { symbol: "AMD", name: "Advanced Micro Devices Inc.", sector: "Technology" },
  { symbol: "NKE", name: "NIKE Inc. Class B", sector: "Consumer Cyclical" },
  { symbol: "LOW", name: "Lowe's Companies Inc.", sector: "Consumer Cyclical" },
  { symbol: "UPS", name: "United Parcel Service Inc. Class B", sector: "Industrials" },
  { symbol: "IBM", name: "International Business Machines Corporation", sector: "Technology" },
  { symbol: "BA", name: "Boeing Company", sector: "Industrials" },
  { symbol: "GE", name: "General Electric Company", sector: "Industrials" },
  { symbol: "CAT", name: "Caterpillar Inc.", sector: "Industrials" },
  { symbol: "GS", name: "Goldman Sachs Group Inc.", sector: "Financial Services" },
  { symbol: "RTX", name: "RTX Corporation", sector: "Industrials" },
  { symbol: "SBUX", name: "Starbucks Corporation", sector: "Consumer Cyclical" },
  { symbol: "MMM", name: "3M Company", sector: "Industrials" },
  { symbol: "DE", name: "Deere & Company", sector: "Industrials" },
  { symbol: "LMT", name: "Lockheed Martin Corporation", sector: "Industrials" },
  { symbol: "SPGI", name: "S&P Global Inc.", sector: "Financial Services" },
  { symbol: "BKNG", name: "Booking Holdings Inc.", sector: "Consumer Cyclical" },
  { symbol: "ADBE", name: "Adobe Inc.", sector: "Technology" },
  { symbol: "ORCL", name: "Oracle Corporation", sector: "Technology" },
  { symbol: "QCOM", name: "QUALCOMM Incorporated", sector: "Technology" },
  { symbol: "TXN", name: "Texas Instruments Incorporated", sector: "Technology" },
  { symbol: "AMGN", name: "Amgen Inc.", sector: "Healthcare" },
  { symbol: "MDT", name: "Medtronic plc", sector: "Healthcare" },
  { symbol: "HON", name: "Honeywell International Inc.", sector: "Industrials" },
  { symbol: "PM", name: "Philip Morris International Inc.", sector: "Consumer Defensive" },
  { symbol: "UNP", name: "Union Pacific Corporation", sector: "Industrials" },
  { symbol: "COP", name: "ConocoPhillips", sector: "Energy" },
  { symbol: "BMY", name: "Bristol-Myers Squibb Company", sector: "Healthcare" },
  { symbol: "CRM", name: "Salesforce Inc.", sector: "Technology" },
  { symbol: "GILD", name: "Gilead Sciences Inc.", sector: "Healthcare" },
  { symbol: "SCHW", name: "Charles Schwab Corporation", sector: "Financial Services" },
  { symbol: "MO", name: "Altria Group Inc.", sector: "Consumer Defensive" },
  { symbol: "C", name: "Citigroup Inc.", sector: "Financial Services" },
  { symbol: "BLK", name: "BlackRock Inc.", sector: "Financial Services" },
  { symbol: "AXP", name: "American Express Company", sector: "Financial Services" },
  { symbol: "AMT", name: "American Tower Corporation", sector: "Real Estate" },
  { symbol: "CI", name: "Cigna Corporation", sector: "Healthcare" },
  { symbol: "DHR", name: "Danaher Corporation", sector: "Healthcare" },
  { symbol: "TMO", name: "Thermo Fisher Scientific Inc.", sector: "Healthcare" },
  { symbol: "ABT", name: "Abbott Laboratories", sector: "Healthcare" },
  { symbol: "NEE", name: "NextEra Energy Inc.", sector: "Utilities" },
  { symbol: "DUK", name: "Duke Energy Corporation", sector: "Utilities" },
  { symbol: "SO", name: "Southern Company", sector: "Utilities" },
  { symbol: "EOG", name: "EOG Resources Inc.", sector: "Energy" },
  { symbol: "SLB", name: "Schlumberger Limited", sector: "Energy" },
  { symbol: "OXY", name: "Occidental Petroleum Corporation", sector: "Energy" },
  { symbol: "TGT", name: "Target Corporation", sector: "Consumer Defensive" },
  { symbol: "CVS", name: "CVS Health Corporation", sector: "Healthcare" },
  { symbol: "ELV", name: "Elevance Health Inc.", sector: "Healthcare" },
  { symbol: "SYK", name: "Stryker Corporation", sector: "Healthcare" },
  { symbol: "TJX", name: "TJX Companies Inc.", sector: "Consumer Cyclical" },
  { symbol: "PNC", name: "PNC Financial Services Group Inc.", sector: "Financial Services" },
  { symbol: "USB", name: "U.S. Bancorp", sector: "Financial Services" },
  { symbol: "MS", name: "Morgan Stanley", sector: "Financial Services" },
  { symbol: "FDX", name: "FedEx Corporation", sector: "Industrials" },
  { symbol: "CSX", name: "CSX Corporation", sector: "Industrials" },
  { symbol: "NSC", name: "Norfolk Southern Corporation", sector: "Industrials" },
  { symbol: "ITW", name: "Illinois Tool Works Inc.", sector: "Industrials" },
  { symbol: "GD", name: "General Dynamics Corporation", sector: "Industrials" },
  { symbol: "CL", name: "Colgate-Palmolive Company", sector: "Consumer Defensive" },
  { symbol: "KMB", name: "Kimberly-Clark Corporation", sector: "Consumer Defensive" },
  { symbol: "AEP", name: "American Electric Power Company Inc.", sector: "Utilities" },
  { symbol: "D", name: "Dominion Energy Inc.", sector: "Utilities" },
  { symbol: "EXC", name: "Exelon Corporation", sector: "Utilities" },
  { symbol: "PLD", name: "Prologis Inc.", sector: "Real Estate" },
  { symbol: "CCI", name: "Crown Castle Inc.", sector: "Real Estate" },
  { symbol: "PSA", name: "Public Storage", sector: "Real Estate" },
  { symbol: "EQIX", name: "Equinix Inc.", sector: "Real Estate" }

    ];
    
    // Generate additional random mock companies to reach 100 entries
    // for (let i = companies.length; i < 100; i++) {
    //   const sectorIndex = Math.floor(Math.random() * sectors.length);
    //   companies.push({
    //     symbol: `STOCK${i}`,
    //     name: `Mock Company ${i}`,
    //     sector: sectors[sectorIndex]
    //   });
    // }
    
    // Convert companies to Stock objects with randomized data
    return companies.map(company => {
      const basePrice = Math.random() * 300 + 50; // Random price between $50 and $350
      const change = (Math.random() * 10 - 5); // Random change between -$5 and +$5
      const changePercent = (change / basePrice) * 100;
      const volume = Math.floor(Math.random() * 10000000) + 100000; // Random volume
      const marketCap = basePrice * (Math.random() * 1000000000 + 1000000000); // Random market cap
      
      return {
        symbol: company.symbol,
        companyName: company.name,
        price: parseFloat(basePrice.toFixed(2)),
        change: parseFloat(change.toFixed(2)),
        changePercent: parseFloat(changePercent.toFixed(2)),
        volume: volume,
        marketCap: marketCap,
        sector: company.sector,
        pe: parseFloat((Math.random() * 50 + 5).toFixed(2)), // Random P/E between 5 and 55
        dividend: parseFloat((Math.random() * 5).toFixed(2))  // Random dividend yield between 0 and 5%
      };
    });
  }

  /**
   * Generate mock historical data for charts
   */
  private generateMockHistoricalData(symbol: string): any[] {
    const data = [];
    const today = new Date();
    const startPrice = Math.random() * 300 + 50; // Random start price
    let currentPrice = startPrice;
    
    // Generate 90 days of data
    for (let i = 90; i >= 0; i--) {
      const date = new Date();
      date.setDate(today.getDate() - i);
      
      // Add some randomness to price movements but with a trend
      const change = (Math.random() - 0.48) * 5; // Slightly biased toward growth
      currentPrice = Math.max(currentPrice + change, 1); // Ensure price doesn't go below $1
      
      data.push({
        timestamp: date.getTime(),
        close: parseFloat(currentPrice.toFixed(2)),
        volume: Math.floor(Math.random() * 10000000) + 100000
      });
    }
    
    return data;
  }












  getTopSnPCompanies(): Company[] {
    // In a real app, this would be an API call
    return this.mockCompanies;
  }





  
  testApiConnection(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/v1/open-close/AAPL/2023-01-09?adjusted=true&apiKey=${this.apiKey}`).pipe(
      catchError(error => {
        console.error('API Test Error:', error);
        return throwError(() => new Error('API connection test failed'));
      })
    );
  }


  getStockData(symbol: string, range: '1D' | '1W' | '1M' | '1Y'): Observable<any[]> {
    
  let functionType = 'TIME_SERIES_INTRADAY';
  let interval = '60min';

  switch (range) {
    case '1D':
      functionType = 'TIME_SERIES_INTRADAY';
      interval = '60min';
      break;
    case '1W':
      functionType = 'TIME_SERIES_DAILY';
      break;
    case '1M':
      functionType = 'TIME_SERIES_WEEKLY';
      break;
    case '1Y':
      functionType = 'TIME_SERIES_MONTHLY';
      break;
  }

  const params = {
    function: functionType,
    symbol: symbol,
    interval: interval,
    apikey: this.apiKey,
    outputsize: 'full'
  };
  const url = `${this.apiUrl}?function=${functionType}&symbol=${symbol}&interval=${interval}&apikey=${this.apiKey}&outputsize=full`;
  return this.http.get<any>(this.apiUrl, { params }).pipe(
    map(response => {
      const timeSeriesKey = functionType === 'TIME_SERIES_INTRADAY' ? `Time Series (${interval})` :
                            functionType === 'TIME_SERIES_DAILY' ? 'Time Series (Daily)' :
                            functionType === 'TIME_SERIES_WEEKLY' ? 'Weekly Time Series' :
                            '	TIME_SERIES_MONTHLY';

      const timeSeries = response[timeSeriesKey];
      if (!timeSeries) {
        throw new Error(`No data available for symbol ${symbol} in range ${range}`);
      }

      return Object.keys(timeSeries).map(timestamp => ({
        x: new Date(timestamp).getTime(),
        open: parseFloat(timeSeries[timestamp]['1. open']),
        high: parseFloat(timeSeries[timestamp]['2. high']),
        low: parseFloat(timeSeries[timestamp]['3. low']),
        close: parseFloat(timeSeries[timestamp]['4. close']),
        volume: parseInt(timeSeries[timestamp]['5. volume'])
      })).reverse();
    })
  );
}


}