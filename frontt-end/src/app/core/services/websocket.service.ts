

import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class WebsocketService {
  private ws!: WebSocket;
  private stockDataSubject = new Subject<any>();
  private apiKey = 'YOUR_FINNHUB_API_KEY'; // Replace with your free Finnhub API key

  constructor() {
    this.connect();
  }

  private connect() {
    this.ws = new WebSocket(`wss://ws.finnhub.io?token=${this.apiKey}`);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'trade') {
        this.stockDataSubject.next(data.data);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.connect(); // Reconnect on close
    };
  }

  subscribeToStock(ticker: string) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'subscribe', symbol: ticker }));
    }
  }

  unsubscribeFromStock(ticker: string) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'unsubscribe', symbol: ticker }));
    }
  }

  getStockUpdates(): Observable<any> {
    return this.stockDataSubject.asObservable();
  }

  closeConnection() {
    this.ws.close();
  }
}