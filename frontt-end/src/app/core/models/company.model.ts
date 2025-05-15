export interface Company {
    ticker: string;
    name: string;
    price: number;
    change: number;
    percentChange: number;
    marketCap?: string;
    sector?: string;
    logoUrl?: string;
  }