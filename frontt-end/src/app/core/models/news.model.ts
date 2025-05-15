
export interface TickerInfo {
  symbol: string;
  changePercent: number;
  isFavorite: boolean;
}

export interface NewsArticle {
  id: string;
  title: string;
  source: string;
  publishedTime: string;
  url: string;
  relatedTickers?: TickerInfo[];
  sentiment?: string;       // Replace polarity
  confidence?: number;      // Replace sentimentScore
  scores?: {                // Add scores breakdown
    negative: number;
    neutral: number;
    positive: number;
  };
}