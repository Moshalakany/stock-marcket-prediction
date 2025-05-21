using StockMarket.DTOs;
using StockMarket.DTOs.Sentiment;

namespace StockMarket.Services.Interfaces
{
    public interface INewsArticleService
    {
        Task<IEnumerable<NewsArticleDto>> GetLatestNewsAsync();
        Task<IEnumerable<NewsArticleDto>> GetNewsBySymbolAsync(string symbol);
        Task<IEnumerable<NewsArticleDto>> RequestNewsScrapingByDateRangeAsync(string symbol, DateTime startDate, DateTime endDate);
        Task<IEnumerable<NewsArticleDto>> RequestLatestPageNewsScrapingAsync(string symbol);
        //Task<IEnumerable<NewsArticleWithSentimentDto>> GetNewsBySentimentAsync(string symbol);
    }
}
