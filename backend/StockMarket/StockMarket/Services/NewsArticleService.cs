using Microsoft.EntityFrameworkCore;
using MongoDB.Driver;
using StockMarket.Data;
using StockMarket.DTOs;
using StockMarket.Entities.NonSQL;
using StockMarket.Services.Interfaces;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace StockMarket.Services
{
    // Create a proper class for the message to avoid serialization issues
    public class NewsScrapingRequest
    {
        required public string? Symbol { get; set; } 
        public string StartDate { get; set; }
        public string EndDate { get; set; }
    }

    public class NewsArticleService : INewsArticleService
    {
        private readonly HttpClient _httpClient;
        private readonly JsonSerializerOptions _jsonOptions;

        public NewsArticleService(IHttpClientFactory httpClientFactory)
        {
            _httpClient = httpClientFactory.CreateClient();
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };
        }

        public async Task<IEnumerable<NewsArticleDto>> GetLatestNewsAsync() 
        {
            var scrappingEndpoint = "http://127.0.0.1:5000/api/news/latest";
            string[] importantSymbols = { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA" };
            List<NewsArticleDto> newsArticles = new List<NewsArticleDto>();
            
            foreach (var symbol in importantSymbols)
            {
                try
                {
                    var url = $"{scrappingEndpoint}/{symbol}";
                    var response = await _httpClient.GetAsync(url);
                    if (response.IsSuccessStatusCode)
                    {
                        var content = await response.Content.ReadAsStringAsync();
                        var articles = JsonSerializer.Deserialize<List<NewsArticleDto>>(content, _jsonOptions);
                        if (articles != null)
                        {
                            newsArticles.AddRange(articles);
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Log the error but continue with other symbols
                    Console.WriteLine($"Error fetching news for {symbol}: {ex.Message}");
                }
            }
            return newsArticles;
        }

        public async Task<IEnumerable<NewsArticleDto>> GetNewsBySymbolAsync(string symbol)
        {
            var scrappingEndpoint = "http://127.0.0.1:5000/api/news/latest";
            List<NewsArticleDto> newsArticles = new List<NewsArticleDto>();
                var url = $"{scrappingEndpoint}/{symbol}";
                using (var client = new HttpClient())
                {
                    var response = await client.GetAsync(url);
                    if (response.IsSuccessStatusCode)
                    {
                        var content = await response.Content.ReadAsStringAsync();
                        Console.WriteLine(content);
                        var articles = JsonSerializer.Deserialize<List<NewsArticleDto>>(content);
                        if (articles != null)
                        {
                            newsArticles.AddRange(articles);
                        }
                    }
                }
            return newsArticles;
        }

        public async Task<IEnumerable<NewsArticleDto>> RequestLatestPageNewsScrapingAsync(string symbol)
        {
            var scrappingEndpoint = $"http://127.0.0.1:5000/api/news/latest-page/{symbol}";
            List<NewsArticleDto> newsArticles = new List<NewsArticleDto>();
            using (var client = new HttpClient())
            {
                var response = await client.GetAsync(scrappingEndpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var articles = JsonSerializer.Deserialize<List<NewsArticleDto>>(content);
                    if (articles != null)
                    {
                        newsArticles.AddRange(articles);
                    }
                }
            }
            return newsArticles;
        }

        public async Task<IEnumerable<NewsArticleDto>> RequestNewsScrapingByDateRangeAsync(string symbol, DateTime startDate, DateTime endDate)
        {
            var ConvertedStartDate = startDate.ToString("dd/MM/yyyy");
            var ConvertedEndDate = endDate.ToString("dd/MM/yyyy");
            var scrappingEndpoint = $"http://127.0.0.1:5000/api/news/range/{symbol}?start_date={ConvertedStartDate}&end_date={ConvertedEndDate}";
            List<NewsArticleDto> newsArticles = new List<NewsArticleDto>();
            using (var client = new HttpClient())
            {
                var response = await client.GetAsync(scrappingEndpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var articles = JsonSerializer.Deserialize<List<NewsArticleDto>>(content);
                    if (articles != null)
                    {
                        newsArticles.AddRange(articles);
                    }
                }
            }
            return newsArticles;
        }
    }
}
