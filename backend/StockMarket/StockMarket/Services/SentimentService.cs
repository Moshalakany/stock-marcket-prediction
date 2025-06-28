using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using StockMarket.DTOs;
using StockMarket.DTOs.Sentiment;
using StockMarket.Services.Interfaces;

namespace StockMarket.Services
{
    public class SentimentService : ISentimentService
    {
        private readonly HttpClient _httpClient;
        private readonly INewsArticleService _newsArticleService;
        private readonly string _sentimentApiBaseUrl;
        
        public SentimentService(
            HttpClient httpClient,
            INewsArticleService newsArticleService,
            IConfiguration configuration)
        {
            _httpClient = httpClient;
            _newsArticleService = newsArticleService;
            _sentimentApiBaseUrl = configuration["SentimentAnalysis:BaseUrl"] ?? "http://localhost:8000";
        }

        public async Task<SentimentHealthResponseDto> CheckHealthAsync()
        {
            var response = await _httpClient.GetAsync($"{_sentimentApiBaseUrl}/health");
            response.EnsureSuccessStatusCode();
            
            return await response.Content.ReadFromJsonAsync<SentimentHealthResponseDto>() 
                   ?? new SentimentHealthResponseDto();
        }

        public async Task<SentimentAnalysisResponseDto> AnalyzeSentimentAsync(SentimentAnalysisRequestDto request)
        {
            var json = JsonSerializer.Serialize(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            
            var response = await _httpClient.PostAsync($"{_sentimentApiBaseUrl}/sentiment/analyze", content);
            response.EnsureSuccessStatusCode();
            
            return await response.Content.ReadFromJsonAsync<SentimentAnalysisResponseDto>() 
                   ?? new SentimentAnalysisResponseDto();
        }

        public async Task<BatchSentimentAnalysisResponseDto> AnalyzeBatchSentimentAsync(BatchSentimentAnalysisRequestDto request)
        {
            var json = JsonSerializer.Serialize(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            
            var response = await _httpClient.PostAsync($"{_sentimentApiBaseUrl}/sentiment/batch", content);
            response.EnsureSuccessStatusCode();
            
            return await response.Content.ReadFromJsonAsync<BatchSentimentAnalysisResponseDto>() 
                   ?? new BatchSentimentAnalysisResponseDto();
        }

        public async Task<BatchSentimentAnalysisResponseDto> AnalyzeNewsSentimentAsync(string symbol)
        {
            // Get latest news for the symbol
            var newsArticles = await _newsArticleService.GetNewsBySymbolAsync(symbol);
            
            // Create a batch request with the news titles
            var batchRequest = new BatchSentimentAnalysisRequestDto
            {
                texts = newsArticles.Select(n => n.title).ToList()
            };
            
            // Send for batch sentiment analysis
            return await AnalyzeBatchSentimentAsync(batchRequest);
        }
    }
}
