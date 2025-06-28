using StockMarket.DTOs.Sentiment;

namespace StockMarket.Services.Interfaces
{
    public interface ISentimentService
    {
        Task<SentimentHealthResponseDto> CheckHealthAsync();
        Task<SentimentAnalysisResponseDto> AnalyzeSentimentAsync(SentimentAnalysisRequestDto request);
        Task<BatchSentimentAnalysisResponseDto> AnalyzeBatchSentimentAsync(BatchSentimentAnalysisRequestDto request);
        Task<BatchSentimentAnalysisResponseDto> AnalyzeNewsSentimentAsync(string symbol);
    }
}
