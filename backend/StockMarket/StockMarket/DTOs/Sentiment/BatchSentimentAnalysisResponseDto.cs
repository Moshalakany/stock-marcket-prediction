namespace StockMarket.DTOs.Sentiment
{
    public class BatchSentimentAnalysisResponseDto
    {
        public List<SentimentAnalysisResponseDto> Results { get; set; } = new List<SentimentAnalysisResponseDto>();
        public int Count { get; set; }
        public string sentiment_mode { get; set; } = string.Empty;
    }
}
