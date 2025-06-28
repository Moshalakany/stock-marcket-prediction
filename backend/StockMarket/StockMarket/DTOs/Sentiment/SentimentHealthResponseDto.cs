namespace StockMarket.DTOs.Sentiment
{
    public class SentimentHealthResponseDto
    {
        public string Status { get; set; } = string.Empty;
        public bool ModelLoaded { get; set; }
    }
}
