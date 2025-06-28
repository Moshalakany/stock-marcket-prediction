namespace StockMarket.DTOs.Sentiment
{
    public class BatchSentimentAnalysisRequestDto
    {
        public List<string> texts { get; set; } = new List<string>();
    }
}
