namespace StockMarket.DTOs
{
    public class NewsArticleWithSentimentDto : NewsArticleDto
    {
        public string Sentiment { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public Dictionary<string, double> SentimentScores { get; set; } = new Dictionary<string, double>();
    }
}
