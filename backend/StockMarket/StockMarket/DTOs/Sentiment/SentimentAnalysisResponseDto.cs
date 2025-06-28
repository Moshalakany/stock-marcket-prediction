namespace StockMarket.DTOs.Sentiment
{
    public class SentimentAnalysisResponseDto
    {
        public string Text { get; set; } = string.Empty;
        public string Sentiment { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public Dictionary<string, double> Scores { get; set; } = new Dictionary<string, double>();
    }
}
