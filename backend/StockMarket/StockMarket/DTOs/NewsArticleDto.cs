namespace StockMarket.DTOs
{
    public class NewsArticleDto
    {
        public string symbol { get; set; }
        public DateTime datetime { get; set; }
        public string title { get; set; }
        public string source { get; set; }
        public string link { get; set; }
    }
    
    public class NewsArticleCreateDto
    {
        public string Symbol { get; set; }
        public DateTime DateTime { get; set; }
        public string Title { get; set; }
        public string Source { get; set; }
        public string Link { get; set; }
    }
    
    public class NewsScrapingRequestDto
    {
        public string Symbol { get; set; }
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
    }
}
