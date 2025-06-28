namespace StockMarket.DTOs
{
    public class WatchlistDto
    {
        public string? Name { get; set; }
    }

    public class WatchlistStockDto
    {
        public string Symbol { get; set; } = string.Empty;
    }
}
