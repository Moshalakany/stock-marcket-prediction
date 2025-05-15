namespace StockMarket.Entities.SQL
{
    public class Watchlist
    {
        public int WatchlistId { get; set; }
        public int UserId { get; set; }

        //navigation properties
        public virtual User? User { get; set; }
        public virtual ICollection<WatchlistStock>? WatchlistStocks { get; set; }
    }
}
