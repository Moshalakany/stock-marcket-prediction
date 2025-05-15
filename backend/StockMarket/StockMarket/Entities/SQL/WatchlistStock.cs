using System.Text.Json.Serialization;

namespace StockMarket.Entities.SQL
{
    public class WatchlistStock
    {
        public int WatchlistId { get; set; }
        public string? StockSymbol { get; set; }
        public DateTime AddedAt { get; set; }

        //Navigation properties
        [JsonIgnore]
        public virtual Watchlist? Watchlist { get; set; }
        [JsonIgnore]
        public virtual Stock? Stock { get; set; }
    }
}
