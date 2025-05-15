using System.ComponentModel.DataAnnotations;

namespace StockMarket.Entities.SQL
{
    public class Stock
    {
        [Key]
        public string Symbol { get; set; } = default!;
        public string? CompanyName { get; set; }
        public string? Sector { get; set; }
        //navigation properties
        public virtual ICollection<PortfolioStock>? PortfolioStocks { get; set; }
        public virtual ICollection<WatchlistStock>? WatchlistStocks { get; set; }
    }
}
