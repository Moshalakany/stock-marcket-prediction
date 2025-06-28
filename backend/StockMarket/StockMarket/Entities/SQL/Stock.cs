using System.ComponentModel.DataAnnotations;

namespace StockMarket.Entities.SQL
{
    public class Stock
    {
        [Key]
        public string symbol { get; set; } = default!;
        public string? Industry { get; set; }
        public string? LongName { get; set; }
        public string? Longbusinesssummary { get; set; }
        public int FullTimeEmployees { get; set; }
        public string? CompanyName { get; set; }
        public string? Sector { get; set; }
        //navigation properties
        public virtual ICollection<PortfolioStock>? PortfolioStocks { get; set; }
        public virtual ICollection<WatchlistStock>? WatchlistStocks { get; set; }
    }
}
