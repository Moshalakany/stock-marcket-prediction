namespace StockMarket.Entities.SQL
{
    public class Portfolio
    {
        public int PortfolioId { get; set; }
        public int UserId { get; set; }
        public decimal TotalValue { get; set; }
        public DateTime CreatedAt { get; set; }
        // Navigation property
        public virtual User? User { get; set; }
        public virtual ICollection<PortfolioStock>? PortfolioStocks { get; set; } = new List<PortfolioStock>();
    }
}
