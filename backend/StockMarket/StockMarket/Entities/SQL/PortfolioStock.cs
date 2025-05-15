using System.Text.Json.Serialization;
namespace StockMarket.Entities.SQL
{
    public class PortfolioStock
    {
        public int PortfolioId { get; set; }
        public string StockSymbol { get; set; } = default!; // Changed from string? to string
        public int Quantity { get; set; }
        public decimal PurchasePrice { get; set; }
        public DateTime PurchaseDate { get; set; }
        public decimal CurrentPrice { get; set; }
        public decimal TotalValue => Quantity * CurrentPrice;

        public decimal GainLoss => (CurrentPrice - PurchasePrice) * Quantity;
        public decimal GainLossPercentage => PurchasePrice == 0 || Quantity == 0 ? 0 : GainLoss / (PurchasePrice * Quantity) * 100; // Added check for division by zero
        //Navigation properties
        [JsonIgnore]
        public virtual Portfolio? Portfolio { get; set; }
        [JsonIgnore]
        public virtual Stock? Stock { get; set; }
    }
}
