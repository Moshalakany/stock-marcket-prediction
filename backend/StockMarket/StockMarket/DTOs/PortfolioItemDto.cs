using System.ComponentModel.DataAnnotations;

namespace StockMarket.DTOs
{
    public class PortfolioItemDto
    {
        public string? Symbol { get; set; } 


        public int Quantity { get; set; }

        public decimal PurchasePrice { get; set; }
    }
}
