using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using StockMarket.Entities.SQL;

namespace StockMarket.Data.config
{
    public class PortfolioStockConfiguration: IEntityTypeConfiguration<PortfolioStock>
    {
        public void Configure(EntityTypeBuilder<PortfolioStock> builder)
        {
            builder.HasKey(s => new { s.PortfolioId, s.StockSymbol });
            builder.HasIndex(s => s.PortfolioId);
            //configer one to many relationship between portfolio and stock
            builder.HasOne(s => s.Portfolio)
                .WithMany(s => s.PortfolioStocks)
                .HasForeignKey(s => s.PortfolioId)
                .OnDelete(DeleteBehavior.Cascade);
            builder.Property(s => s.Quantity)
                .IsRequired();
            builder.Property(s=>s.PurchaseDate).HasConversion(
                s => s,
                s => DateTime.SpecifyKind(s, DateTimeKind.Utc)
            );
            builder.HasOne(ps => ps.Stock)
               .WithMany(s => s.PortfolioStocks)
               .HasForeignKey(ps => ps.StockSymbol)
               .OnDelete(DeleteBehavior.Restrict);
        }
    }
}
