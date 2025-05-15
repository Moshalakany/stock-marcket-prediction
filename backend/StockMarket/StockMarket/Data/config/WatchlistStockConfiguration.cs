using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using StockMarket.Entities.SQL;

namespace StockMarket.Data.config
{
    public class WatchlistStockConfiguration : IEntityTypeConfiguration<WatchlistStock>
    {
        public void Configure(EntityTypeBuilder<WatchlistStock> builder)
        {
            builder.HasKey(s => new { s.WatchlistId, s.StockSymbol });
            builder.HasIndex(s => s.WatchlistId);
            //configer one to many relationship between watchlist and stock
            builder.HasOne(s => s.Watchlist)
                .WithMany(s => s.WatchlistStocks)
                .HasForeignKey(s => s.WatchlistId)
                .OnDelete(DeleteBehavior.Cascade);
            builder.Property(s => s.StockSymbol)
                .IsRequired();
            builder.HasOne(ps => ps.Stock)
                .WithMany(s => s.WatchlistStocks)
                .HasForeignKey(ps => ps.StockSymbol)
                .OnDelete(DeleteBehavior.Restrict);
        }
    }
}
