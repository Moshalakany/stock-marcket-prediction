using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using StockMarket.Entities.SQL;

namespace StockMarket.Data.config
{
    public class StockConfiguration : IEntityTypeConfiguration<Stock>
    {
        public void Configure(EntityTypeBuilder<Stock> builder)
        {
            builder.HasKey(s => s.symbol);
            
            builder.Property(s => s.symbol)
                .IsRequired();
            
            builder.Property(s => s.CompanyName);
            
            builder.Property(s => s.Sector);
            builder.HasIndex(s=>s.Sector);
        }
    }
}
