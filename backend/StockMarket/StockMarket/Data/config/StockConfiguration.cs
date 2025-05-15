using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using StockMarket.Entities.SQL;

namespace StockMarket.Data.config
{
    public class StockConfiguration : IEntityTypeConfiguration<Stock>
    {
        public void Configure(EntityTypeBuilder<Stock> builder)
        {
            builder.HasKey(s => s.Symbol);
            
            builder.Property(s => s.Symbol)
                .IsRequired();
            
            builder.Property(s => s.CompanyName);
            
            builder.Property(s => s.Sector);
            builder.HasIndex(s=>s.Sector);
        }
    }
}
