using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using StockMarket.Entities.SQL;

namespace StockMarket.Data.config
{
    public class PortfolioConfiguration : IEntityTypeConfiguration<Portfolio>
    {
        public void Configure(EntityTypeBuilder<Portfolio> builder)
        {
            builder.HasKey(s => s.PortfolioId);
            builder.HasAlternateKey(s => s.UserId);
            //configer one to one relationship between user and potfolio
            builder.HasOne(s => s.User)
                .WithOne(s => s.Portfolio)
                .HasForeignKey<Portfolio>(s => s.UserId)
                .OnDelete(DeleteBehavior.Cascade);
            //configure foreign key

            builder.Property(s => s.UserId)
                .IsRequired();
            builder.Property(s=>s.CreatedAt).HasConversion(
                s => s,
                s => DateTime.SpecifyKind(s, DateTimeKind.Utc)
            );            
        }
    }
}
