using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using StockMarket.Entities.SQL;

namespace StockMarket.Data.config
{
    public class WatchlistConfiguration:IEntityTypeConfiguration<Watchlist>
    {
        public void Configure(EntityTypeBuilder<Watchlist> builder)
        {
            builder.HasKey(s => s.WatchlistId);
            builder.Property(s => s.WatchlistId)
                .ValueGeneratedOnAdd()
                .HasColumnName("WatchlistId")
                .HasColumnType("int");
            builder.Property(s => s.UserId)
                .IsRequired()
                .HasColumnName("UserId")
                .HasColumnType("int");
            builder.Property(s => s.Name)
                .HasMaxLength(100)
                .HasColumnName("Name")
                .HasColumnType("nvarchar(100)");
            builder.HasOne(s => s.User)
                .WithOne(s => s.Watchlist)
                .HasForeignKey<Watchlist>(s => s.UserId);
        }
    }
}
