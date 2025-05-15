using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using StockMarket.Entities.SQL;

namespace StockMarket.Data.config
{
    public class UserConfiguration : IEntityTypeConfiguration<User>
    {
        void IEntityTypeConfiguration<User>.Configure(EntityTypeBuilder<User> builder)
        {
            builder.HasKey(s => s.UserId);
            builder.HasAlternateKey(s => s.Username);
            builder.HasAlternateKey(s => s.Email);
            builder.Property(s => s.UserId)
                .ValueGeneratedOnAdd()
                .HasColumnName("UserId")
                .HasColumnType("int");
            builder.Property(s => s.Username)
                .IsRequired()
                .HasMaxLength(20);
            builder.Property(s => s.Email)
                .IsRequired()
                .HasMaxLength(100);
            builder.Property(s => s.PasswordHash)
                .IsRequired()
                .HasMaxLength(100);
            builder.Property(s => s.CreatedAt).HasConversion(
                v => v,
                v => DateTime.SpecifyKind(v, DateTimeKind.Utc))
                .HasColumnType("datetime")
                .HasDefaultValueSql("GETDATE()")
                .ValueGeneratedOnAdd()
                .HasColumnName("CreatedAt");
        }
    }
}
