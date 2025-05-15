using Microsoft.EntityFrameworkCore;
using Microsoft.Identity.Client;
using StockMarket.Data.config;

namespace StockMarket.Data
{
    public class SQLAppDbContext : DbContext
    {
        public DbSet<Entities.SQL.User> Users { get; set; }
        public DbSet<Entities.SQL.Stock> Stocks { get; set; }
        public DbSet<Entities.SQL.Watchlist> Watchlists { get; set; }

        public DbSet<Entities.SQL.WatchlistStock> WatchlistStocks { get; set; }
        public DbSet<Entities.SQL.Portfolio> Portfolios { get; set; }
        public DbSet<Entities.SQL.PortfolioStock> PortfolioStocks { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder) 
        {
            base.OnModelCreating(modelBuilder);
            modelBuilder.ApplyConfigurationsFromAssembly(typeof(UserConfiguration).Assembly);
        }
        public SQLAppDbContext(DbContextOptions options) : base(options)
        {

        }
       

    }
}
