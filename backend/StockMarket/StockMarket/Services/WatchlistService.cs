using Microsoft.EntityFrameworkCore;
using StockMarket.Data;
using StockMarket.Entities.SQL;
using StockMarket.Services.Interfaces;
namespace StockMarket.Services
{
    public class WatchlistService : IWatchlistService
    {
        private readonly SQLAppDbContext _context;
        private readonly IStockService _stockService;

        public WatchlistService(SQLAppDbContext context, IStockService stockService)
        {
            _context = context;
            _stockService = stockService;
        }

        public async Task<Watchlist?> GetWatchlistAsync(int userId)
        {
            return await _context.Watchlists
                .Include(w => w.WatchlistStocks!)
                    .ThenInclude(ws => ws.Stock)
                .FirstOrDefaultAsync(w => w.UserId == userId);
        }

        public async Task<Watchlist?> CreateWatchlistAsync(int userId, string name)
        {
            // Check if user exists
            var user = await _context.Users.FindAsync(userId);
            if (user == null) return null;

            // Check if user already has a watchlist
            var existingWatchlist = await _context.Watchlists.FirstOrDefaultAsync(w => w.UserId == userId);
            if (existingWatchlist != null) return existingWatchlist;

            // Create new watchlist
            var watchlist = new Watchlist
            {
                UserId = userId,
                Name = name,
                WatchlistStocks = new List<WatchlistStock>()
            };

            _context.Watchlists.Add(watchlist);
            await _context.SaveChangesAsync();
            return watchlist;
        }

        public async Task<bool> AddStockToWatchlistAsync(int userId, string symbol)
        {
            // Check if the stock exists
            var stock = await _stockService.GetStockBySymbolAsync(symbol);
            if (stock == null) return false;

            // Get or create user's watchlist
            var watchlist = await _context.Watchlists
                .Include(w => w.WatchlistStocks)
                .FirstOrDefaultAsync(w => w.UserId == userId);

            if (watchlist == null)
            {
                watchlist = new Watchlist
                {
                    UserId = userId,
                    Name = "My Watchlist",
                    WatchlistStocks = new List<WatchlistStock>()
                };
                _context.Watchlists.Add(watchlist);
            }

            // Check if stock is already in watchlist
            if (watchlist.WatchlistStocks?.Any(ws => ws.StockSymbol == symbol) == true)
                return true; // Stock already in watchlist

            // Add stock to watchlist
            var watchlistStock = new WatchlistStock
            {
                WatchlistId = watchlist.WatchlistId,
                StockSymbol = symbol,
                AddedAt = DateTime.UtcNow
            };

            watchlist.WatchlistStocks?.Add(watchlistStock);
            await _context.SaveChangesAsync();
            return true;
        }

        public async Task<bool> RemoveStockFromWatchlistAsync(int userId, string symbol)
        {
            var watchlist = await _context.Watchlists
                .Include(w => w.WatchlistStocks)
                .FirstOrDefaultAsync(w => w.UserId == userId);

            if (watchlist == null || watchlist.WatchlistStocks == null)
                return false;

            var stockToRemove = watchlist.WatchlistStocks.FirstOrDefault(ws => ws.StockSymbol == symbol);
            if (stockToRemove == null)
                return false;

            _context.WatchlistStocks.Remove(stockToRemove);
            await _context.SaveChangesAsync();
            return true;
        }

        public async Task<bool> DeleteWatchlistAsync(int userId)
        {
            var watchlist = await _context.Watchlists.FirstOrDefaultAsync(w => w.UserId == userId);
            if (watchlist == null)
                return false;

            _context.Watchlists.Remove(watchlist);
            await _context.SaveChangesAsync();
            return true;
        }

        public async Task<IEnumerable<Stock>?> GetWatchlistStocksAsync(int userId)
        {
            var watchlist = await _context.Watchlists
                .Include(w => w.WatchlistStocks!)
                .ThenInclude(ws => ws.Stock)
                .FirstOrDefaultAsync(w => w.UserId == userId);

            return watchlist?.WatchlistStocks?.Select(ws => ws.Stock!).ToList();
        }

        public async Task<bool> UpdateWatchlistNameAsync(int userId, string name)
        {
            var watchlist = await _context.Watchlists.FirstOrDefaultAsync(w => w.UserId == userId);
            if (watchlist == null)
                return false;

            watchlist.Name = name;
            await _context.SaveChangesAsync();
            return true;
        }
    }
}
