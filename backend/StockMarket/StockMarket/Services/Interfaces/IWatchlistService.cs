using StockMarket.DTOs;
using StockMarket.Entities.SQL;

namespace StockMarket.Services.Interfaces
{
    public interface IWatchlistService
    {
        Task<Watchlist?> GetWatchlistAsync(int userId);
        Task<Watchlist?> CreateWatchlistAsync(int userId, string name);
        Task<bool> AddStockToWatchlistAsync(int userId, string symbol);
        Task<bool> RemoveStockFromWatchlistAsync(int userId, string symbol);
        Task<bool> DeleteWatchlistAsync(int userId);
        Task<IEnumerable<Stock>?> GetWatchlistStocksAsync(int userId);
        Task<bool> UpdateWatchlistNameAsync(int userId, string name);
    }
}
