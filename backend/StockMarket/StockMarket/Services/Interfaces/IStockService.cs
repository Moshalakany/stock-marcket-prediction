using StockMarket.DTOs;
using StockMarket.Entities.SQL;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace StockMarket.Services.Interfaces
{
    public interface IStockService
    {
        Task<IEnumerable<Stock>> GetAllStocksAsync();
        Task<Stock?> GetStockBySymbolAsync(string symbol);
        Task<Stock?> CreateStockAsync(StockDto stockDto);
        Task<Stock?> UpdateStockAsync(string symbol, StockDto stockDto);
        Task<bool> DeleteStockAsync(string symbol);
    }
}
