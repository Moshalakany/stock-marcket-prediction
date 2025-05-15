using StockMarket.DTOs;
using StockMarket.Entities.SQL;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace StockMarket.Services.Interfaces
{
    public interface IPortfolioService
    {
        Task<Portfolio?> GetPortfolioAsync(int userId);
        Task<PortfolioItemDto?> AddStockToPortfolioAsync(int userId, PortfolioItemDto portfolioItemDto);
        Task<bool> RemoveStockFromPortfolioAsync(int userId, string symbol);
    }
}
