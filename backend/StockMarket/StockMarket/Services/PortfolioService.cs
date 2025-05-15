using Microsoft.EntityFrameworkCore;
using StockMarket.Data;
using StockMarket.DTOs;
using StockMarket.Entities.SQL;
using StockMarket.Services.Interfaces;
using System.Linq;
using System.Threading.Tasks;

namespace StockMarket.Services
{
    public class PortfolioService(SQLAppDbContext context, IStockService stockService,IGetLiveDataService liveDataService) : IPortfolioService
    {
        public async Task<Portfolio?> GetPortfolioAsync(int userId)
        {
            var portfolio = await context.Portfolios
                .Include(p => p.PortfolioStocks!)
                .ThenInclude(ps => ps.Stock)
                .FirstOrDefaultAsync(p => p.UserId == userId);

            if (portfolio != null)
            {
                if (portfolio.PortfolioStocks != null && portfolio.PortfolioStocks.Any())
                {
                    // PortfolioStock.TotalValue is a calculated property (Quantity * CurrentPrice).
                    // This relies on PortfolioStock.CurrentPrice being correctly populated/updated.
                    // For the purpose of GetPortfolioAsync, we sum the existing TotalValues.
                    portfolio.TotalValue = portfolio.PortfolioStocks.Sum(ps => ps.TotalValue);
                }
                else
                {
                    portfolio.TotalValue = 0; // If no stocks, total value is 0.
                }
            }

            return portfolio;
        }

        public async Task<PortfolioItemDto?> AddStockToPortfolioAsync(int userId, PortfolioItemDto portfolioItemDto)
        {
            
           
            var user = await context.Users.FindAsync(userId);
            if (user == null) return null; // Or throw an exception
            var stock = await stockService.GetStockBySymbolAsync(portfolioItemDto.Symbol);
            if (stock == null)
            {
                // Or handle stock not found, maybe return a specific error DTO or throw
                return null;
            }

            var portfolio = await context.Portfolios
                .Include(p => p.PortfolioStocks!)
                .ThenInclude(ps => ps.Stock)
                .FirstOrDefaultAsync(p => p.UserId == userId);
            if (portfolio == null)
            {
                portfolio = new Portfolio { UserId = userId, PortfolioStocks = new List<PortfolioStock>() };
                context.Portfolios.Add(portfolio);
            }

            var portfolioStock = portfolio.PortfolioStocks?.FirstOrDefault(ps => ps.StockSymbol == stock.Symbol);            
            if (portfolioStock != null)
            {
                var totalQuantity = portfolioStock.Quantity + portfolioItemDto.Quantity;
                portfolioStock.Quantity = totalQuantity;
            }
            else
            {
                var currentPrice = await liveDataService.GetCurrentPriceAsync(stock.Symbol);
                portfolioStock = new PortfolioStock
                {
                    Portfolio = portfolio,
                    StockSymbol = stock.Symbol,
                    Stock = stock, 
                    Quantity = portfolioItemDto.Quantity,
                    PurchaseDate = DateTime.UtcNow,  
                    CurrentPrice = currentPrice,
                    PurchasePrice = currentPrice
                };
                portfolio.PortfolioStocks?.Add(portfolioStock);
            }

            await context.SaveChangesAsync();
            return new PortfolioItemDto
            {
                Symbol = stock.Symbol,
                Quantity = portfolioItemDto.Quantity,
                PurchasePrice = portfolioItemDto.PurchasePrice 
            };
        }

        public async Task<bool> RemoveStockFromPortfolioAsync(int userId, string symbol)
        {
            var portfolio = await context.Portfolios
                .Include(p => p.PortfolioStocks!)
                .ThenInclude(ps => ps.Stock)
                .FirstOrDefaultAsync(p => p.UserId == userId);

            if (portfolio == null || portfolio.PortfolioStocks == null) return false;

            var stockToRemove = portfolio.PortfolioStocks.FirstOrDefault(ps => ps.Stock != null && ps.Stock.Symbol == symbol);

            if (stockToRemove == null) return false;

            context.PortfolioStocks.Remove(stockToRemove);
            await context.SaveChangesAsync();
            return true;
        }
    }
}
