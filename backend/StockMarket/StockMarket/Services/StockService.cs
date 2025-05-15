using Microsoft.EntityFrameworkCore;
using StockMarket.Data;
using StockMarket.DTOs;
using StockMarket.Entities.SQL;
using StockMarket.Services.Interfaces;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace StockMarket.Services
{
    public class StockService(SQLAppDbContext context) : IStockService
    {
        public async Task<IEnumerable<Stock>> GetAllStocksAsync()
        {
            return await context.Stocks.ToListAsync();
        }

        public async Task<Stock?> GetStockBySymbolAsync(string symbol)
        {
            return await context.Stocks.FirstOrDefaultAsync(s => s.Symbol == symbol);
        }

        public async Task<Stock?> CreateStockAsync(StockDto stockDto)
        {
            // Check if a stock with the same symbol already exists
            if (stockDto.Symbol == null || await context.Stocks.AnyAsync(s => s.Symbol == stockDto.Symbol))
            {
                return null; // Symbol already exists or is null, cannot create stock
            }

            var stock = new Stock
            {
                Symbol = stockDto.Symbol,
                CompanyName = stockDto.CompanyName,
                Sector = stockDto.Sector
            };

            context.Stocks.Add(stock);
            await context.SaveChangesAsync();
            return stock;
        }

        public async Task<Stock?> UpdateStockAsync(string symbol, StockDto stockDto)
        {
            var stock = await context.Stocks.FirstOrDefaultAsync(s => s.Symbol == symbol);
            if (stock == null)
            {
                return null;
            }

            stock.CompanyName = stockDto.CompanyName;
            stock.Sector = stockDto.Sector;

            await context.SaveChangesAsync();
            return stock;
        }
        public async Task<decimal> GetCurrentPriceAsync(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
            }

            var requestURI = $"https://coincodex.com/stonks_api/get_quote/{symbol}";

            try
            {
                using (var client = new HttpClient())
                {
                    var response = await client.GetAsync(requestURI);
                    response.EnsureSuccessStatusCode();

                    var content = await response.Content.ReadAsStringAsync();

                    try
                    {
                        var jsonResponse = JsonConvert.DeserializeObject<CurrentLivePriceDto>(content);

                        if (jsonResponse == null)
                        {
                            throw new InvalidOperationException("Received null response from API");
                        }

                        return jsonResponse.latestPrice;
                    }
                    catch (JsonException ex)
                    {
                        throw new FormatException($"Failed to deserialize API response: {ex.Message}", ex);
                    }
                }
            }
            catch (HttpRequestException ex)
            {
                throw new Exception($"API request failed for symbol {symbol}: {ex.Message}", ex);
            }
            catch (Exception ex) when (ex is not ArgumentException && ex is not FormatException)
            {
                throw new Exception($"Unexpected error getting price for {symbol}: {ex.Message}", ex);
            }
        }

        public async Task<bool> DeleteStockAsync(string symbol)
        {
            var stock = await context.Stocks.FirstOrDefaultAsync(s => s.Symbol == symbol);
            if (stock == null)
            {
                return false;
            }

            context.Stocks.Remove(stock);
            await context.SaveChangesAsync();
            return true;
        }
    }
}
