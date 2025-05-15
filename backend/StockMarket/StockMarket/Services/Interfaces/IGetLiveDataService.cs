using System.ComponentModel.DataAnnotations;

namespace StockMarket.Services.Interfaces
{
    public interface IGetLiveDataService
    {
        public Task<decimal> GetCurrentPriceAsync(string symbol);
    }
}
