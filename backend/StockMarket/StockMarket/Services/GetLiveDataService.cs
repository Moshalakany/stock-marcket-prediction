using Newtonsoft.Json;
using StockMarket.DTOs;
using StockMarket.Services.Interfaces;
using System;
using System.Net.Http;
using System.Threading.Tasks;

namespace StockMarket.Services
{
    public class GetLiveDataService: IGetLiveDataService
    {
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
    }
}
