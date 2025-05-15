using Microsoft.AspNetCore.Mvc;
using StockMarket.DTOs;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using StockMarket.Services.Interfaces;

namespace StockMarket.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class StockController(IStockService stockService) : ControllerBase
    {
        [HttpGet]
        public async Task<IActionResult> GetAllStocks()
        {
            var stocks = await stockService.GetAllStocksAsync();
            return Ok(stocks);
        }

        [HttpGet("{symbol}")]
        public async Task<IActionResult> GetStockBySymbol(string symbol)
        {
            var stock = await stockService.GetStockBySymbolAsync(symbol);
            if (stock == null)
            {
                return NotFound();
            }
            return Ok(stock);
        }

        [HttpPost]
        [Authorize(Roles = "Admin")] 
        public async Task<IActionResult> CreateStock([FromBody] StockDto stockDto)
        {
            var stock = await stockService.CreateStockAsync(stockDto);
            if (stock == null)
            {
                return BadRequest("Could not create stock.");
            }
            return CreatedAtAction(nameof(GetStockBySymbol), new { symbol = stock.Symbol }, stock);
        }

        [HttpPut("{symbol}")]
        [Authorize(Roles = "Admin")] 
        public async Task<IActionResult> UpdateStock(string symbol, [FromBody] StockDto stockDto)
        {
            var stock = await stockService.UpdateStockAsync(symbol, stockDto);
            if (stock == null)
            {
                return NotFound();
            }
            return Ok(stock);
        }

        [HttpDelete("{symbol}")]
        [Authorize(Roles = "Admin")] 
        public async Task<IActionResult> DeleteStock(string symbol)
        {
            var result = await stockService.DeleteStockAsync(symbol);
            if (!result)
            {
                return NotFound();
            }
            return NoContent();
        }
    }
}
