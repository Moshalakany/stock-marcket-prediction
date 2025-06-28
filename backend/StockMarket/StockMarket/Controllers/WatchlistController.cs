using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using StockMarket.DTOs;
using StockMarket.Services.Interfaces;
using System.Security.Claims;

namespace StockMarket.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class WatchlistController : ControllerBase
    {
        private readonly IWatchlistService _watchlistService;

        public WatchlistController(IWatchlistService watchlistService)
        {
            _watchlistService = watchlistService;
        }

        [HttpGet]
        public async Task<IActionResult> GetWatchlist()
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            var watchlist = await _watchlistService.GetWatchlistAsync(userId.Value);
            if (watchlist == null)
                return NotFound("Watchlist not found.");

            return Ok(watchlist);
        }

        [HttpPost]
        public async Task<IActionResult> CreateWatchlist([FromBody] WatchlistDto watchlistDto)
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            var watchlist = await _watchlistService.CreateWatchlistAsync(userId.Value, watchlistDto.Name ?? "My Watchlist");
            if (watchlist == null)
                return BadRequest("Failed to create watchlist.");

            return Ok(watchlist);
        }

        [HttpPost("add")]
        public async Task<IActionResult> AddStockToWatchlist([FromBody] WatchlistStockDto stockDto)
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            bool result = await _watchlistService.AddStockToWatchlistAsync(userId.Value, stockDto.Symbol);
            if (!result)
                return BadRequest("Failed to add stock to watchlist.");

            return Ok(new { message = "Stock added to watchlist successfully." });
        }

        [HttpDelete("remove/{symbol}")]
        public async Task<IActionResult> RemoveStockFromWatchlist(string symbol)
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            bool result = await _watchlistService.RemoveStockFromWatchlistAsync(userId.Value, symbol);
            if (!result)
                return BadRequest("Failed to remove stock from watchlist.");

            return Ok(new { message = "Stock removed from watchlist successfully." });
        }

        [HttpDelete]
        public async Task<IActionResult> DeleteWatchlist()
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            bool result = await _watchlistService.DeleteWatchlistAsync(userId.Value);
            if (!result)
                return BadRequest("Failed to delete watchlist.");

            return Ok(new { message = "Watchlist deleted successfully." });
        }

        [HttpGet("stocks")]
        public async Task<IActionResult> GetWatchlistStocks()
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            var stocks = await _watchlistService.GetWatchlistStocksAsync(userId.Value);
            if (stocks == null)
                return NotFound("Watchlist not found.");

            return Ok(stocks);
        }

        [HttpPut("name")]
        public async Task<IActionResult> UpdateWatchlistName([FromBody] WatchlistDto watchlistDto)
        {
            var userId = GetUserId();
            if (userId == null) return Unauthorized();

            if (string.IsNullOrEmpty(watchlistDto.Name))
                return BadRequest("Watchlist name cannot be empty.");

            bool result = await _watchlistService.UpdateWatchlistNameAsync(userId.Value, watchlistDto.Name);
            if (!result)
                return BadRequest("Failed to update watchlist name.");

            return Ok(new { message = "Watchlist name updated successfully." });
        }

        private int? GetUserId()
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier);
            if (userIdClaim != null && int.TryParse(userIdClaim.Value, out int userId))
                return userId;

            return null;
        }
    }
}
