using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using StockMarket.DTOs;
using StockMarket.Services.Interfaces;
using System.Security.Claims;
using System.Threading.Tasks;

namespace StockMarket.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize] // Require authentication for all actions in this controller
    public class PortfolioController(IPortfolioService portfolioService) : ControllerBase
    {
        private int GetUserIdFromClaims()
        {

            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier);
            if (userIdClaim == null || !int.TryParse(userIdClaim.Value, out var userId))
            {
                // This should not happen if [Authorize] is working correctly and token is valid
                throw new UnauthorizedAccessException("User ID not found in token.");
            }
            return userId;
        }

        [HttpGet]
        public async Task<IActionResult> GetPortfolio()
        {
            var userId =GetUserIdFromClaims();
        
            var portfolio = await portfolioService.GetPortfolioAsync(userId);
            if (portfolio == null)
            {
                // Return an empty portfolio or a specific structure if preferred
                return Ok(new { UserId = userId, Stocks = new List<object>() }); 
            }
            return Ok(portfolio);
        }

        [HttpPost("add")]
        public async Task<IActionResult> AddStockToPortfolio([FromBody] PortfolioItemDto portfolioItemDto)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(ModelState);
            }
            var userId = GetUserIdFromClaims();
            var addedStock = await portfolioService.AddStockToPortfolioAsync(userId, portfolioItemDto);
            if (addedStock == null)
            {
                return BadRequest("Could not add stock to portfolio. Ensure stock symbol is valid.");
            }
            // Consider returning the updated portfolio or the specific item added
            return Ok(addedStock); 
        }

        [HttpDelete("remove/{symbol}")]
        public async Task<IActionResult> RemoveStockFromPortfolio(string symbol)
        {
            var userId = GetUserIdFromClaims();
            var result = await portfolioService.RemoveStockFromPortfolioAsync(userId, symbol);
            if (!result)
            {
                return NotFound("Stock not found in portfolio or could not be removed.");
            }
            return NoContent(); // Successfully removed
        }
    }
}
