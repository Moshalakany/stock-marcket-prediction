using Microsoft.AspNetCore.Mvc;
using StockMarket.DTOs;
using StockMarket.Services.Interfaces;

namespace StockMarket.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class NewsArticleController : ControllerBase
    {
        private readonly INewsArticleService _newsArticleService;
        private readonly ILogger<NewsArticleController> _logger;

        public NewsArticleController(
            INewsArticleService newsArticleService,
            ILogger<NewsArticleController> logger)
        {
            _newsArticleService = newsArticleService;
            _logger = logger;
        }

        [HttpGet("{symbol}")]
        public async Task<ActionResult<IEnumerable<NewsArticleDto>>> GetNewsBySymbol(string symbol)
        {
            try
            {
                var news = await _newsArticleService.GetNewsBySymbolAsync(symbol);
                return Ok(news);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error retrieving news for {symbol}: {ex.Message}");
                return StatusCode(500, "Error retrieving news articles");
            }
        }

        [HttpGet("latest/")]
        public async Task<ActionResult<IEnumerable<NewsArticleDto>>> GetLatestNews()
        {
            try
            {
                var news = await _newsArticleService.GetLatestNewsAsync();
                return Ok(news);
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error retrieving latest news: {ex.Message}");
                return StatusCode(500, "Error retrieving latest news articles");
            }
        }





        [HttpGet("range/{symbol}")]
        public async Task<IActionResult> GetNewsByDateRange(string symbol, [FromQuery] DateTime startDate, [FromQuery] DateTime endDate)
        {
            try
            {
                if (startDate == default || endDate == default)
                {
                    return BadRequest("Please provide valid start_date and end_date parameters");
                }
                
                var result = await _newsArticleService.RequestNewsScrapingByDateRangeAsync(symbol, startDate, endDate);
                if (result!=null)
                    return Ok(result);
                else
                    return StatusCode(500, "Failed to request news scraping by date range");
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error requesting news by date range for {symbol}: {ex.Message}");
                return StatusCode(500, "Error requesting news scraping by date range");
            }
        }

        [HttpGet("latest-page/{symbol}")]
        public async Task<IActionResult> GetLatestPageNews(string symbol)
        {
            try
            {
                var result = await _newsArticleService.RequestLatestPageNewsScrapingAsync(symbol);
                if (result != null)
                    return Ok(result);
                else
                    return StatusCode(500, "Failed to request latest page news scraping");
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error requesting latest page news for {symbol}: {ex.Message}");
                return StatusCode(500, "Error requesting latest page news scraping");
            }
        }
    }
}
