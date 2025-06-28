using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using StockMarket.DTOs.Sentiment;
using StockMarket.Services.Interfaces;

namespace StockMarket.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SentimentController : ControllerBase
    {
        private readonly ISentimentService _sentimentService;
        
        public SentimentController(ISentimentService sentimentService)
        {
            _sentimentService = sentimentService;
        }
        
        [HttpGet("health")]
        public async Task<ActionResult<SentimentHealthResponseDto>> CheckHealth()
        {
            try
            {
                var result = await _sentimentService.CheckHealthAsync();
                return Ok(result);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Failed to check sentiment service health: {ex.Message}");
            }
        }
        
        [HttpPost("analyze")]
        public async Task<ActionResult<SentimentAnalysisResponseDto>> AnalyzeSentiment([FromBody] SentimentAnalysisRequestDto request)
        {
            if (string.IsNullOrEmpty(request.text))
            {
                return BadRequest("Text cannot be empty");
            }
            
            try
            {
                var result = await _sentimentService.AnalyzeSentimentAsync(request);
                return Ok(result);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Sentiment analysis failed: {ex.Message}");
            }
        }
        
        [HttpPost("batch")]
        public async Task<ActionResult<BatchSentimentAnalysisResponseDto>> AnalyzeBatchSentiment([FromBody] BatchSentimentAnalysisRequestDto request)
        {
            if (request.texts == null || !request.texts.Any())
            {
                return BadRequest("No texts provided for analysis");
            }
            
            try
            {
                var result = await _sentimentService.AnalyzeBatchSentimentAsync(request);
                return Ok(result);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Batch sentiment analysis failed: {ex.Message}");
            }
        }
        
        [HttpGet("stock/{symbol}")]
        public async Task<ActionResult<BatchSentimentAnalysisResponseDto>> AnalyzeStockNewsSentiment(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                return BadRequest("Symbol cannot be empty");
            }
            
            try
            {
                var result = await _sentimentService.AnalyzeNewsSentimentAsync(symbol);
                return Ok(result);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"News sentiment analysis failed: {ex.Message}");
            }
        }
    }
}
