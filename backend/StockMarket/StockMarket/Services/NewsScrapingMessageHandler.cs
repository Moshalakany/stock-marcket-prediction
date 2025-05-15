//using System.Text.Json;
//using StockMarket.DTOs;
//using StockMarket.Services.Interfaces;

//namespace StockMarket.Services
//{
//    public class NewsScrapingMessageHandler : IHostedService
//    {
//        private readonly IRabbitMQService _rabbitMQService;
//        private readonly IServiceProvider _serviceProvider;
//        private readonly ILogger<NewsScrapingMessageHandler> _logger;

//        public NewsScrapingMessageHandler(
//            IRabbitMQService rabbitMQService,
//            IServiceProvider serviceProvider,
//            ILogger<NewsScrapingMessageHandler> logger)
//        {
//            _rabbitMQService = rabbitMQService;
//            _serviceProvider = serviceProvider;
//            _logger = logger;
//        }

//        public Task StartAsync(CancellationToken cancellationToken)
//        {
//            _logger.LogInformation("Starting news scraping message handler");
            
//            if (_rabbitMQService == null)
//            {
//                _logger.LogError("RabbitMQ service is not initialized");
//                return Task.CompletedTask;
//            }
            
//            _rabbitMQService.StartListening(async (message) =>
//            {
//                try
//                {
//                    _logger.LogInformation($"Received news scraping response: {message}");
                    
//                    // Try to deserialize as a list of news articles
//                    var articles = JsonSerializer.Deserialize<List<NewsArticleCreateDto>>(message);
                    
//                    if (articles != null && articles.Any())
//                    {
//                        using (var scope = _serviceProvider.CreateScope())
//                        {
//                            var newsService = scope.ServiceProvider.GetRequiredService<INewsArticleService>();
//                            await newsService.ProcessScrapedNewsAsync(articles);
//                        }
//                    }
//                }
//                catch (Exception ex)
//                {
//                    _logger.LogError($"Error processing news scraping message: {ex.Message}");
//                }
//            });
            
//            return Task.CompletedTask;
//        }

//        public Task StopAsync(CancellationToken cancellationToken)
//        {
//            _logger.LogInformation("Stopping news scraping message handler");
//            return Task.CompletedTask;
//        }
//    }
//}
