using Microsoft.Extensions.Hosting;
using StockMarket.Services.Interfaces;

namespace StockMarket.Services
{
    public class NewsConsumerHostedService : IHostedService
    {
        private readonly IRabbitMQNewsConsumer _consumer;
        private readonly ILogger<NewsConsumerHostedService> _logger;

        public NewsConsumerHostedService(
            IRabbitMQNewsConsumer consumer,
            ILogger<NewsConsumerHostedService> logger)
        {
            _consumer = consumer;
            _logger = logger;
        }

        public Task StartAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting RabbitMQ news consumer");
            _consumer.StartConsuming();
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Stopping RabbitMQ news consumer");
            _consumer.StopConsuming();
            return Task.CompletedTask;
        }
    }
}
