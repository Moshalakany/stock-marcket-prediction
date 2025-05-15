// using System.Text;
// using System.Text.Json;
// using RabbitMQ.Client;
// using RabbitMQ.Client.Events;
// using StockMarket.DTOs;
// using StockMarket.Services.Interfaces;

// namespace StockMarket.Services
// {
//     public class RabbitMQNewsConsumer : IRabbitMQNewsConsumer, IDisposable
//     {
//         private readonly IConnection _connection;
//         private readonly IModel _channel;
//         private readonly string _responseQueueName = "news_scraping_response";
//         private readonly IServiceProvider _serviceProvider; // Replace direct injection with IServiceProvider
//         private readonly ILogger<RabbitMQNewsConsumer> _logger;
//         private bool _isConsuming;

//         public RabbitMQNewsConsumer(
//             IConfiguration configuration,
//             IServiceProvider serviceProvider, 
//             ILogger<RabbitMQNewsConsumer> logger)
//         {
//             ConnectionFactory factory = new ConnectionFactory
//             {
//                 HostName = configuration["RabbitMQ:HostName"] ?? "localhost",
//                 Port = int.Parse(configuration["RabbitMQ:Port"] ?? "5672"),
//                 UserName = configuration["RabbitMQ:UserName"] ?? "guest",
//                 Password = configuration["RabbitMQ:Password"] ?? "guest"
//             };
            
//             _connection = factory.CreateConnection();
//             _channel = _connection.CreateModel();
//             _serviceProvider = serviceProvider; // Store the service provider
//             _logger = logger;

//             _channel.QueueDeclare(queue: _responseQueueName, durable: true, exclusive: false, autoDelete: false);
//         }

//         public void StartConsuming()
//         {
//             if (_isConsuming)
//                 return;

//             _isConsuming = true;
//             var consumer = new EventingBasicConsumer(_channel);
            
//             consumer.Received += async (model, ea) =>
//             {
//                 try
//                 {
//                     var body = ea.Body.ToArray();
//                     var message = Encoding.UTF8.GetString(body);
//                     _logger.LogInformation($"Received news scraping response: {message}");

//                     var articles = JsonSerializer.Deserialize<List<NewsArticleCreateDto>>(message);
                    
//                     if (articles != null && articles.Any())
//                     {
//                         using (var scope = _serviceProvider.CreateScope())
//                         {
//                             var newsArticleService = scope.ServiceProvider.GetRequiredService<INewsArticleService>();
//                             await newsArticleService.ProcessScrapedNewsAsync(articles);
//                         }
                        
//                         _logger.LogInformation($"Processed {articles.Count} news articles");
//                     }
                    
//                     _channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
//                 }
//                 catch (Exception ex)
//                 {
//                     _logger.LogError($"Error processing news message: {ex.Message}");
//                     _channel.BasicNack(deliveryTag: ea.DeliveryTag, multiple: false, requeue: false);
//                 }
//             };

//             _channel.BasicConsume(queue: _responseQueueName, autoAck: false, consumer: consumer);
//             _logger.LogInformation("Started consuming news messages from RabbitMQ");
//         }

//         public void StopConsuming()
//         {
//             _isConsuming = false;
//             _logger.LogInformation("Stopped consuming news messages from RabbitMQ");
//         }

//         public void Dispose()
//         {
//             _channel?.Dispose();
//             _connection?.Dispose();
//         }
//     }
// }
