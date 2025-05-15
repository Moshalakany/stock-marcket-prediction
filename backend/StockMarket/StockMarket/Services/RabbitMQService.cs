using System.Text;
using System.Text.Json;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using StockMarket.Services.Interfaces;

namespace StockMarket.Services
{
    public class RabbitMQService : IDisposable, IRabbitMQService
    {
        private readonly IConnection _connection;
        private readonly IModel _channel;
        private readonly string _requestQueueName = "ai_request_queue";
        private readonly string _responseQueueName = "ai_response_queue";

        public RabbitMQService(IConfiguration configuration)
        {
            ConnectionFactory factory = new ConnectionFactory
            {
                HostName = configuration["RabbitMQ:HostName"] ?? "localhost",
                Port = int.Parse(configuration["RabbitMQ:Port"] ?? "5672"),
                UserName = configuration["RabbitMQ:UserName"] ?? "guest",
                Password = configuration["RabbitMQ:Password"] ?? "guest"
            };
            _connection = factory.CreateConnection();
            _channel = _connection.CreateModel();

            _channel.QueueDeclare(queue: _requestQueueName, durable: true, exclusive: false, autoDelete: false);
            _channel.QueueDeclare(queue: _responseQueueName, durable: true, exclusive: false, autoDelete: false);
        }

        public void SendMessage<T>(T message, string routingKey = "")
        {
            var messageJson = JsonSerializer.Serialize(message);
            var body = Encoding.UTF8.GetBytes(messageJson);

            _channel.BasicPublish(exchange: "",
                routingKey: _requestQueueName,
                basicProperties: null,
                body: body);
        }

        public void StartListening(Func<string, Task> onMessageReceived)
        {
            var consumer = new EventingBasicConsumer(_channel);
            consumer.Received += async (model, ea) =>
            {
                var body = ea.Body.ToArray();
                var message = Encoding.UTF8.GetString(body);

                await onMessageReceived(message);

                _channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
            };

            _channel.BasicConsume(queue: _responseQueueName, autoAck: false, consumer: consumer);
        }

        public void Dispose()
        {
            _channel?.Dispose();
            _connection?.Dispose();
        }
    }
}