namespace StockMarket.Services.Interfaces
{
    public interface IRabbitMQService
    {
        void SendMessage<T>(T message, string routingKey = "");
        void StartListening(Func<string, Task> onMessageReceived);
    }
}