namespace StockMarket.Services.Interfaces
{
    public interface IRabbitMQNewsConsumer
    {
        void StartConsuming();
        void StopConsuming();
    }
}
