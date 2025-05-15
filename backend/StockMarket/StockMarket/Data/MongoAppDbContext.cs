using MongoDB.Driver;
using StockMarket.Entities.NonSQL;
using Microsoft.Extensions.Configuration;

namespace StockMarket.Data
{
    public class MongoAppDbContext
    {
        private readonly IMongoDatabase _database;
        private readonly MongoClient _client;

        public MongoAppDbContext(IConfiguration configuration)
        {
            // Get MongoDB configuration from appsettings.json
            var connectionString = configuration.GetConnectionString("MongoDbConnection") ?? "mongodb://localhost:27017";
            var databaseName = configuration.GetConnectionString("MongoDbName") ?? "StockMarket";
            
            // Create MongoDB client and get database
            _client = new MongoClient(connectionString);
            _database = _client.GetDatabase(databaseName);
            
            // Ensure collections exist and indexes are created
            ConfigureCollections();
        }

        // Collections
        public IMongoCollection<NewsArticle> NewsArticles => _database.GetCollection<NewsArticle>("NewsArticles");

        private void ConfigureCollections()
        {
            // Create indexes for NewsArticles collection
            var newsArticlesCollection = _database.GetCollection<NewsArticle>("NewsArticles");
            
            // Create index on Symbol field for faster lookups
            var symbolIndexKeys = Builders<NewsArticle>.IndexKeys.Ascending(a => a.Symbol);
            var symbolIndexModel = new CreateIndexModel<NewsArticle>(symbolIndexKeys);
            newsArticlesCollection.Indexes.CreateOne(symbolIndexModel);
            
            // Create compound index on Symbol and Title for uniqueness checks
            var symbolTitleIndexKeys = Builders<NewsArticle>.IndexKeys
                .Ascending(a => a.Symbol)
                .Ascending(a => a.Title);
            var symbolTitleIndexModel = new CreateIndexModel<NewsArticle>(
                symbolTitleIndexKeys, 
                new CreateIndexOptions { Unique = true });
            
            try
            {
                newsArticlesCollection.Indexes.CreateOne(symbolTitleIndexModel);
            }
            catch
            {
                // Index might already exist, ignore error
            }
            
            // Create index on DateTime field for sorting
            var dateTimeIndexKeys = Builders<NewsArticle>.IndexKeys.Descending(a => a.DateTime);
            var dateTimeIndexModel = new CreateIndexModel<NewsArticle>(dateTimeIndexKeys);
            newsArticlesCollection.Indexes.CreateOne(dateTimeIndexModel);
        }
    }
}
