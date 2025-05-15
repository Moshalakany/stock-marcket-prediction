using System.ComponentModel.DataAnnotations;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace StockMarket.Entities.NonSQL
{
    public class NewsArticle
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string Id { get; set; }
        
        [Required]
        public string Symbol { get; set; }
        
        public DateTime DateTime { get; set; }
        
        public string Title { get; set; }
        
        public string Source { get; set; }
        
        public string Link { get; set; }
        
    }
}
