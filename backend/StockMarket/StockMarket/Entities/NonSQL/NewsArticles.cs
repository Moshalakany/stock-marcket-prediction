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
        public string symbol { get; set; }
        
        public DateTime datetime { get; set; }
        
        public string title { get; set; }
        
        public string source { get; set; }
        
        public string link { get; set; }
        
    }
}
