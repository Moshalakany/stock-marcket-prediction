using ExcelDataReader;
using MongoDB.Driver;
using StockMarket.Data;
using StockMarket.Entities.NonSQL;
using System.Globalization;
using System.Net;

namespace StockMarket.Scripts
{
    public class StockNewsFromExcelToDb
    {
        private IEnumerable<NewsArticle> loadAndParseExcelList(string filePath)
        {
            var newsArticles = new List<NewsArticle>();
            using (var stream = File.Open(filePath, FileMode.Open, FileAccess.Read))
            {
                using (var reader = new StreamReader(stream))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        var values = line.Split(',');
                        if (values.Length < 3) continue; // Skip invalid lines
                        var fileName = Path.GetFileName(filePath);
                        DateTime datetime;
                        if (DateTime.TryParseExact(values[1],
                                                   "M/d/yyyy h:mm:ss tt",
                                                   CultureInfo.InvariantCulture,
                                                   DateTimeStyles.None,
                                                   out datetime))
                        {
                            // parsed successfully
                        }
                        else
                        {
                            datetime = DateTime.MinValue;
                        }

                        var newsArticle = new NewsArticle
                        {

                            symbol = fileName,
                            // date in excel in this format: 4/25/2018 2:30:00 PM
                            // convert to DateTime in this format: 2018-04-25T14
                            datetime = datetime,
                            title = values[2],
                            source = values[3],
                            link = values[4]
                        };
                        newsArticles.Add(newsArticle);
                    }
                }
            }
            return newsArticles;
        }
        public void SaveToMongoCollection(string folderPath) 
        {
            var mongoDbContext = new MongoAppDbContext(new ConfigurationBuilder().Build());
            Console.WriteLine(folderPath);
            var files = Directory.GetFiles(folderPath, "*.csv");
            foreach (var file in files)
            {
                var newsArticles = loadAndParseExcelList(file);
                foreach (var newsArticle in newsArticles)
                {
                    try
                    {
                        mongoDbContext.NewsArticles.InsertOne(newsArticle);
                    }
                    catch (MongoWriteException ex) when (ex.WriteError.Category == ServerErrorCategory.DuplicateKey)
                    {
                        Console.WriteLine($"Duplicate key error for {newsArticle.symbol}: {ex.Message}");
                        continue;
                    }
                }
            }
        }
    }
}
