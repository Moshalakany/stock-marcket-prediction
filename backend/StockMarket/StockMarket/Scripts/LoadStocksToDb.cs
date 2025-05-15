using StockMarket.Data;
using StockMarket.Entities.SQL;

namespace StockMarket.Scripts
{
    public class LoadStocksToDb(SQLAppDbContext context)
    {
        string csvFilePath = "E:\\Stock Market GP\\Current\\stock-marcket-prediction\\backend\\StockMarket\\StockMarket\\Scripts\\sp500_companies.csv";
        public void LoadStocks()
        {
            var stocks = new List<Stock>();
            using (var reader = new StreamReader(csvFilePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var values = line.Split(',');
                    if (values.Length < 3) continue; // Skip invalid lines
                    try
                    {
                        var stock = new Stock
                        {
                            //Exchange,Symbol,Shortname,Longname,Sector,Industry,Currentprice,Marketcap,Ebitda,Revenuegrowth,City,State,Country,Fulltimeemployees,Longbusinesssummary,Weight

                            symbol = values[1],
                            CompanyName = values[3],
                            Sector = values[4],
                            Industry = values[5],
                            FullTimeEmployees = Convert.ToInt32(values[13]),
                            Longbusinesssummary = values[14],
                        };

                        if (stock.symbol == "AAPL") 
                        {
                            continue;
                        }
                        stocks.Add(stock);
                    }
                    catch(Exception e) 
                    {
                        Console.WriteLine(e.Message);
                        continue;
                    }
                }
            }
            // Save to SQL database
            using (context)
            {
                context.Stocks.AddRange(stocks);
                context.SaveChanges();
            }
        }
    }
}
