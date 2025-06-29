using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.IdentityModel.Tokens;
using StockMarket.Data;
using StockMarket.Scripts;
using StockMarket.Services;
using StockMarket.Services.Interfaces;
using System.Security.Claims;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.ReferenceHandler = System.Text.Json.Serialization.ReferenceHandler.Preserve;
    });
builder.Services.AddHttpClient();

// Get connection strings with environment variable support
var sqlConnectionString = Environment.GetEnvironmentVariable("SQL_CONNECTION_STRING") 
    ?? builder.Configuration.GetConnectionString("SqlDbConnection");
var mongoConnectionString = Environment.GetEnvironmentVariable("MONGO_CONNECTION_STRING") 
    ?? builder.Configuration.GetConnectionString("MongoDbConnection");

// Configure SQL database
builder.Services.AddDbContextPool<SQLAppDbContext>(options => {
    options.UseSqlServer(sqlConnectionString);
});

// Configure MongoDB
builder.Services.AddSingleton(provider => {
    var mongoDbName = Environment.GetEnvironmentVariable("MONGO_DB_NAME") 
        ?? builder.Configuration["ConnectionStrings:MongoDbName"] 
        ?? "StockMarket";
    return new MongoAppDbContext(mongoConnectionString, mongoDbName);
});

// JWT authentication configuration
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidIssuer = Environment.GetEnvironmentVariable("JWT_ISSUER") ?? builder.Configuration["AppSettings:Issuer"],
            ValidateAudience = true,
            ValidAudience = Environment.GetEnvironmentVariable("JWT_AUDIENCE") ?? builder.Configuration["AppSettings:Audience"],
            ValidateLifetime = true, 
            IssuerSigningKey = new SymmetricSecurityKey(
                Encoding.UTF8.GetBytes(Environment.GetEnvironmentVariable("JWT_TOKEN") ?? builder.Configuration["AppSettings:Token"]!)),
            ValidateIssuerSigningKey = true
        };

        options.Events = new JwtBearerEvents
        {
            OnTokenValidated = async context =>
            {
                var userIdClaim = context.Principal?.FindFirstValue(ClaimTypes.NameIdentifier);
                if (string.IsNullOrEmpty(userIdClaim) || !int.TryParse(userIdClaim, out var userId))
                {
                    context.Fail("Invalid token: User ID claim is missing or invalid.");
                    return;
                }

                using (var scope = context.HttpContext.RequestServices.CreateScope())
                {
                    var dbContext = scope.ServiceProvider.GetRequiredService<SQLAppDbContext>();
                    var user = await dbContext.Users.FindAsync(userId);

                    if (user == null || user.RefreshToken == null || user.RefreshTokenExpiryTime <= DateTime.UtcNow)
                    {
                        context.Fail("Token revoked or associated session is invalid.");
                        return;
                    }
                }
            }
        };
    });
//news articles script
//System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
//StockNewsFromExcelToDb stockNewsFromExcelToDb = new StockNewsFromExcelToDb();
//stockNewsFromExcelToDb.SaveToMongoCollection("E:\\Stock Market GP\\Current\\stock-marcket-prediction\\news\\news_data");
//builder.Services.AddSingleton<RabbitMQService>();


builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<IStockService, StockService>();
builder.Services.AddScoped<IPortfolioService, PortfolioService>(); 
builder.Services.AddScoped<IGetLiveDataService, GetLiveDataService>();
builder.Services.AddSingleton<INewsArticleService, NewsArticleService>();
builder.Services.AddScoped<ISentimentService, SentimentService>();

builder.Services.AddScoped<IWatchlistService, WatchlistService>();

//// Register NewsArticle-related services
//builder.Services.AddScoped<INewsArticleService, NewsArticleService>();
//builder.Services.AddSingleton<IRabbitMQNewsConsumer, RabbitMQNewsConsumer>();
//builder.Services.AddSingleton<IRabbitMQService, RabbitMQService>();

//// Optional: Add a Hosted Service to start consuming automatically
//builder.Services.AddHostedService<NewsConsumerHostedService>();

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAngularApp",
        policy => policy
            .WithOrigins("http://localhost:4200") // Your Angular URL
            .AllowAnyMethod()
            .AllowAnyHeader());
});


var app = builder.Build();
app.UseCors("AllowAngularApp");
// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment() || Environment.GetEnvironmentVariable("ENABLE_SWAGGER")?.ToLower() == "true")
{

    app.MapOpenApi();
    app.MapSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "StockMarket API V1");
        c.RoutePrefix = string.Empty; 
    });
}

app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();

await app.RunAsync();
