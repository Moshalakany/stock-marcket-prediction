using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.IdentityModel.Tokens;
using StockMarket.Data;
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
builder.Services.AddDbContextPool<SQLAppDbContext>(options => {
    options.UseSqlServer(builder.Configuration.GetConnectionString("SqlDbConnection"));
});
builder.Services.AddSingleton<MongoAppDbContext>();
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidIssuer = builder.Configuration["AppSettings:Issuer"],
            ValidateAudience = true,
            ValidAudience = builder.Configuration["AppSettings:Audience"],
            ValidateLifetime = true, 
            IssuerSigningKey = new SymmetricSecurityKey(
                Encoding.UTF8.GetBytes(builder.Configuration["AppSettings:Token"]!)),
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
//builder.Services.AddSingleton<RabbitMQService>();
builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<IStockService, StockService>();
builder.Services.AddScoped<IPortfolioService, PortfolioService>(); 
builder.Services.AddScoped<IGetLiveDataService, GetLiveDataService>();
builder.Services.AddSingleton<INewsArticleService, NewsArticleService>();
builder.Services.AddScoped<IWatchlistService, WatchlistService>();

//// Register NewsArticle-related services
//builder.Services.AddScoped<INewsArticleService, NewsArticleService>();
//builder.Services.AddSingleton<IRabbitMQNewsConsumer, RabbitMQNewsConsumer>();
//builder.Services.AddSingleton<IRabbitMQService, RabbitMQService>();

//// Optional: Add a Hosted Service to start consuming automatically
//builder.Services.AddHostedService<NewsConsumerHostedService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
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
