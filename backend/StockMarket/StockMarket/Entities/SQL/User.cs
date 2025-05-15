namespace StockMarket.Entities.SQL
{
    public class User
    {
        public int UserId { get; set; }
        public string? Username { get; set; }
        public string? Email { get; set; }
        public string? PasswordHash { get; set; }
        public Portfolio? Portfolio { get; set; }

        public Watchlist? Watchlist { get; set; }

        public DateTime CreatedAt { get; set; }
        public string Role { get; set; } = string.Empty;
        public string? RefreshToken { get; set; }
        public DateTime? RefreshTokenExpiryTime { get; set; }
        
    }
}
