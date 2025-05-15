using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace StockMarket.Migrations
{
    /// <inheritdoc />
    public partial class initialMigration : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Stocks",
                columns: table => new
                {
                    Symbol = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    CompanyName = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    Sector = table.Column<string>(type: "nvarchar(450)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Stocks", x => x.Symbol);
                });

            migrationBuilder.CreateTable(
                name: "Users",
                columns: table => new
                {
                    UserId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Username = table.Column<string>(type: "nvarchar(20)", maxLength: 20, nullable: false),
                    Email = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    PasswordHash = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime", nullable: false, defaultValueSql: "GETDATE()"),
                    Role = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    RefreshToken = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    RefreshTokenExpiryTime = table.Column<DateTime>(type: "datetime2", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Users", x => x.UserId);
                    table.UniqueConstraint("AK_Users_Email", x => x.Email);
                    table.UniqueConstraint("AK_Users_Username", x => x.Username);
                });

            migrationBuilder.CreateTable(
                name: "Portfolios",
                columns: table => new
                {
                    PortfolioId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    UserId = table.Column<int>(type: "int", nullable: false),
                    TotalValue = table.Column<decimal>(type: "decimal(18,2)", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Portfolios", x => x.PortfolioId);
                    table.UniqueConstraint("AK_Portfolios_UserId", x => x.UserId);
                    table.ForeignKey(
                        name: "FK_Portfolios_Users_UserId",
                        column: x => x.UserId,
                        principalTable: "Users",
                        principalColumn: "UserId",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Watchlists",
                columns: table => new
                {
                    WatchlistId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    UserId = table.Column<int>(type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Watchlists", x => x.WatchlistId);
                    table.ForeignKey(
                        name: "FK_Watchlists_Users_UserId",
                        column: x => x.UserId,
                        principalTable: "Users",
                        principalColumn: "UserId",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "PortfolioStocks",
                columns: table => new
                {
                    PortfolioId = table.Column<int>(type: "int", nullable: false),
                    StockSymbol = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    Quantity = table.Column<int>(type: "int", nullable: false),
                    PurchasePrice = table.Column<decimal>(type: "decimal(18,2)", nullable: false),
                    PurchaseDate = table.Column<DateTime>(type: "datetime2", nullable: false),
                    CurrentPrice = table.Column<decimal>(type: "decimal(18,2)", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_PortfolioStocks", x => new { x.PortfolioId, x.StockSymbol });
                    table.ForeignKey(
                        name: "FK_PortfolioStocks_Portfolios_PortfolioId",
                        column: x => x.PortfolioId,
                        principalTable: "Portfolios",
                        principalColumn: "PortfolioId",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_PortfolioStocks_Stocks_StockSymbol",
                        column: x => x.StockSymbol,
                        principalTable: "Stocks",
                        principalColumn: "Symbol",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "WatchlistStocks",
                columns: table => new
                {
                    WatchlistId = table.Column<int>(type: "int", nullable: false),
                    StockSymbol = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    AddedAt = table.Column<DateTime>(type: "datetime2", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_WatchlistStocks", x => new { x.WatchlistId, x.StockSymbol });
                    table.ForeignKey(
                        name: "FK_WatchlistStocks_Stocks_StockSymbol",
                        column: x => x.StockSymbol,
                        principalTable: "Stocks",
                        principalColumn: "Symbol",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_WatchlistStocks_Watchlists_WatchlistId",
                        column: x => x.WatchlistId,
                        principalTable: "Watchlists",
                        principalColumn: "WatchlistId",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_PortfolioStocks_PortfolioId",
                table: "PortfolioStocks",
                column: "PortfolioId");

            migrationBuilder.CreateIndex(
                name: "IX_PortfolioStocks_StockSymbol",
                table: "PortfolioStocks",
                column: "StockSymbol");

            migrationBuilder.CreateIndex(
                name: "IX_Stocks_Sector",
                table: "Stocks",
                column: "Sector");

            migrationBuilder.CreateIndex(
                name: "IX_Watchlists_UserId",
                table: "Watchlists",
                column: "UserId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_WatchlistStocks_StockSymbol",
                table: "WatchlistStocks",
                column: "StockSymbol");

            migrationBuilder.CreateIndex(
                name: "IX_WatchlistStocks_WatchlistId",
                table: "WatchlistStocks",
                column: "WatchlistId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "PortfolioStocks");

            migrationBuilder.DropTable(
                name: "WatchlistStocks");

            migrationBuilder.DropTable(
                name: "Portfolios");

            migrationBuilder.DropTable(
                name: "Stocks");

            migrationBuilder.DropTable(
                name: "Watchlists");

            migrationBuilder.DropTable(
                name: "Users");
        }
    }
}
