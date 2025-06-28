using Microsoft.EntityFrameworkCore.Migrations;


#nullable disable

namespace StockMarket.Migrations
{
    /// <inheritdoc />
    public partial class watchlistUpdate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Name",
                table: "Watchlists",
                type: "nvarchar(100)",
                maxLength: 100,
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Name",
                table: "Watchlists");
        }
    }
}
