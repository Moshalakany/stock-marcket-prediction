using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace StockMarket.Migrations
{
    /// <inheritdoc />
    public partial class StockUpdated : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "Symbol",
                table: "Stocks",
                newName: "symbol");

            migrationBuilder.AddColumn<int>(
                name: "FullTimeEmployees",
                table: "Stocks",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "Industry",
                table: "Stocks",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "LongName",
                table: "Stocks",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "Longbusinesssummary",
                table: "Stocks",
                type: "nvarchar(max)",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "FullTimeEmployees",
                table: "Stocks");

            migrationBuilder.DropColumn(
                name: "Industry",
                table: "Stocks");

            migrationBuilder.DropColumn(
                name: "LongName",
                table: "Stocks");

            migrationBuilder.DropColumn(
                name: "Longbusinesssummary",
                table: "Stocks");

            migrationBuilder.RenameColumn(
                name: "symbol",
                table: "Stocks",
                newName: "Symbol");
        }
    }
}
