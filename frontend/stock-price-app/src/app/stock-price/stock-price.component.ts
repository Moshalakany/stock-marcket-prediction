import { Component, OnInit } from '@angular/core';
import { StockService } from '../../stock.service';
import { ChartOptions, ChartDataset } from 'chart.js';

@Component({
  selector: 'app-stock-price',           // This is the component's selector
  templateUrl: './stock-price.component.html', // Path to the template HTML file
  styleUrls: ['./stock-price.component.css']   // Path to the component's CSS file
})
export class StockPriceComponent implements OnInit {
  appleStockPrice: number = 0;
  activeTab: string = 'overview'; // Initially set to 'overview' tab

  chartOptions: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        ticks: { color: '#909090' }, // for dark theme
      },
      y: {
        ticks: { color: '#909090' },
        grid: { color: 'rgba(127, 131, 127, 0.24)' },
        beginAtZero: true
      }
    },
    plugins: {
      legend: {
        labels: { color: '#ffffff' } // legend text color
      }
    }
  };

  chartData: ChartDataset[] = [
    {
      data: [235, 59, 70, 239, 240],
      label: 'Stock Price',
      borderColor: '#04e82a',         // line color
      pointBackgroundColor: '#04e82a',
      backgroundColor: 'rgba(31, 132, 6, 0.1)',  // fill color under the line
      borderWidth: 2,                  // thickness of the line
      pointRadius: 3,                  // size of the dots
      pointHoverRadius: 1,             // size of dots on hover
      fill: true,
      type: 'line'                      // whether to fill area under the line
    }
  ];

  chartLabels: string[] = [];

  constructor(private stockService: StockService) {}

  ngOnInit() {
    // Fetch the current price
    this.stockService.getStockPrice('AAPL').subscribe((data) => {
      this.appleStockPrice = data.price;
    });

    // Fetch the historical data
    this.stockService.getStockHistory('AAPL').subscribe((data) => {
      const history = data.history;
      this.chartData[0].data = history.map((item: { price: any; }) => item.price);
      this.chartLabels = history.map((item: { date: any; }) => item.date);
    });
  }

  // Method to handle tab switching
  selectTab(tab: string) {
    this.activeTab = tab;
  }

  // Method for subscribing to updates
  subscribeForUpdates() {
    // Placeholder logic for subscription (you can replace this with actual logic)
    alert('You have been subscribed for updates!');
  }

}
