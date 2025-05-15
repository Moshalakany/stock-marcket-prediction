import { Component } from '@angular/core';
import { NewsListComponent } from '../news/components/news-list/news-list.component';
import { CommonModule } from '@angular/common';
import { StockChartComponent } from '../stocks/components/stock-chart/stock-chart.component';  // ✅ Try adding this
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
  standalone: false,
})
export class HomeComponent {
  darkMode = false;
  selectedSymbol = 'AAPL';
  selectedInterval = '1D';
  intervals = ['1D', '1W', '1M', '3M', '1Y', '5Y'];

  constructor() { }

  ngOnInit(): void {
    // Check user preference for dark mode
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || 
        (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      this.darkMode = true;
    }
  }

  toggleTheme(): void {
    this.darkMode = !this.darkMode;
    localStorage.setItem('theme', this.darkMode ? 'dark' : 'light');
  }

  updateChartInterval(interval: string): void {
    this.selectedInterval = interval;
  }
}
