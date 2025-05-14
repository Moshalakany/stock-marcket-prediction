import { Component, OnInit, OnDestroy, ViewChild, Input, SimpleChanges } from '@angular/core';
import { StockService } from '../../services/stock.service';
import { CommonModule } from '@angular/common';
import {
  ChartComponent,
  ApexAxisChartSeries,
  ApexChart,
  ApexXAxis,
  ApexDataLabels,
  ApexTooltip,
  ApexStroke,
  ApexYAxis,
  ApexTitleSubtitle,
  ApexFill,
  ApexLegend
} from 'ng-apexcharts';
import { interval, Subscription } from 'rxjs';
import { NgApexchartsModule } from 'ng-apexcharts';

@Component({
  selector: 'app-chart-for-stock',
  standalone: false,
  templateUrl: './chart-for-stock.component.html',
  styleUrl: './chart-for-stock.component.css'
})
export class ChartForStockComponent implements OnInit, OnDestroy {
  @ViewChild('chart') chart!: ChartComponent;
  @Input() symbol: string = 'AAPL'; // Default symbol, overridden by parent

  public series: ApexAxisChartSeries = [];
  public chartOptions: ApexChart;
  public xaxis: ApexXAxis;
  public yaxis: ApexYAxis;
  public dataLabels: ApexDataLabels;
  public tooltip: ApexTooltip;
  public stroke: ApexStroke;
  public title: ApexTitleSubtitle = {
    text: 'Stock Price',
    align: 'left',
    style: { fontSize: '24px', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: '#F1F5F9' }
  };
  public fill: ApexFill;
  public legend: ApexLegend;

  private pollingSubscription!: Subscription;
  private pollingInterval = 120000; // 2 minutes

  constructor(private stockService: StockService) {
    const data = this.generateStockData();
    this.series = [{
      name: this.symbol,
      data: data
    }];
    this.chartOptions = {
      type: 'area',
      height: 450,
      animations: { enabled: true, speed: 800 },
      zoom: {
        enabled: true,
        type: 'x',
        autoScaleYaxis: true,
        zoomedArea: {
          fill: { color: '#90CAF9', opacity: 0.4 },
          stroke: { color: '#0D47A1', opacity: 0.4, width: 1 }
        }
      },
      toolbar: {
        show: true,
        tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true },
        autoSelected: 'zoom'
      }
    };

    this.xaxis = {
      type: "datetime",
      labels: {
        datetimeUTC: false,
        style: { 
          colors: '#94A3B8',
          fontFamily: 'Inter, sans-serif'
        }
      }
    };

    this.yaxis = {
      title: { 
        text: "Stock Price",
        style: { 
          color: '#94A3B8',
          fontSize: '14px',
          fontFamily: 'Inter, sans-serif',
          fontWeight: 500
        }
      },
      labels: { 
        formatter: (val) => `$${val.toFixed(2)}`,
        style: { 
          colors: '#94A3B8',
          fontFamily: 'Inter, sans-serif'
        }
      }
    };

    this.dataLabels = { enabled: false };
    this.fill = {
      type: 'gradient',
      gradient: {
        shade: 'dark',
        type: 'vertical',
        shadeIntensity: 0.5,
        gradientToColors: ['#3B82F6'],
        inverseColors: false,
        opacityFrom: 0.7,
        opacityTo: 0.1
      }
    };

    this.tooltip = {
      enabled: true,
      theme: 'dark',
      x: { format: 'dd MMM yyyy HH:mm' },
      y: {
        formatter: (val, { seriesIndex, dataPointIndex, w }) => {
          if (!w) return `$${val}`;
  
          const seriesName = w.config.series[seriesIndex].name;
  
          if (seriesName === this.symbol) { // Candlestick (OHLC)
            const data = w.config.series[seriesIndex].data[dataPointIndex];
            return `
              <strong>Open:</strong> $${data.y[0]?.toFixed(2)}<br>
              <strong>High:</strong> $${data.y[1]?.toFixed(2)}<br>
              <strong>Low:</strong> $${data.y[2]?.toFixed(2)}<br>
              <strong>Close:</strong> $${data.y[3]?.toFixed(2)}
            `;
          } else if (seriesName === 'Volume') { // Volume bar chart
            return `<strong>Volume:</strong> ${val.toLocaleString()}`;
          } else {
            return `$${val}`; // Default case
          }
        }
      }
    };

    this.stroke = { curve: 'smooth', width: 2 };
    this.legend = { fontFamily: 'Inter, sans-serif', labels: { colors: '#94A3B8' } };
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['symbol'] && changes['symbol'].currentValue) {
      this.getStockData(this.symbol);
    }
  }

  ngOnInit(): void {
    this.startPolling();
    this.updateXAxis();
  }

  ngOnDestroy(): void {
    if (this.pollingSubscription) {
      this.pollingSubscription.unsubscribe();
    }
  }
  selectedTimeRange: '1D' | '1W' | '1M' | '1Y' = '1M';

  getStockData(symbol: string) {
     this.updateChartTitle();
    this.updateChartTitle();
    this.stockService.getStockData(symbol, this.selectedTimeRange).subscribe({
      next: (data) => this.updateChartData(data),
      error: (err) => console.error('Error fetching stock data:', err)
    });
  }



  // getStockData(symbol: string) {
  //   this.selectedSymbol = symbol.toUpperCase();
  //   this.updateChartTitle();
  //   this.stockService.getStockData(symbol, this.selectedTimeRange).subscribe({
  //     next: (data) => this.updateChartData(data),
  //     error: (err) => console.error('Error fetching stock data:', err)
  //   });
  // }


  onTimeRangeChange(range: '1D' | '1W' | '1M' | '1Y') {
    this.selectedTimeRange = range;
    this.updateXAxis();
    this.getStockData(this.symbol);
  }


  private startPolling() {
    // this.pollingSubscription = interval(this.pollingInterval).subscribe(() => {
    //   this.stockService.getStockData(this.symbol).subscribe({
    //     next: (data) => this.appendNewData(data),
    //     error: (err) => console.error('Polling error:', err)
    //   });
    // });
  }

  public updateChartData(newData: any[]) {
    this.series = [{
      name: this.symbol,
      type: 'area',
      data: newData.map(d => ({
        x: d.x,
        y: [d.open, d.high, d.low, d.close] // OHLC format
      }))
    }];
  }

  private appendNewData(newData: any[]) {
    const currentData = this.series[0].data as any[];
    const lastTimestamp = currentData.length > 0 ? currentData[currentData.length - 1].x : 0;
    const newPoints = newData.filter(point => point.x > lastTimestamp);

    if (newPoints.length > 0) {
      this.series = [{ name: this.symbol, data: [...currentData, ...newPoints] }];
    }
  }

  private updateChartTitle() {
    this.title = { ...this.title, text: `${this.symbol} Hourly Stock Price` };
  }

  private generateStockData() {
    const data = [];
    const baseDate = new Date('2025-03-13T09:00:00');
    let baseValue = 5200;

    for (let day = 0; day < 5; day++) {
      const currentDay = new Date(baseDate.getTime() + day * 24 * 60 * 60 * 1000);
      for (let hour = 0; hour < 7; hour++) {
        const timestamp = new Date(currentDay);
        timestamp.setHours(9 + hour, 30, 0, 0);
        const change = (Math.random() - 0.5) * 5;
        baseValue += change;
        data.push({ x: timestamp.getTime(), y: baseValue });
      }
    }
    return data;
  }

  onChartFocus() {
    document.body.style.overflow = 'hidden';
  }

  onChartBlur() {
    document.body.style.overflow = 'auto';
  }


  private updateXAxis() {
    let tickAmount: number | undefined;
    let labelFormat: string;
    let min: number | undefined;
    let max: number | undefined;
  
    const now = new Date().getTime();
  
    switch (this.selectedTimeRange) {
      case '1D':
        tickAmount = 6; // Show ~6 ticks (every 4 hours in a trading day)
        labelFormat = 'HH:mm'; // Show hours and minutes
        min = now - 24 * 60 * 60 * 1000; // Last 24 hours
        max = now;
        break;
      case '1W':
        tickAmount = 7; // Show daily ticks
        labelFormat = 'dd MMM'; // Show day and month (e.g., "15 Mar")
        min = now - 7 * 24 * 60 * 60 * 1000; // Last 7 days
        max = now;
        break;
      case '1M':
        tickAmount = 4; // Show weekly ticks
        labelFormat = 'dd MMM'; // Show day and month
        min = now - 30 * 24 * 60 * 60 * 1000; // Last 30 days
        max = now;
        break;
      case '1Y':
        tickAmount = 12; // Show monthly ticks
        labelFormat = 'MMM yyyy'; // Show month and year (e.g., "Mar 2024")
        min = now - 365 * 24 * 60 * 60 * 1000; // Last 365 days
        max = now;
        break;
    }
  
    this.xaxis = {
      type: 'datetime',
      min,
      max,
      tickAmount,
      labels: {
        datetimeUTC: false,
        format: labelFormat,
        style: {
          colors: '#94A3B8',
          fontFamily: 'Inter, sans-serif'
        }
      }
    };
  }
}