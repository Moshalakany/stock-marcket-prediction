import { NgModule, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { NgApexchartsModule } from 'ng-apexcharts';
import { NewsRoutingModule } from './news-routing.module';
import { NewsComponent } from './news.component';
import { NewsListComponent } from './components/news-list/news-list.component';
import { NewsDetailComponent } from './components/news-detail/news-detail.component';
import {NewsCardComponent} from './components/news-card/news-card.component';
import { StockChartComponent } from '../stocks/components/stock-chart/stock-chart.component';
import {  StocksModule } from '../stocks/stocks.module';
import {ButtonComponent} from '../../shared/components/button/button.component';
import { NewsPageComponent } from './components/news-page/news-page.component';
import { FormsModule } from '@angular/forms';
import { NewsPageForStockComponent } from './components/news-page-for-stock/news-page-for-stock.component';
import { TimeAgoPipe } from '../../pipes/time-ago.pipe';
@NgModule({
  declarations: [
    NewsComponent,
    NewsListComponent,
    NewsDetailComponent,
    NewsCardComponent,
    NewsPageComponent,
    NewsPageForStockComponent,
    TimeAgoPipe
  ],
  imports: [
    CommonModule,
    NewsRoutingModule,
    RouterModule,
    StockChartComponent,
    NgApexchartsModule,
    StocksModule,
    ButtonComponent,
    FormsModule,
  
  ],
  exports: [ 
    NewsListComponent,
    NewsDetailComponent,
    NewsComponent,
    NewsCardComponent,
    CommonModule,
    NewsPageComponent,
    NewsPageForStockComponent,
    TimeAgoPipe
  ]
  ,
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class NewsModule { }
