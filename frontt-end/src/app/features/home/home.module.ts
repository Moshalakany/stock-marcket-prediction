import { NgModule, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

import { HomeRoutingModule } from './home-routing.module';

import { NewsModule } from '../news/news.module';
import { HomeComponent } from './home.component';

import { StockChartComponent } from '../stocks/components/stock-chart/stock-chart.component'; 
import {  StocksModule } from '../stocks/stocks.module';
import { FooterComponent } from './components/footer/footer.component';
@NgModule({
  declarations: [
    HomeComponent,
    FooterComponent,
  ],
  imports: [
    CommonModule,
    HomeRoutingModule,
    
    StocksModule,
    RouterModule,
    StockChartComponent
    
  ]
  
})
export class HomeModule { }
