import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { NgChartsModule } from 'ng2-charts';  // <-- use NgChartsModule instead of ChartsModule

import { AppComponent } from './app.component';
import { NavComponent } from './nav/nav.component';
import { StockPriceComponent } from './stock-price/stock-price.component';
import { RecentlyViewedComponent } from './recently-viewed/recently-viewed.component';


@NgModule({
  declarations: [
    AppComponent,
    StockPriceComponent,
    NavComponent,
    RecentlyViewedComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    NgChartsModule  // <-- Import the new module name here
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
