import { NgModule, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { BrowserModule, provideClientHydration, withEventReplay } from '@angular/platform-browser';
import { StocksModule } from './features/stocks/stocks.module';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { StockChartComponent } from './features/stocks/components/stock-chart/stock-chart.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations'; 
import { StockService } from './features/stocks/services/stock.service';
import { FormsModule } from '@angular/forms';
import { NgApexchartsModule } from 'ng-apexcharts';
import { HttpClientModule } from '@angular/common/http';
import { NavBarComponent } from './shared/components/nav-bar/nav-bar.component';
import { CoreModule } from './core/core.module';
import { TimeAgoPipe } from './pipes/time-ago.pipe';
import { ShortNumberPipe } from './pipes/short-number.pipe';

@NgModule({
  declarations: [
    AppComponent,
    NavBarComponent,
  ],
  exports: [
    AppComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    NgApexchartsModule,
    StocksModule,
    HttpClientModule,
    FormsModule,
    CoreModule,
    StockChartComponent
  ],
  providers: [
    provideClientHydration(withEventReplay()),
    StockService
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA], 
  bootstrap: [AppComponent]
})
export class AppModule { }


