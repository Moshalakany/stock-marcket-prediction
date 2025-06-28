import { NgModule , CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { StockChartComponent } from '../stocks/components/stock-chart/stock-chart.component';
import { FormsModule } from '@angular/forms';
import { CompanyProfileComponent } from './components/company-profile/company-profile.component';
import { CompanyListComponent } from './components/company-list/company-list.component';
import { RouterModule } from '@angular/router';
import { MatIconModule } from '@angular/material/icon';
import {StocksComponent} from './stocks.component';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { ReactiveFormsModule } from '@angular/forms';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule } from '@angular/material/paginator';
import { StocksRoutingModule } from './stocks-routing.module';
import { ChartForStockComponent } from './components/chart-for-stock/chart-for-stock.component';
import { PredictionChartComponent } from './components/prediction-chart/prediction-chart.component';
import { IndexesSummaryComponent } from './components/indexes-summary/indexes-summary.component';
import { ShortNumberPipe } from '../../pipes/short-number.pipe';
@NgModule({
  declarations: [
    CompanyListComponent,
    StocksComponent,
    ChartForStockComponent,
    PredictionChartComponent,
    IndexesSummaryComponent,
    ShortNumberPipe
  ],
  imports: [
    CommonModule,
    NgApexchartsModule,
    FormsModule,
    RouterModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    ReactiveFormsModule,
    MatSelectModule,
    MatProgressSpinnerModule,
    MatTableModule,
    MatPaginatorModule,
    StocksRoutingModule,
    RouterModule
  ],
  
  exports: [
    CompanyListComponent,
    ChartForStockComponent,
    PredictionChartComponent,
    IndexesSummaryComponent,
  ]
})
export class StocksModule { }
