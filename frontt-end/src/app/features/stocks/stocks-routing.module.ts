import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {CompanyListComponent} from './components/company-list/company-list.component';
import {StocksComponent} from './stocks.component';
const routes: Routes = [
  {path: '', component: StocksComponent}
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class StocksRoutingModule { }
