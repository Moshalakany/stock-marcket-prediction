import { NgModule } from '@angular/core';
import { RouterModule, Routes, PreloadAllModules } from '@angular/router';
import { HomeModule } from './features/home/home.module'; // Adjust the path as necessary
import { NewsModule } from './features/news/news.module'; // Adjust the path as necessary
import { StocksModule } from './features/stocks/stocks.module'; // Adjust the path as necessary
import { NewsComponent } from './features/news/news.component';
import {CompanyProfileComponent} from './features/stocks/components/company-profile/company-profile.component'
const routes: Routes = [
  { path: '', component: NewsComponent }, // Default route
  { path: 'stocks', loadChildren: () => import('./features/stocks/stocks.module').then(m => m.StocksModule) },
  { path: 'news', loadChildren: () => import('./features/news/news.module').then(m => m.NewsModule) },
  { path: 'stocks/:ticker', component: CompanyProfileComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes, {
    preloadingStrategy: PreloadAllModules
  })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
