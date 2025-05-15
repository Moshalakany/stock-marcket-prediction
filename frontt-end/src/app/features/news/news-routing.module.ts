import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { NewsDetailComponent } from './components/news-detail/news-detail.component';
import {NewsComponent} from './news.component';
import { NewsListComponent } from './components/news-list/news-list.component';  
import { NewsPageComponent } from './components/news-page/news-page.component';

const routes: Routes = [
  { path: '', component: NewsPageComponent },  // Default component for /news,  
  { path: ':id', component: NewsDetailComponent }  
];
@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class NewsRoutingModule { }
