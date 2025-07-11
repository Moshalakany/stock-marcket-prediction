import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { PredictionsRoutingModule } from './predictions-routing.module';
import { PredictionsComponent } from './predictions.component';


@NgModule({
  declarations: [
    PredictionsComponent
  ],
  imports: [
    CommonModule,
    PredictionsRoutingModule
  ]
})
export class PredictionsModule { }
