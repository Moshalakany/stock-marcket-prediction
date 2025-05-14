import { Component, OnInit, Input } from '@angular/core';

import { TickerInfo } from '../../../../core/models/news.model';
import { NewsArticle } from '../../../../core/models/news.model';
import {TimeAgoPipe} from '../../../../pipes/time-ago.pipe';
@Component({
  selector: 'app-news-card',
  templateUrl: './news-card.component.html',
  styleUrls: ['./news-card.component.css'],
  standalone: false
})
export class NewsCardComponent implements OnInit {
  @Input() article!: NewsArticle;
  polarity: string = 'positive';
  constructor() { }

  ngOnInit(): void { }
}