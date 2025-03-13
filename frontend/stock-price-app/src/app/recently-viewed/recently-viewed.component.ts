import { Component } from '@angular/core';

@Component({
  selector: 'app-recently-viewed',
  templateUrl: './recently-viewed.component.html',
  styleUrls: ['./recently-viewed.component.css']
})
export class RecentlyViewedComponent {
  recentlyViewedItems = [
    {
      name: 'Apple',
      price: '$238.03',
      change: -2.31,
      
    },
    {
      name: 'Pi Network',
      price: '$1.663239',
      change: -11.86,
     
    },
    {
      name: 'Bitcoin',
      price: '$86,411',
      change: -8.06,
     
    }
  ];

  getChangeClass(change: number) {
    return change < 0 ? 'negative' : 'positive';
  }
}
