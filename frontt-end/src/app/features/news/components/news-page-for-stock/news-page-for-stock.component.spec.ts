import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NewsPageForStockComponent } from './news-page-for-stock.component';

describe('NewsPageForStockComponent', () => {
  let component: NewsPageForStockComponent;
  let fixture: ComponentFixture<NewsPageForStockComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [NewsPageForStockComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(NewsPageForStockComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
