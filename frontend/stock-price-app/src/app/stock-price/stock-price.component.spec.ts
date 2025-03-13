import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StockPriceComponent } from './stock-price.component';

describe('StockPriceComponent', () => {
  let component: StockPriceComponent;
  let fixture: ComponentFixture<StockPriceComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [StockPriceComponent]
    });
    fixture = TestBed.createComponent(StockPriceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
