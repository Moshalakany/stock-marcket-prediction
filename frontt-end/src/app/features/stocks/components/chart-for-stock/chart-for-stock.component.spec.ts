import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChartForStockComponent } from './chart-for-stock.component';

describe('ChartForStockComponent', () => {
  let component: ChartForStockComponent;
  let fixture: ComponentFixture<ChartForStockComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ChartForStockComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChartForStockComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
