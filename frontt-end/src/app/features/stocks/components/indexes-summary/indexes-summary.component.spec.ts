import { ComponentFixture, TestBed } from '@angular/core/testing';

import { IndexesSummaryComponent } from './indexes-summary.component';

describe('IndexesSummaryComponent', () => {
  let component: IndexesSummaryComponent;
  let fixture: ComponentFixture<IndexesSummaryComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [IndexesSummaryComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(IndexesSummaryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
