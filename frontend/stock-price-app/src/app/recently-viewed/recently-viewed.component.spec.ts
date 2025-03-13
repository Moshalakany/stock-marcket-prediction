import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecentlyViewedComponent } from './recently-viewed.component';

describe('RecentlyViewedComponent', () => {
  let component: RecentlyViewedComponent;
  let fixture: ComponentFixture<RecentlyViewedComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [RecentlyViewedComponent]
    });
    fixture = TestBed.createComponent(RecentlyViewedComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
