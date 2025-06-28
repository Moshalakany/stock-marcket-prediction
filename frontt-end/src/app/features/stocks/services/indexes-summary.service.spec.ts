import { TestBed } from '@angular/core/testing';

import { IndexesSummaryService } from './indexes-summary.service';

describe('IndexesSummaryService', () => {
  let service: IndexesSummaryService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(IndexesSummaryService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
