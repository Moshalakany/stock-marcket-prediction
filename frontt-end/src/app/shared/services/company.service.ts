import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable  } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class CompanyService {
  private selectedCompany = new BehaviorSubject<string>('AAPL'); // Default value
  selectedCompany$ = this.selectedCompany.asObservable();

  private selectedCompanySubject = new BehaviorSubject<string>('');
  
  // Observable that components can subscribe to
  selectedCompany1$: Observable<string> = this.selectedCompanySubject.asObservable();


  updateCompany(company: string) {
    this.selectedCompany.next(company);
  }

  setSelectedCompany(symbol: string): void {
    console.log(`CompanyService: Setting selected company to ${symbol}`);
    this.selectedCompanySubject.next(symbol);
  }

  /**
   * Get the currently selected company symbol
   * @returns The current company symbol
   */
  getSelectedCompany(): string {
    return this.selectedCompanySubject.getValue();
  }
  
  /**
   * Clear the selected company
   */
  clearSelectedCompany(): void {
    this.selectedCompanySubject.next('');
  }
}
