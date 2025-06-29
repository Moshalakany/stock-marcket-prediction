import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class UserServiceService {

  private userSubject = new BehaviorSubject<string | null>(null); // null = not logged in
  public user$: Observable<string | null> = this.userSubject.asObservable();

  constructor() {
    const storedUser = localStorage.getItem('userName');
    if (storedUser) {
      this.userSubject.next(storedUser);
    }
  }

  setUser(userName: string) {
    localStorage.setItem('userName', userName);
    this.userSubject.next(userName);
  }

  clearUser() {
    localStorage.removeItem('userName');
    this.userSubject.next(null);
  }

  get currentUser(): string | null {
    return this.userSubject.value;
  }

  isLoggedIn(): boolean {
    return !!this.userSubject.value;
  }
}
