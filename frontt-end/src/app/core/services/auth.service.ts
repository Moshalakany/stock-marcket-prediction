import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { LoginRequest ,AuthResponse } from '../models/user';

@Injectable({
  providedIn: 'root'
})
export class AuthService {

  private apiUrl = 'https://localhost:52307/api/Auth';

  constructor(private http: HttpClient) {}

  login(data: LoginRequest): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/login`, data);
  }

register(data: { username: string; email: string; password: string }): Observable<any> {

  return this.http.post(`${this.apiUrl}/register`, data);
  
}

logout(): Observable<any> {
  return this.http.post(`${this.apiUrl}/logout`, {});
}
}
