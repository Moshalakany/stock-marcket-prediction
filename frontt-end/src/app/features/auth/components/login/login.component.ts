import { Component } from '@angular/core';
import { AuthService } from '../../../../core/services/auth.service';
import { LoginRequest } from '../../../../core/models/user';
import { Router } from '@angular/router';
import { UserServiceService } from '../../../../core/services/user-service.service';
@Component({
  selector: 'app-login',
  standalone: false,
  templateUrl: './login.component.html',
  styleUrl: './login.component.css'
})
export class LoginComponent {

  errorMessage: string | null = null;
  loginData: LoginRequest = {
    username: '',
    password: ''
  };

  constructor(private authService: AuthService, private router: Router, private userService: UserServiceService) {}

  onLogin() {
    this.authService.login(this.loginData).subscribe({
      next: (response) => {
        console.log('Login successful:', response);
        // Save token in localStorage or service
        localStorage.setItem('accessToken', response.accessToken);
        localStorage.setItem('userName', this.loginData.username); // Save from form input
        this.userService.setUser(this.loginData.username);
        this.router.navigate(['/']); 
      },
      error: (error) => {
        console.error('Login failed:', error);
         this.errorMessage = 'Invalid username or password. Please try again.';
      }
    });
  }
   clearError() {
    if (this.errorMessage) {
      this.errorMessage = null;
    }
  }
}
