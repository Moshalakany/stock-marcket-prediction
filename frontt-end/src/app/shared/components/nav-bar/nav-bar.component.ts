import { Component } from '@angular/core';
import { Router , NavigationEnd } from '@angular/router';
import { UserServiceService } from '../../../core/services/user-service.service';
import { Observable } from 'rxjs';
import { AuthService } from '../../../core/services/auth.service';
@Component({
  selector: 'app-nav-bar',
  standalone: false,
  templateUrl: './nav-bar.component.html',
  styleUrl: './nav-bar.component.scss'
})
export class NavBarComponent{
  darkMode = false;
  mobileMenuOpen = false;
  hideBanner = false;
  userNameO$!: Observable<string | null>;
  userName: string | null = '';
  constructor(private router: Router,  private userService: UserServiceService, private authService: AuthService) {
     this.userNameO$ = this.userService.user$;
    this.router.events.subscribe(event => {
      if (event instanceof NavigationEnd) {
        this.hideBanner = this.router.url.includes('/auth/login') || this.router.url.includes('/auth/register');
      }
    });
  }
  ngOnInit(): void {
   
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || 
        (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      this.darkMode = true;
    }
    this.userName = localStorage.getItem('userName');
    
    window.addEventListener('storage', this.handleStorageChange.bind(this));
  }

  ngOnDestroy(): void {
    window.removeEventListener('storage', this.handleStorageChange.bind(this));
  }

  handleStorageChange(event: StorageEvent): void {
    if (event.key === 'theme') {
      this.darkMode = event.newValue === 'dark';
    }
  }

  toggleMobileMenu(): void {
    this.mobileMenuOpen = !this.mobileMenuOpen;
  }
  navigateToLogin() {
    this.router.navigate(['/auth/login']);
  }
   logout() {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('userName');
    this.userService.clearUser();
    this.router.navigate(['/auth/login']);
    this.authService.logout().subscribe({
      next: () => {
        console.log('Logout successful');
      },
      error: (error) => {
        console.error('Logout failed:', error);
      }
    });
  }
}
