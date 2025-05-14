import { Component } from '@angular/core';
import { Router , NavigationEnd } from '@angular/router';
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

  constructor(private router: Router) {
    this.router.events.subscribe(event => {
      if (event instanceof NavigationEnd) {
        this.hideBanner = this.router.url.includes('/auth/login') || this.router.url.includes('/auth/register');
      }
    });
  }
  ngOnInit(): void {
    // Check user preference for dark mode
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || 
        (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      this.darkMode = true;
    }

    // Subscribe to theme changes
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
}
