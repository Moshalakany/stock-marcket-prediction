// shared/button/button.component.ts
import { Component, Input, Output, EventEmitter } from '@angular/core';
import {CommonModule} from  '@angular/common';
@Component({
  selector: 'app-button',
  templateUrl: './button.component.html',
  styleUrls: ['./button.component.scss'],
  standalone: true,
  imports: [CommonModule]
})
export class ButtonComponent {
  @Input() variant: 'primary' | 'secondary' | 'text' = 'primary';
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  @Input() customClasses = '';
  @Input() disabled = false;
  @Output() onClick = new EventEmitter<Event>();

  get buttonClasses(): string {
    return `transition-all duration-200 font-medium rounded-lg focus:outline-none focus:ring-4 ${
      this.sizeClasses[this.size]
    } ${this.variantClasses[this.variant]}`;
  }

  private sizeClasses = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2.5 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  private variantClasses = {
    primary: 'bg-primary text-white hover:bg-secondary focus:ring-blue-300 dark:focus:ring-blue-800',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-300 dark:bg-gray-600 dark:text-white dark:hover:bg-gray-700 dark:focus:ring-gray-800',
    text: 'text-primary hover:bg-gray-100 dark:hover:bg-gray-800 focus:ring-gray-300'
  };
}