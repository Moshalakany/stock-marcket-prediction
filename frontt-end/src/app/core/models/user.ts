export interface User {
  id: string;
  username: string;
  email: string;
  roles: string[];
}

export interface LoginRequest {
  username: string;
  password: string;
}
export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
}
