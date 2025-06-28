# Stock Market Prediction Platform Documentation

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Development Environment Setup](#development-environment-setup)
- [Deployment](#deployment)
- [Service Documentation](#service-documentation)
  - [SQL Server Service](#sql-server-service)
  - [MongoDB Service](#mongodb-service)
  - [Sentiment Service](#sentiment-service)
  - [Scraping Service](#scraping-service)
  - [Backend Service](#backend-service)

---

## Project Overview

This platform provides stock market prediction, portfolio management, news sentiment analysis, and data scraping functionalities. It uses a microservices architecture with .NET for backend APIs, Python for ML services, and both SQL Server and MongoDB for data storage.

---

## Project Structure

```
StockMarket/
├── Controllers/         # API controllers
├── Data/                # Database contexts (SQL and MongoDB)
├── DTOs/                # Data Transfer Objects
├── Entities/            # Entity models (SQL and NonSQL)
├── Migrations/          # Entity Framework migrations
├── Scripts/             # Data import/export scripts
├── Services/            # Business logic and interfaces
├── Docs/                # Documentation
├── appsettings*.json    # Configuration files
├── Dockerfile           # Dockerfile for .NET backend
└── StockMarket.csproj   # Project file
```

---

## Development Environment Setup

### Prerequisites

- [.NET 9 SDK](https://dotnet.microsoft.com/download)
- [Docker](https://www.docker.com/products/docker-desktop)
- [Python 3.9+](https://www.python.org/downloads/) (for ML services)
- [SQL Server](https://www.microsoft.com/en-us/sql-server/sql-server-downloads)
- [MongoDB](https://www.mongodb.com/try/download/community)

### Steps

1. **Clone the repository**
   ```sh
   git clone <repo-url>
   cd StockMarket
   ```

2. **Configure Environment**
   - Edit `appsettings.json` and `appsettings.Development.json` for connection strings and secrets.

3. **Restore .NET Dependencies**
   ```sh
   dotnet restore
   ```

4. **Apply Database Migrations**
   ```sh
   dotnet ef database update
   ```

5. **Run the Backend**
   ```sh
   dotnet run --project StockMarket
   ```

6. **(Optional) Run ML and Scraping Services**
   - See [Sentiment Service](#sentiment-service) and [Scraping Service](#scraping-service) below.

---

## Deployment

### Docker Compose (Recommended)

You can use Docker Compose to orchestrate all services (SQL Server, MongoDB, backend, ML, and scraping). Example `docker-compose.yml` (not included here) should define all services and networks.

### Individual Service Deployment

See each service section below for Docker instructions.

---

## Service Documentation

### SQL Server Service

#### Description

Provides persistent storage for user, portfolio, and stock data using Microsoft SQL Server.

#### Docker Deployment

```yaml
services:
  sqlserver:
    image: mcr.microsoft.com/mssql/server:2022-latest
    environment:
      SA_PASSWORD: "YourStrong!Passw0rd"
      ACCEPT_EULA: "Y"
    ports:
      - "1433:1433"
    volumes:
      - sqlserver_data:/var/opt/mssql
volumes:
  sqlserver_data:
```

- Update your `appsettings.json` with the correct connection string.

---

### MongoDB Service

#### Description

Stores news articles and other non-relational data.

#### Docker Deployment

```yaml
services:
  mongodb:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
volumes:
  mongodb_data:
```

- Update your `appsettings.json` with the correct MongoDB connection string.

---

### Sentiment Service

#### Description

A Python Flask API for sentiment analysis of news articles.

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY start_flask_sentiment_api.py .
COPY model_files/ ./model_files/
ENV MODEL_DIR=/app/model_files
ENV DATA_DIR=/app/model_files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "start_flask_sentiment_api.py"]
```

#### Build & Run

```sh
docker build -t sentiment-service .
docker run -p 8000:8000 sentiment-service
```

---

### Scraping Service

#### Description

A Python Flask API for scraping news data.

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask-cors
COPY . .
RUN mkdir -p news_data
EXPOSE 5000
CMD ["python", "app.py"]
```

#### Build & Run

```sh
docker build -t scraping-service .
docker run -p 5000:5000 scraping-service
```

---

### Backend Service

#### Description

The main .NET backend API service provides endpoints for user management, portfolio operations, stock data, and integration with SQL Server, MongoDB, and external ML/scraping services.

#### Docker Deployment

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: StockMarket/Dockerfile
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - ASPNETCORE_ENVIRONMENT=Production
      # Add other environment variables as needed
    depends_on:
      - sqlserver
      - mongodb
```

- Update your `appsettings.json` for connection strings and secrets.
- The backend exposes ports 8080 and 8081 by default.

---

## Additional Notes

- Ensure all environment variables and secrets are set securely in production.
- For local development, you may run SQL Server and MongoDB as containers or use local installations.
- For full orchestration, consider writing a `docker-compose.yml` to manage all services together.

---
