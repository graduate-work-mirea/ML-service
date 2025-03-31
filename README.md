# ML-service

## Overview

ML-service is a microservice for demand forecasting within a product demand evaluation system. This MVP (Minimum Viable Product) service uses a simple linear regression model to predict future product demand based on historical sales data.

## Features

- Consumes processed sales data from RabbitMQ
- Trains linear regression models using Python's scikit-learn
- Saves models in ONNX format for efficient inference
- Provides REST API for requesting demand forecasts
- Stores forecast results in PostgreSQL database

## Architecture

The service follows a microservice architecture pattern and consists of the following components:

1. **RabbitMQ Consumer**: Listens for processed sales data and accumulates it for model training
2. **ML Processing**: Python-based machine learning component for training and prediction
3. **REST API**: HTTP interface for requesting demand forecasts
4. **Database**: PostgreSQL storage for forecast results

## Input/Output

### Inputs

1. **RabbitMQ Queue (processed_data_queue)**:
   ```json
   {
     "product_id": "123",
     "sales": 100,
     "date": "2024-10-01T00:00:00Z"
   }
   ```

2. **REST API (POST /predict)**:
   ```json
   {
     "product_id": "123"
   }
   ```

### Outputs

1. **REST API Response**:
   ```json
   {
     "product_id": "123",
     "forecasted_sales": 105,
     "date": "2024-10-02T00:00:00Z"
   }
   ```

2. **PostgreSQL Database**:
   - Forecasts are stored in the `forecasts` table

## Technical Details

- **Go (1.21+)**: Main service implementation, REST API, and data handling
- **Python (3.11)**: Machine learning processing with scikit-learn and ONNX
- **RabbitMQ**: Message queue for receiving processed data
- **PostgreSQL**: Database for storing forecasts
- **ONNX**: Model serialization format for interoperability

## Setup and Deployment

### Environment Variables

- `RABBITMQ_URL`: Connection URL for RabbitMQ (default: `amqp://guest:guest@localhost:5672/`)
- `POSTGRES_DSN`: Connection string for PostgreSQL (default: `postgres://postgres:postgres@localhost:5432/ml_service?sslmode=disable`)

### Running with Docker

```bash
docker build -t ml-service .
docker run -p 8080:8080 -e RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/ -e POSTGRES_DSN=postgres://postgres:postgres@postgres:5432/ml_service?sslmode=disable ml-service
```

### Database Migration

Before running the service, ensure the PostgreSQL database has the required schema:

```sql
CREATE TABLE forecasts (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(50),
    forecast_result JSONB NOT NULL, -- {"forecasted_sales": 105, "date": "2024-10-02T00:00:00Z"}
    forecasted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Endpoints

- `POST /predict`: Request a forecast for a specific product
- `GET /health`: Health check endpoint

## ML Processing

The service implements a simple linear regression model that:

1. Collects at least 3 data points for a product
2. Trains a linear regression model on sales vs. time
3. Predicts the next day's sales
4. Saves the model in ONNX format for future predictions

## Limitations (MVP)

- Simple linear regression model only
- Requires at least 3 data points per product
- Predicts only one day ahead
- No model versioning or evaluation metrics
- Retrains model with each new batch of data