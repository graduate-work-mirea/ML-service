# ML Price and Sales Prediction Service

This service provides an API for predicting product prices and sales demand using LightGBM machine learning models.

## Features

- Training LightGBM models for product price and sales demand prediction
- REST API for making predictions based on product features
- Automatic model training on startup if models don't exist

## Architecture

The service follows a clean architecture pattern:

- **Repository**: Handles file operations and Python script execution
- **Service**: Contains business logic for model training and prediction
- **Controller**: Exposes REST APIs for client interaction

## API Endpoints

The service exposes the following endpoints:

- `POST /api/v1/predict`: Make a prediction for product price and sales
- `POST /api/v1/train`: Train new models using the processed data
- `GET /api/v1/status`: Check if models are trained and available

## Setup and Configuration

1. Install dependencies:
   ```
   go mod download
   pip install -r requirements.txt
   ```

2. Configure environment variables (see `.env.example`):
   ```
   cp .env.example .env
   ```

3. Run the service:
   ```
   go run main.go
   ```

## Data Requirements

The service expects processed data files in the `processor_data/processed` directory:
- `train_data.csv`: Training data with features and target variables
- `test_data.csv`: Test data with similar structure

## Models

The service uses LightGBM to train two regression models:
1. **Price Prediction**: Predicts product price 7 days in the future
2. **Sales Prediction**: Predicts total sales quantity over the next 7 days

Models are stored in the configured `MODEL_PATH` directory.

## Example Prediction Request

```json
{
  "product_name": "Детская книга \"Гарри Поттер\" Дж. Роулинг",
  "brand": "Махаон",
  "category": "Книги",
  "region": "Москва",
  "seller": "АО «Шарапов»",
  "price": 750.0,
  "original_price": 750.0,
  "discount_percentage": 0.0,
  "stock_level": 388.0,
  "customer_rating": 4.8,
  "review_count": 663.0,
  "delivery_days": 1.0,
  "is_weekend": false,
  "is_holiday": false,
  "day_of_week": 5,
  "month": 3,
  "quarter": 1,
  "sales_quantity_lag_1": 9.0,
  "price_lag_1": 835.0,
  "sales_quantity_lag_3": 7.0,
  "price_lag_3": 469.0,
  "sales_quantity_lag_7": 15.0,
  "price_lag_7": 1020.0,
  "sales_quantity_rolling_mean_3": 20.33,
  "price_rolling_mean_3": 831.33,
  "sales_quantity_rolling_mean_7": 22.0,
  "price_rolling_mean_7": 813.10
}
```