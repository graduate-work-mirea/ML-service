import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
import pickle
import sys
import argparse
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split

class LightGBMPredictor:
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the LightGBM predictor

        Args:
            model_dir: Directory to save/load model files
        """
        self.model_dir = model_dir
        self.price_model = None
        self.sales_model = None
        self.feature_names = None
        self.categorical_features = None

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Prepare features for training/prediction

        Args:
            df: Pandas DataFrame with processed data

        Returns:
            Tuple of features DataFrame, feature names, and categorical feature names
        """
        # Categorical features
        categorical_features = ['brand', 'region', 'category', 'seller', 'day_of_week', 'month', 'quarter']

        # Numerical features
        numerical_features = [
            'price', 'original_price', 'discount_percentage', 'stock_level',
            'customer_rating', 'review_count', 'delivery_days', 'is_weekend', 'is_holiday',
            'sales_quantity_lag_1', 'price_lag_1', 'sales_quantity_lag_3', 'price_lag_3',
            'sales_quantity_lag_7', 'price_lag_7', 'sales_quantity_rolling_mean_3',
            'price_rolling_mean_3', 'sales_quantity_rolling_mean_7', 'price_rolling_mean_7'
        ]

        # Combine features
        feature_names = numerical_features + categorical_features

        # Ensure is_weekend and is_holiday are converted to integers
        df['is_weekend'] = df['is_weekend'].astype(int)
        df['is_holiday'] = df['is_holiday'].astype(int)

        # Convert categorical features to category type
        for cat_feat in categorical_features:
            if cat_feat in df.columns:
                df[cat_feat] = df[cat_feat].astype('category')

        # Return features DataFrame and names
        return df[feature_names], feature_names, categorical_features

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Проверка данных на наличие необходимых столбцов и минимального количества строк

        Args:
            df: Pandas DataFrame с данными

        Returns:
            True, если данные валидны, False в противном случае
        """
        required_columns = ['price_target', 'sales_target', 'brand', 'region', 'category', 'seller', 'price', 'original_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Отсутствуют обязательные столбцы: {missing_columns}")
            return False
        if len(df) < 10:
            print("Недостаточно данных для обучения")
            return False
        return True

    def remove_outliers(self, df: pd.DataFrame, columns: List[str], lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
        """
        Remove outliers from the DataFrame based on specified columns and quantiles.

        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            lower_quantile: Lower quantile threshold (default: 0.01)
            upper_quantile: Upper quantile threshold (default: 0.99)

        Returns:
            DataFrame with outliers removed
        """
        for col in columns:
            lower_bound = df[col].quantile(lower_quantile)
            upper_bound = df[col].quantile(upper_quantile)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def train(self, train_data_path: str, val_data_path: str) -> Dict[str, Any]:
        # Function to log to both stderr and stdout
        def log_info(msg):
            sys.stderr.write(msg + "\n")
            print(f"INFO: {msg}")
            
        log_info(f"Загрузка обучающих данных из {train_data_path}")
        train_df = pd.read_csv(train_data_path)

        log_info(f"Загрузка валидационных данных из {val_data_path}")
        val_df = pd.read_csv(val_data_path)

        if not self.validate_data(train_df) or not self.validate_data(val_df):
            error_msg = "Некорректные обучающие или валидационные данные"
            log_info(f"ОШИБКА: {error_msg}")
            raise ValueError(error_msg)

        # Удаление выбросов из тренировочных данных
        train_df = self.remove_outliers(train_df, ['price_target', 'sales_target'])

        # Удаление строк с пропущенными значениями в целевых переменных
        train_df = train_df.dropna(subset=['price_target', 'sales_target'])
        val_df = val_df.dropna(subset=['price_target', 'sales_target'])

        X_train, self.feature_names, self.categorical_features = self._prepare_features(train_df)
        y_price_train = train_df['price_target'].values
        y_sales_train = train_df['sales_target'].values

        X_val, _, _ = self._prepare_features(val_df)
        y_price_val = val_df['price_target'].values
        y_sales_val = val_df['sales_target'].values

        log_info(f"Обучение на {len(X_train)} примерах с {len(self.feature_names)} признаками")
        log_info(f"Валидация на {len(X_val)} примерах")

        lgb_train_price = lgb.Dataset(
            X_train,
            label=y_price_train,
            categorical_feature=self.categorical_features,
            silent=True
        )
        lgb_val_price = lgb.Dataset(
            X_val,
            label=y_price_val,
            reference=lgb_train_price,
            categorical_feature=self.categorical_features,
            silent=True
        )
        lgb_train_sales = lgb.Dataset(
            X_train,
            label=y_sales_train,
            categorical_feature=self.categorical_features,
            silent=True
        )
        lgb_val_sales = lgb.Dataset(
            X_val,
            label=y_sales_val,
            reference=lgb_train_sales,
            categorical_feature=self.categorical_features,
            silent=True
        )

        # Обновленные параметры модели с уменьшенной сложностью
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # Уменьшено с 31 до 15
            'max_depth': 6,    # Добавлено ограничение глубины
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50)
        ]

        log_info("Обучение модели предсказания цены...")
        self.price_model = lgb.train(
            params,
            lgb_train_price,
            num_boost_round=1000,
            valid_sets=[lgb_train_price, lgb_val_price],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        log_info("Обучение модели предсказания продаж...")
        self.sales_model = lgb.train(
            params,
            lgb_train_sales,
            num_boost_round=1000,
            valid_sets=[lgb_train_sales, lgb_val_sales],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        self.save_models()

        metrics = {
            "price_model": {
                "best_iteration": self.price_model.best_iteration,
                "best_score": self.price_model.best_score['valid']['rmse']
            },
            "sales_model": {
                "best_iteration": self.sales_model.best_iteration,
                "best_score": self.sales_model.best_score['valid']['rmse']
            }
        }
        
        # Log the training results
        log_info(f"Обучение завершено. Метрики моделей:")
        log_info(f"Модель цены - Лучшая итерация: {metrics['price_model']['best_iteration']}, Лучший RMSE: {metrics['price_model']['best_score']:.2f}")
        log_info(f"Модель продаж - Лучшая итерация: {metrics['sales_model']['best_iteration']}, Лучший RMSE: {metrics['sales_model']['best_score']:.2f}")
        
        # Print the final JSON result - this will be parsed by the Go service
        print(json.dumps(metrics))
        
        return metrics

    def save_models(self) -> None:
        """Save trained models to disk"""
        if self.price_model is not None:
            with open(os.path.join(self.model_dir, 'price_model.pkl'), 'wb') as f:
                pickle.dump(self.price_model, f)

        if self.sales_model is not None:
            with open(os.path.join(self.model_dir, 'sales_model.pkl'), 'wb') as f:
                pickle.dump(self.sales_model, f)

        # Save feature names and categorical features
        if self.feature_names is not None and self.categorical_features is not None:
            with open(os.path.join(self.model_dir, 'feature_info.json'), 'w') as f:
                json.dump({
                    'feature_names': self.feature_names,
                    'categorical_features': self.categorical_features
                }, f)

    def load_models(self) -> bool:
        """
        Load trained models from disk

        Returns:
            True if models were loaded successfully, False otherwise
        """
        try:
            # Load price model
            with open(os.path.join(self.model_dir, 'price_model.pkl'), 'rb') as f:
                self.price_model = pickle.load(f)

            # Load sales model
            with open(os.path.join(self.model_dir, 'sales_model.pkl'), 'rb') as f:
                self.sales_model = pickle.load(f)

            # Load feature info
            with open(os.path.join(self.model_dir, 'feature_info.json'), 'r') as f:
                feature_info = json.load(f)
                self.feature_names = feature_info['feature_names']
                self.categorical_features = feature_info['categorical_features']

            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict(self, product_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Make predictions for a product

        Args:
            product_data: Dictionary with product features

        Returns:
            Dictionary with predicted price and sales
        """
        if self.price_model is None or self.sales_model is None:
            if not self.load_models():
                raise ValueError("Models not trained or loaded properly")

        # Convert product data to DataFrame
        df = pd.DataFrame([product_data])

        # Convert booleans to integers
        if 'is_weekend' in df.columns:
            df['is_weekend'] = df['is_weekend'].astype(int)
        if 'is_holiday' in df.columns:
            df['is_holiday'] = df['is_holiday'].astype(int)

        # Convert categorical features to category type
        for cat_feat in self.categorical_features:
            if cat_feat in df.columns:
                df[cat_feat] = df[cat_feat].astype('category')

        # Prepare features
        X = df[self.feature_names]

        # Make predictions
        price_pred = self.price_model.predict(X)[0]
        sales_pred = self.sales_model.predict(X)[0]

        return {
            "predicted_price": float(price_pred),
            "predicted_sales": float(sales_pred)
        }

def main():
    """
    Main entry point for the script
    """
    # Function to log to both stderr and stdout
    def log_info(msg):
        sys.stderr.write(msg + "\n")
        print(f"INFO: {msg}")
    
    parser = argparse.ArgumentParser(description="LightGBM Model for Product Price and Sales Prediction")
    parser.add_argument("action", choices=["train", "predict"], help="Action to perform: train or predict")
    parser.add_argument("train_data", help="Path to training data CSV for training or JSON string for prediction")
    parser.add_argument("--val-data", help="Path to validation data CSV (required for training)")
    parser.add_argument("--model-dir", default="models", help="Directory for model files")

    args = parser.parse_args()
    log_info(f"Запуск с параметрами: action={args.action}, data={args.train_data}, model_dir={args.model_dir}")

    predictor = LightGBMPredictor(model_dir=args.model_dir)

    if args.action == "train":
        if not args.val_data:
            log_info("ОШИБКА: необходимо указать путь к валидационным данным с помощью --val-data")
            sys.exit(1)
        log_info(f"Запуск обучения моделей с данными: {args.train_data} и {args.val_data}")
        metrics = predictor.train(args.train_data, args.val_data)
        # Note: train() function now handles the printing of the metrics JSON
    elif args.action == "predict":
        try:
            product_data = json.loads(args.train_data)
            log_info("Запуск предсказания для данных продукта")
            prediction = predictor.predict(product_data)
            log_info(f"Результат предсказания: цена={prediction['predicted_price']:.2f}, продажи={prediction['predicted_sales']:.2f}")
            print(json.dumps(prediction))
        except json.JSONDecodeError:
            log_info("ОШИБКА: некорректный формат JSON для предсказания")
            print(json.dumps({"error": "Invalid JSON input for prediction"}))
            sys.exit(1)
        except Exception as e:
            log_info(f"ОШИБКА при предсказании: {str(e)}")
            print(json.dumps({"error": str(e)}))
            sys.exit(1)

if __name__ == "__main__":
    main()