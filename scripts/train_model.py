#!/usr/bin/env python3
"""
Train a linear regression model on sales data and save it in ONNX format.
"""
import sys
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def train_model(data_json):
    """
    Train a linear regression model on the provided data and save it as an ONNX file.
    
    Args:
        data_json (str): JSON string containing sales data
    
    Returns:
        str: Path to the saved model
    """
    try:
        # Parse input data
        data = json.loads(data_json)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert dates to numeric (days since first date)
        df['date'] = pd.to_datetime(df['date'])
        first_date = df['date'].min()
        df['days_since_first'] = (df['date'] - first_date).dt.days
        
        # Prepare data for training
        X = df[['days_since_first']].values
        y = df['sales'].values
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Convert to ONNX
        initial_type = [('float_input', FloatTensorType([None, 1]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Create models directory if it doesn't exist
        os.makedirs("../models", exist_ok=True)
        
        # Save model
        product_id = df['product_id'].iloc[0]
        model_path = f"../models/model_{product_id}.onnx"
        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Return model path and last date for prediction
        result = {
            "model_path": model_path,
            "product_id": product_id,
            "last_date": df['date'].max().isoformat(),
            "last_days_since_first": int(df['days_since_first'].max()),
            "first_date": first_date.isoformat()
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Read input data from stdin
    input_data = sys.stdin.read()
    
    # Train model and print result to stdout
    result = train_model(input_data)
    print(result)
