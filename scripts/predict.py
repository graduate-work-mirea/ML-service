#!/usr/bin/env python3
"""
Load an ONNX model and make predictions for the next day.
"""
import sys
import json
import os
import numpy as np
import onnxruntime as rt
from datetime import datetime, timedelta

def predict(input_json):
    """
    Load the ONNX model and make a prediction for the next day.
    
    Args:
        input_json (str): JSON string containing prediction parameters
    
    Returns:
        str: JSON string with prediction results
    """
    try:
        # Parse input data
        data = json.loads(input_json)
        product_id = data["product_id"]
        model_info = data["model_info"]
        
        # Load model
        model_path = model_info["model_path"]
        if not os.path.exists(model_path):
            return json.dumps({"error": f"Model not found at {model_path}"})
        
        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        
        # Calculate next day
        last_days = model_info["last_days_since_first"]
        next_day = last_days + 1
        
        # Make prediction
        prediction = sess.run(
            [label_name], 
            {input_name: np.array([[next_day]], dtype=np.float32)}
        )[0][0]
        
        # Calculate next date
        first_date = datetime.fromisoformat(model_info["first_date"])
        next_date = first_date + timedelta(days=next_day)
        
        # Format result
        result = {
            "product_id": product_id,
            "forecasted_sales": float(prediction),
            "date": next_date.isoformat()
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Read input data from stdin
    input_data = sys.stdin.read()
    
    # Make prediction and print result to stdout
    result = predict(input_data)
    print(result)
