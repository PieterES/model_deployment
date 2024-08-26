import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import math
import datetime
from fastapi import FastAPI

# Initialize Flask app

app = Flask(__name__)
# Load the pre-trained model
def load_model():
    with open('models/forecasting_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model


# Define a function to convert log-transformed predictions to actual units
def convert_log_to_units(prediction):
    return int(math.exp(prediction))

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    data = request.get_json(force=True)
    print(data)
    # Check if the correct data keys are provided
    required_keys = ['StoreCount', 'ShelfCapacity', 'PromoShelfCapacity', 'IsPromo', 'ItemNumber', 
                     'CategoryCode', 'GroupCode', 'month', 'weekday', 'UnitSales_-7', 'UnitSales_-14', 'UnitSales_-21']
    
    if not all(key in data for key in required_keys):
        return jsonify({'error': 'Invalid input data'}), 400
    
    # Create a DataFrame from the input data
    df = pd.DataFrame([data])

    # Ensure correct data types
    # df['IsPromo'] = df['IsPromo'] == 'true'  # Assuming boolean is sent as 'true'/'false' string
    df['ItemNumber'] = df['ItemNumber'].astype('category')
    df['CategoryCode'] = df['CategoryCode'].astype('category')
    df['GroupCode'] = df['GroupCode'].astype('category')
    
    # Predict using the loaded model
    prediction = model.predict(df)
    predicted_units = convert_log_to_units(prediction[0])
    
    return jsonify({'predicted_units': predicted_units})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

# import argparse
# import pandas as pd
# import numpy as np
# import pickle
# from src.utils import convert_log_to_units

# def load_model():
#     loaded_model = pickle.load(open('models/forecasting_model.pkl', 'rb'))
#     return loaded_model

# def predict_sales(input_data):
#     print(input_data)
#     model = load_model()
#     prediction = model.predict(input_data)
#     return convert_log_to_units(prediction)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Predict Sales')
#     parser.add_argument('--StoreCount', type=int, required=True, help='Store Count')
#     parser.add_argument('--ShelfCapacity', type=float, required=True, help='Shelf Capacity')
#     parser.add_argument('--PromoShelfCapacity', type=int, required=True, help='Promo Shelf Capacity')
#     parser.add_argument('--IsPromo', type=bool, required=True, help='Is Promo (True/False)')
#     parser.add_argument('--ItemNumber', type=int, required=True, help='Item Number')
#     parser.add_argument('--CategoryCode', type=int, required=True, help='Category Code')
#     parser.add_argument('--GroupCode', type=int, required=True, help='Group Code')
#     parser.add_argument('--month', type=int, required=True, help='Month')
#     parser.add_argument('--weekday', type=int, required=True, help='Weekday')
#     parser.add_argument('--UnitSales_-7', type=float, required=True, help='Unit Sales 7 days ago')
#     parser.add_argument('--UnitSales_-14', type=float, required=True, help='Unit Sales 14 days ago')
#     parser.add_argument('--UnitSales_-21', type=float, required=True, help='Unit Sales 21 days ago')

#     args = parser.parse_args()
#     input_data = pd.DataFrame([vars(args)])
#     prediction = predict_sales(input_data.values.tolist())
#     print(f"Predicted Sales: {prediction}")
