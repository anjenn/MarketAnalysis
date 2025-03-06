import os
import tensorflow as tf # Keras
from catboost import CatBoostRegressor # CatBoost
import xgboost as xgb # XGBoost
import numpy as np
import json
import joblib
import utils

product_path = "./Products"

# Load the data
def load_files(PRODUCT_NAME):
    data = []
    for root, _, files in os.walk(product_path):
        for file in files:
            file_name = os.path.splitext(file)[0]  # Remove file extension

            if PRODUCT_NAME in file_name:
                try:
                    with open(product_path + '/' + file_name + '.json', 'r', encoding='utf-8') as f:  # Open the file
                        data = json.load(f)  # Load JSON data
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_name}")
    return data


def UserInt():
    # Predefined
    PRODUCT_NAME = input("Please enter product name:") or "해바라기기"
    DATA = load_files(PRODUCT_NAME)
    FIXED_ITEM_COUNT = int(input("Please enter item count:")) or 4
    QUANTITY = int(input("Please enter quantity:")) or 1000
    MODEL_INPUT = input("Please enter model type: ") or "XGB"
    MODEL_TYPE = None

    # Caculations
    review_ratios = [obj['REVIEW_RATIO'] for obj in DATA]
    review_counts = [obj['REVIEW_COUNT'] for obj in DATA]
    inv_quantity = 1 / QUANTITY
    log_quantity = np.log1p(QUANTITY)
    total_quantity = FIXED_ITEM_COUNT * QUANTITY

    # Filtering outliers
    top_10_percentile_r = np.percentile(review_ratios, 90)
    top_10_percentile_c = np.percentile(review_counts, 90)
    top_10_values_r = [value for value in review_ratios if value >= top_10_percentile_r]
    top_10_values_c = [value for value in review_ratios if value >= top_10_percentile_c]
    average_top_10_r  = np.mean(top_10_values_r)
    average_top_10_c  = np.mean(top_10_values_c)

    if MODEL_INPUT == 'XGB':
        xgb_loaded = xgb.XGBRegressor()
        xgb_loaded.load_model('xgb_model.json')
        MODEL_TYPE = xgb_loaded
    if MODEL_INPUT == 'CAT':
        catBoost_loaded = CatBoostRegressor()
        MODEL_TYPE = catBoost_loaded.load_model('catboost_model.bin')
    if MODEL_INPUT == 'KERAS':
        MODEL_TYPE = tf.keras.models.load_model("optimal_price_model.keras")

    scaler_X = joblib.load('scaler_x.pkl')
    X_new = np.array([FIXED_ITEM_COUNT, QUANTITY, average_top_10_r, average_top_10_c, inv_quantity, log_quantity, total_quantity]).reshape(1, -1)
    X_new_scaled = scaler_X.transform(X_new)

    # Make predictions
    # y_pred_log = model.predict(X_new_scaled)
    y_pred_log = MODEL_TYPE.predict(X_new_scaled)

    y_pred = np.expm1(y_pred_log) 
    # Output the predictions
    print("Predicted Optimal Price:", y_pred) 


UserInt()