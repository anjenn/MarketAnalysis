import tensorflow as tf # Keras
from catboost import CatBoostRegressor # CatBoost
import xgboost as xgb # XGBoost
import numpy as np
import joblib
import utils

# Load the data
DATA = utils.read_json("./Products/해바라기씨유.json")
review_ratios = [obj['REVIEW_RATIO'] for obj in DATA]
review_counts = [obj['REVIEW_RATIO'] for obj in DATA]

FIXED_ITEM_COUNT = 4

top_10_percentile_r = np.percentile(review_ratios, 90)
top_10_percentile_c = np.percentile(review_counts, 90)
top_10_values_r = [value for value in review_ratios if value >= top_10_percentile_r]
top_10_values_c = [value for value in review_ratios if value >= top_10_percentile_c]
average_top_10_r  = np.mean(top_10_values_r)
average_top_10_c  = np.mean(top_10_values_c)

# Load the trained model
keras_model = tf.keras.models.load_model("optimal_price_model.keras")
loaded_model = CatBoostRegressor()
catboost_model = loaded_model.load_model('catboost_model.bin')

scaler_X = joblib.load('scaler_x.pkl')

quantity = 5000 # change this!
inv_quantity = 1 / quantity
log_quantity = np.log1p(quantity)
total_quantity = FIXED_ITEM_COUNT * quantity

X_new = np.array([FIXED_ITEM_COUNT, quantity, average_top_10_r, average_top_10_c, inv_quantity, log_quantity, total_quantity]).reshape(1, -1)

X_new_scaled = scaler_X.transform(X_new)

# Make predictions
# y_pred_log = model.predict(X_new_scaled)
y_pred_log = catboost_model.predict(X_new_scaled)

y_pred = np.expm1(y_pred_log) 

# Output the predictions
print("Predicted Optimal Price:", y_pred) 