import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import cross_val_score
import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = utils.read_json("./Products/해바라기씨유.json")
required_keys = {"ITEM_COUNT", "REVIEW_COUNT", "UNIT_PRICE", "QUANTITY"}

cleaned_data = [{k: v for k, v in item.items() if k in required_keys} for item in data]

df = pd.DataFrame(cleaned_data)

# REMOVING OUTLIERS
bins = np.linspace(df['QUANTITY'].min(), df['QUANTITY'].max(), num=10)  # 10 bins
labels = [f"Range {i}" for i in range(len(bins)-1)]  # Labels for each range
df['QUANTITY_RANGE'] = pd.cut(df['QUANTITY'], bins=bins, labels=labels, include_lowest=True)
price_median = df.groupby('QUANTITY_RANGE')['UNIT_PRICE'].median().reset_index()
price_median.columns = ['QUANTITY_RANGE', 'MEDIAN_PRICE']
df = df.merge(price_median, on='QUANTITY_RANGE', how='left')
df['PRICE_RATIO'] = df['UNIT_PRICE'] / df['MEDIAN_PRICE']
df_filtered = df[df['PRICE_RATIO'] <= 1.2]  # Filter out extreme outliers
df_filtered = df_filtered.drop(columns=['QUANTITY_RANGE', 'MEDIAN_PRICE', 'PRICE_RATIO'])
print(df_filtered)


# FEATURE ENGINEERING
df_filtered['INV_QUANTITY'] = 1 / df_filtered['QUANTITY']
df_filtered['LOG_QUANTITY'] = np.log1p(df_filtered['QUANTITY'])  # log(1 + quantity)
df_filtered['TOTAL_QUANTITY'] = df_filtered['ITEM_COUNT'] * df_filtered['QUANTITY']

X_raw = df_filtered[["ITEM_COUNT", "QUANTITY", "REVIEW_COUNT", "INV_QUANTITY", "LOG_QUANTITY", "TOTAL_QUANTITY"]].values
# X_raw  = df[["ITEM_COUNT", "QUANTITY", "REVIEW_COUNT"]].values
y = df_filtered["UNIT_PRICE"].values  # Predict total price

# DATA TILING
X_tiled = np.tile(X_raw, (4, 1))  # Duplicate the features
y_tiled = np.tile(y, 4)  # Duplicate the target variable

X_train, X_val, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# Normalization
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
y_log = np.log1p(y_train)  # Log transformation for training target
y_val_log = np.log1p(y_val)  # Log transformation for validation target

joblib.dump(scaler_X, 'scaler_x.pkl')  # Save the scaler for future use

# df.describe()

# # Feedforward Neural Network (FNN) Model
# model = models.Sequential([
#     layers.Input(shape=(X_train_scaled.shape[1],)),
#     # layers.Dense(64, activation="relu"),
#     # layers.Dense(128, activation="relu"),
#     # layers.Dense(64, activation="relu"),
#     layers.Dense(128, activation="relu", kernel_regularizer=l2(0.0001)),
#     layers.Dropout(0.2), # Add dropout layer to prevent overfitting
#     layers.Dense(64, activation="relu", kernel_regularizer=l2(0.0001)),
#     # layers.Dropout(0.2),
#     layers.Dense(32, activation="relu", kernel_regularizer=l2(0.0001)),
#     # layers.Dropout(0.2),
#     layers.Dense(1)  # Output: Predicted price
# ])
# model.compile(optimizer=Adam(learning_rate=0.0005), loss="mae")
# early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
# model.fit(
#     X_train_scaled, y_log,
#     epochs=100, batch_size=16,
#     validation_data=(X_val_scaled, y_val_log),
#     callbacks=[early_stop]) # Adjust epochs, batch_size, and validation_split as needed

# model.save("optimal_price_model.keras")  # Save the model in HDF5 format

# # Mean Absolute Error (MAE): 28.593538556780153
# # Root Mean Squared Error (RMSE): 44.2110070907123
# # Indices of residual outliers: []

# # #############################################################################
# import xgboost as xgb


# # XGBoost Model
# model = xgb.XGBRegressor(
#     n_estimators=2000,  # Number of boosting rounds
#     learning_rate=0.005,  # Learning rate
#     max_depth=5,  # Maximum depth of trees
#     min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child
#     subsample=0.8,  # Fraction of samples used in each boosting round
#     colsample_bytree=0.8,  # Fraction of features used for each boosting round
#     objective='reg:squarederror',  # Regression objective
#     n_jobs=-1  # Use all available CPU threads
# )

# # Train the model
# model.fit(X_train_scaled, y_log)

# # Mean Absolute Error (MAE): 9.128461565290198
# # Root Mean Squared Error (RMSE): 22.163604236348917
# # Indices of residual outliers: [9]
# # # #############################################################################

#############################################################################
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score


# CatBoost model
model = CatBoostRegressor(
    iterations=1200,  # Number of boosting rounds
    learning_rate=0.01,  # Learning rate
    depth=7,  # Depth of trees
    loss_function='RMSE',  # Loss function to minimize
    cat_features=[],  # Specify categorical features if any (empty list if none)
    verbose=100, # Displaying progress every 100 iterations
    early_stopping_rounds=50,
    l2_leaf_reg=3.0
)

# Train the model
model.fit(X_train_scaled, y_log)
model.get_feature_importance()

# Mean Absolute Error (MAE): 10.878789415160254
# Root Mean Squared Error (RMSE): 16.674119917892348
cv_scores = cross_val_score(model, X_train_scaled, y_log, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-Validation MAE: {-np.mean(cv_scores)}")

model.save_model('catboost_model.bin')

# #############################################################################
# #############################################################################

# # Evaluate the model
y_pred_original = np.expm1(model.predict(X_val_scaled)).ravel()  # Predicted unit price in original scale
y_val_original = np.expm1(y_val_log).ravel()  # Actual unit price in original scale

mae = mean_absolute_error(y_val_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))

# Plot Actual vs Predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_val_original, y_pred_original, alpha=0.6, c='blue')
plt.plot([y_val_original.min(), y_val_original.max()], [y_val_original.min(), y_val_original.max()], 'k--', lw=2)
plt.xlabel('Actual Unit Price')
plt.ylabel('Predicted Unit Price')
plt.title('Actual vs Predicted Unit Price')
plt.grid(True)
plt.show()

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# #############################################################################
# Residuals Calculation
residuals = np.abs(y_val_original - y_pred_original)  # Calculate absolute residuals

# You can plot the residuals to visually inspect them
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (|Actual - Predicted|)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Optionally, you can calculate a threshold to detect outliers in residuals
threshold = np.mean(residuals) + 3 * np.std(residuals)  # 3 standard deviations as threshold
outliers = residuals > threshold  # Detect residuals above the threshold

# Display outliers (if any)
outlier_indices = np.where(outliers)[0]
print(f"Indices of residual outliers: {outlier_indices}")

# You can remove outliers based on the residuals (if needed)
X_val_no_outliers = X_val_scaled[~outliers]  # Remove outliers from validation set features
y_val_no_outliers = y_val_original[~outliers]  # Remove outliers from the target variable

# Optionally, you can evaluate the model performance without outliers
# model.evaluate(X_val_no_outliers, y_val_no_outliers) # Only for Keras models

# Plot Actual vs Predicted Prices with Outliers Removed (if you removed them)
y_pred_no_outliers = np.expm1(model.predict(X_val_no_outliers))
plt.figure(figsize=(8, 6))
plt.scatter(y_val_no_outliers, y_pred_no_outliers, alpha=0.6, c='blue')
plt.plot([y_val_no_outliers.min(), y_val_no_outliers.max()], [y_val_no_outliers.min(), y_val_no_outliers.max()], 'k--', lw=2)
plt.xlabel('Actual Unit Price')
plt.ylabel('Predicted Unit Price')
plt.title('Actual vs Predicted Unit Price (Outliers Removed)')
plt.grid(True)
plt.show()
