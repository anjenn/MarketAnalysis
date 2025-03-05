import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import utils

data = utils.read_json("./Products/해바라기씨유.json")
required_keys = {"ITEM_COUNT", "REVIEW_COUNT", "UNIT_PRICE", "QUANTITY"}

cleaned_data = [{k: v for k, v in item.items() if k in required_keys} for item in data]

df = pd.DataFrame(cleaned_data)
# df_scaled = df.copy()  # Create a copy to avoid modifying the original dataframe
# df_scaled[["ITEM_COUNT", "UNIT_PRICE", "QUANTITY", "REVIEW_COUNT"]] = scaler.fit_transform(df[["ITEM_COUNT", "UNIT_PRICE", "QUANTITY", "REVIEW_COUNT"]])

df['INV_QUANTITY'] = 1 / df['QUANTITY']
df['LOG_QUANTITY'] = np.log1p(df['QUANTITY'])  # log(1 + quantity)

X_raw = df[["ITEM_COUNT", "QUANTITY", "REVIEW_COUNT", "INV_QUANTITY", "LOG_QUANTITY"]].values
# X_raw  = df[["ITEM_COUNT", "QUANTITY", "REVIEW_COUNT"]].values
y = df["UNIT_PRICE"].values  # Predict total price

X_train, X_val, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler() # Normalization

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

y_log = np.log1p(y_train)  # Log transformation for training target
y_val_log = np.log1p(y_val)  # Log transformation for validation target

# df.describe()

print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of X_val_scaled: {X_val_scaled.shape}")
print(f"Shape of y_log: {y_log.shape}")
print(f"Shape of y_val_log: {y_val_log.shape}")

model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    # layers.Dense(64, activation="relu"),
    # layers.Dense(128, activation="relu"),
    # layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
    # layers.Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    layers.Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
    layers.Dense(1)  # Output: Predicted price
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(
    X_train_scaled, y_log,
    epochs=100, batch_size=16,
    validation_data=(X_val_scaled, np.log1p(y_val)),
    callbacks=[early_stop])

# Adjust epochs, batch_size, and validation_split as needed

model.save("optimal_price_model.keras")  # Save the model in HDF5 format
joblib.dump(scaler_X, 'scaler_x.pkl')  # Save the scaler for future use



# Evaluation
y_pred_original = np.expm1(model.predict(X_val_scaled))  # Predicted unit price in original scale
y_val_original = np.expm1(y_val_log)  # Actual unit price in original scale

# Plot Actual vs Predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_val_original, y_pred_original, alpha=0.6, c='blue')
plt.plot([y_val_original.min(), y_val_original.max()], [y_val_original.min(), y_val_original.max()], 'k--', lw=2)
plt.xlabel('Actual Unit Price')
plt.ylabel('Predicted Unit Price')
plt.title('Actual vs Predicted Unit Price')
plt.grid(True)
plt.show()

# Optionally, you can also calculate MAE and RMSE to quantify the performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_val_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# #############################################################################
# Residuals Analysis
residuals = y_val_original - y_pred_original

# Plot Residuals Distribution (Histogram and Kernel Density Estimate)
plt.figure(figsize=(8, 6))

# Histogram for Residuals Distribution
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Optionally, use a KDE plot for a smoother residuals distribution view
plt.figure(figsize=(8, 6))
sns.kdeplot(residuals, fill=True)
plt.title('Residuals Distribution (KDE)')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Check for specific outliers (high residuals)
outliers = residuals[np.abs(residuals) > np.percentile(np.abs(residuals), 95)]  # For example, look for top 5% residuals
print("Outliers (high residuals):", outliers)