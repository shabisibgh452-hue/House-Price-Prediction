# training.py

import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import preprocessed data
from preprocessing import X_scaled, y

# Split Data (85% Training, 15% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.15,
    random_state=42
)

# ===============================
# Linear Regression
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
print("========== Linear Regression ==========")
print("MAE :", mean_absolute_error(y_test, lr_pred))
print("MSE :", mean_squared_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2 Score:", r2_score(y_test, lr_pred))

# Save Linear Regression Model
joblib.dump(lr_model, "linear_regression_model.pkl")

# ===============================
# Random Forest Regressor
# ===============================
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n========== Random Forest Regressor ==========")
print("MAE :", mean_absolute_error(y_test, rf_pred))
print("MSE :", mean_squared_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))

# Save Random Forest Model
joblib.dump(rf_model, "random_forest_model.pkl")

print("\nModel Training Completed Successfully!")