import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


def evaluate_model(y_test, pred, model_name):
    print(f"\n📊 {model_name} Performance:")
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2 Score:", r2_score(y_test, pred))


def train_models(X, y):
    print("\n🚀 Training Models...\n")

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Models
    # -----------------------------
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # -----------------------------
    # Training
    # -----------------------------
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------
    evaluate_model(y_test, lr_pred, "Linear Regression")
    evaluate_model(y_test, rf_pred, "Random Forest")
    evaluate_model(y_test, xgb_pred, "XGBoost")

    # -----------------------------
    # Save Models
    # -----------------------------
    joblib.dump(lr, "models/linear_regression.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xgb, "models/xgboost.pkl")

    print("\n✅ Models Saved in /models folder")

    return xgb  # Best model