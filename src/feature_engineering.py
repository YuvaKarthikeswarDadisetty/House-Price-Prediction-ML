import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


def feature_engineering(df):
    print("\n⚙️ Starting Feature Engineering...\n")

    y = df['SalePrice']
    X = df.drop('SalePrice', axis=1)

    # -----------------------------
    # Encode Categorical Features
    # -----------------------------
    cat_cols = X.select_dtypes(include=['object']).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    joblib.dump(encoders, "models/encoders.pkl")

    print("✅ Categorical Encoding Done")

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    # Save columns for prediction alignment
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    print("✅ Feature Scaling Done")
    print("✅ Artifacts Saved (scaler, encoders, columns)")

    return pd.DataFrame(X_scaled, columns=X.columns), y