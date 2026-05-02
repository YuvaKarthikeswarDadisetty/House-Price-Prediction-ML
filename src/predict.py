import pandas as pd
import joblib


def predict_price(input_data: dict):
    print("\n🏠 Predicting House Price...\n")

    # Load artifacts
    model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    columns = joblib.load("models/columns.pkl")

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # -----------------------------
    # Encode categorical
    # -----------------------------
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    # Align columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)

    price = prediction[0]
    print(f"💰 Predicted Price: {price:,.2f}")

    return price