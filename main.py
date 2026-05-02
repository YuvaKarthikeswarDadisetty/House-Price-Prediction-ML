from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.eda import perform_eda
from src.feature_engineering import feature_engineering
from src.model_training import train_models
from src.predict import predict_price


if __name__ == "__main__":
    train, test = load_data()

    # Cleaning
    train = clean_data(train)
    train.to_csv("data/cleaned_train.csv", index=False)

    # EDA
    perform_eda(train)

    # Feature Engineering
    X, y = feature_engineering(train)
    X.to_csv("data/X_processed.csv", index=False)
    y.to_csv("data/y_processed.csv", index=False)

    # Training
    best_model = train_models(X, y)

    print("\n🏆 Best Model: XGBoost")

    # -----------------------------
    # Sample Prediction
    # -----------------------------
    sample_input = {
        'OverallQual': 7,
        'GrLivArea': 1800,
        'GarageCars': 2,
        'GarageArea': 500,
        'TotalBsmtSF': 900,
        '1stFlrSF': 1000,
        'FullBath': 2,
        'YearBuilt': 2005
    }

    predict_price(sample_input)

    print("\n🚀 Phase 7 Completed Successfully!")