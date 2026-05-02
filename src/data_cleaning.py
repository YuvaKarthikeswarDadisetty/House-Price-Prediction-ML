import pandas as pd

def clean_data(df):
    print("\n🔧 Cleaning Data...\n")

    # Drop Id column (not useful for prediction)
    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print("✅ Cleaning Completed")

    return df