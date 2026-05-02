import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df):
    print("\n📊 Performing EDA...\n")

    os.makedirs("images", exist_ok=True)

    # -----------------------------
    # 1. SalePrice Distribution
    # -----------------------------
    plt.figure()
    sns.histplot(df['SalePrice'], kde=True)
    plt.title("SalePrice Distribution")
    plt.savefig("images/01_price_distribution.png")
    plt.close()

    # -----------------------------
    # 2. Log Transformation
    # -----------------------------
    plt.figure()
    sns.histplot(df['SalePrice'], kde=True)
    plt.title("SalePrice (Skewed Distribution)")
    plt.savefig("images/02_price_skewed.png")
    plt.close()

    # -----------------------------
    # 3. Correlation Heatmap
    # -----------------------------
    plt.figure(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("images/03_heatmap.png")
    plt.close()

    # -----------------------------
    # 4. Top Features vs Price
    # -----------------------------
    features = [
        'OverallQual',
        'GrLivArea',
        'GarageCars',
        'GarageArea',
        'TotalBsmtSF',
        '1stFlrSF',
        'FullBath',
        'YearBuilt'
    ]

    for feature in features:
        if feature in df.columns:
            plt.figure()
            sns.scatterplot(x=df[feature], y=df['SalePrice'])
            plt.title(f"{feature} vs SalePrice")
            plt.savefig(f"images/{feature}_vs_price.png")
            plt.close()

    # -----------------------------
    # 5. Boxplot (Outliers)
    # -----------------------------
    plt.figure()
    sns.boxplot(x=df['SalePrice'])
    plt.title("SalePrice Boxplot")
    plt.savefig("images/04_boxplot.png")
    plt.close()

    print("✅ EDA Completed. Images saved in /images")