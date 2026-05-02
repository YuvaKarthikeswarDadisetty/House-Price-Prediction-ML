import pandas as pd

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    print("✅ Data Loaded Successfully")
    print("Train Shape:", train.shape)
    print("Test Shape:", test.shape)

    return train, test