from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent / "data-animal" / "animal_dataset.csv"

# Adjusted features to match a Ben 10 dataset context instead of animal features
FEATURES = [
    "Power_Level",
]

def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    X_raw = df[FEATURES].to_numpy(dtype="float32")  
    X = StandardScaler().fit_transform(X_raw).astype("float32")  

    return {"X": X, "X_raw": X_raw, "df": df, "features": FEATURES}

if __name__ == "__main__":
    data = load_data()
    print("size data :", data["X"].shape)
    print("mean after scale (should be close to 0) :", data["X"].mean(axis=0).round(3))