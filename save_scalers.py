import joblib
from src.data_prep import load_scenario_a

data_a = load_scenario_a()
scaler_a = data_a  # wrong — we need to extract scaler properly
# from src.data_prep import load_scenario_a
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from pathlib import Path
# import joblib

# DATA_DIR = Path("data")

# df = pd.read_csv(DATA_DIR / "heart_2022_with_nans.csv")

# from src.data_prep import encode_scenario_a_raw
# sub, feature_names = encode_scenario_a_raw(df)
# X = sub.values.astype(float)

# scaler = StandardScaler()
# scaler.fit(X)

# joblib.dump(scaler, "models/scaler_scenario_a.pkl")
# print("Scenario A scaler saved")
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

DATA_DIR = Path("data")

df = pd.read_csv(DATA_DIR / "cardio_train.csv", sep=";")

feature_cols = [
    "age","gender","height","weight",
    "ap_hi","ap_lo","cholesterol","gluc",
    "smoke","alco","active"
]

sub = df[feature_cols].copy()
sub["age"] = (sub["age"] / 365.25).astype(float)

X = sub.values.astype(float)

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, "models/scaler_scenario_b.pkl")
print("Scenario B scaler saved")