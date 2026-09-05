import pandas as pd
from pathlib import Path

path = Path(__file__).parents[1] / "data" / "tourism.csv"
required = {"CustomerID", "ProdTaken", "Age", "TypeofContact", "MonthlyIncome"}
df = pd.read_csv(path)
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")
if not set(df["ProdTaken"].dropna().unique()) <= {0, 1}:
    raise ValueError("ProdTaken must be binary")
print(f"Registered {len(df):,} rows and {df.shape[1]} columns from {path}")
