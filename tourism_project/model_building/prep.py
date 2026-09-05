import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

root = Path(__file__).parents[2]
df = pd.read_csv(root / "tourism_project/data/tourism.csv")
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")
X, y = df.drop(columns="ProdTaken"), df["ProdTaken"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
for name, obj in {"Xtrain.csv":Xtr,"Xtest.csv":Xte,"ytrain.csv":ytr,"ytest.csv":yte}.items():
    obj.to_csv(root / name, index=False)
print("Saved stratified train/test artifacts")
