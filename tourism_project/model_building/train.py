from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

root = Path(__file__).parents[2]
Xtr, Xte = pd.read_csv(root/"Xtrain.csv"), pd.read_csv(root/"Xtest.csv")
ytr = pd.read_csv(root/"ytrain.csv").iloc[:,0]
yte = pd.read_csv(root/"ytest.csv").iloc[:,0]
cat = Xtr.select_dtypes("object").columns.tolist(); num = Xtr.columns.difference(cat).tolist()
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                         ("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                                           ("encode",OneHotEncoder(handle_unknown="ignore"))]), cat)])
pipe = Pipeline([("preprocessor",pre),("model",RandomForestClassifier(
    n_jobs=1, random_state=42, class_weight="balanced"))])
grid = GridSearchCV(pipe,{"model__n_estimators":[200,400],"model__max_depth":[6,12],
    "model__min_samples_leaf":[1,3]},
    scoring="recall",cv=3,n_jobs=1).fit(Xtr,ytr)
p = grid.best_estimator_.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
print("ROC-AUC",roc_auc_score(yte,p)); print(classification_report(yte,pred))
out=root/"tourism_project/deployment/tourism_model.joblib"; out.parent.mkdir(parents=True,exist_ok=True)
joblib.dump({"model":grid.best_estimator_,"threshold":.5,"features":Xtr.columns.tolist()},out)
