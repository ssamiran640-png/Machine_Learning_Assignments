import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data = {
    "Age": [22, 25, None, 30, 28, None, 35],
    "Salary": [30000, 45000, 50000, None, 60000, 55000, None],
    "Department": ["IT", "HR", "IT", "Finance", None, "HR", "Finance"],
    "Years_of_Experience": [1, 3, None, 6, 4, 5, 10]
}
df = pd.DataFrame(data)

print("--- Synthetic Dataset with Missing Values ---")
print(df)
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

numeric_features = ["Age", "Salary", "Years_of_Experience"]
categorical_features = ["Department"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(df)

print("\n--- Processed Feature Matrix Shape ---")
print(X_processed.shape)