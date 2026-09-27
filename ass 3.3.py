import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data = {
    "Age": [20, 21, None, 23, 24],
    "Income": [25000, 30000, 28000, None, 40000],
    "City": ["Kolkata", "Durgapur", "Kolkata", "Asansol", "Durgapur"],
    "Purchased": [0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df.drop("Purchased", axis=1)
numeric_features = ["Age", "Income"]
categorical_features = ["City"]

standard_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

standard_preprocessor = ColumnTransformer(
    transformers=[
        ("num", standard_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

minmax_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

minmax_preprocessor = ColumnTransformer(
    transformers=[
        ("num", minmax_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_standard = standard_preprocessor.fit_transform(X)
X_minmax = minmax_preprocessor.fit_transform(X)

print("--- StandardScaler Output (Age, Income columns) ---")
print(X_standard[:, :2])
print("Range: Min =", X_standard[:, :2].min(), "| Max =", X_standard[:, :2].max())
print("Mean ~0, Std ~1 (z-score normalization)")

print("\n--- MinMaxScaler Output (Age, Income columns) ---")
print(X_minmax[:, :2])
print("Range: Min =", X_minmax[:, :2].min(), "| Max =", X_minmax[:, :2].max())
print("Values scaled strictly between 0 and 1")

print("\n--- Observation ---")
print("StandardScaler centers data around mean 0 with unit variance, allowing negative values.")
print("MinMaxScaler compresses data into a fixed [0, 1] range, preserving relative distances but sensitive to outliers.")