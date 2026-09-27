import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
y = df["Purchased"]

numeric_features = ["Age", "Income"]
categorical_features = ["City"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
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

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("--- Original Dataset ---")
print(df)
print("\n--- Processed Feature Matrix Shape ---")
print(X_processed.shape)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])