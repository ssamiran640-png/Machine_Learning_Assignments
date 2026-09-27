import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Area_sqft": [650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000],
    "Bedrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    "Price_Lakhs": [25, 32, 38, 45, 52, 58, 65, 72, 78, 85]
}
df = pd.DataFrame(data)
X = df[["Area_sqft", "Bedrooms"]]
y = df["Price_Lakhs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("coefficients:", model.coef_)
print("intercept:", model.intercept_)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE =", mae)
print("MSE =", mse)
print("RMSE =", rmse)
print("R2 =", r2)