import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Area_sqft": [650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000],
    "Price_Lakhs": [25, 32, 38, 45, 52, 58, 65, 72, 78, 85]
}
df = pd.DataFrame(data)
print(df)

X = df[["Area_sqft"]].values
y = df["Price_Lakhs"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("slope =", model.coef_[0])
print("intercept =", model.intercept_)
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 score:", round(r2, 4))

plt.scatter(X, y, color="blue", label="actual")
plt.plot(X, model.predict(X), color="red", label="predicted line")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("House Area vs Price")
plt.legend()
plt.grid(True)
plt.show()