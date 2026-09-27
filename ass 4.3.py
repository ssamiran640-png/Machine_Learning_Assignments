import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = {
    "Area_sqft": [650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000],
    "Price_Lakhs": [25, 32, 38, 45, 52, 58, 65, 72, 78, 85]
}
df = pd.DataFrame(data)

X = df[["Area_sqft"]].values
y = df["Price_Lakhs"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred1 = lr.predict(X_test)
r2_1 = r2_score(y_test, pred1)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
pred2 = pr.predict(X_test_poly)
r2_2 = r2_score(y_test, pred2)

print("Linear R2:", r2_1)
print("Polynomial R2:", r2_2)

if r2_2 > r2_1:
    print("poly regression did better here")
else:
    print("linear regression was enough, poly didn't help much")