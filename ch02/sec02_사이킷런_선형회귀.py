"""
으뜸 딥러닝 — 02장 02절
사이킷런 선형회귀
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Training data (feature matrix must be 2-D)
X_train = x.reshape(-1, 1)   # shape (m, 1)
y_train = y                   # shape (m,)

model = LinearRegression()
model.fit(X_train, y_train)   # compute optimal params via normal equation

print(f"weight w: {model.coef_[0]:.4f}")
print(f"bias   b: {model.intercept_:.4f}")

y_pred = model.predict(X_train)
print(f"MSE: {mean_squared_error(y_train, y_pred):.4f}")
