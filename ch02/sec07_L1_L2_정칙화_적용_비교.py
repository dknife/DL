"""
으뜸 딥러닝 — 02장 07절
L1/L2 정칙화 적용 비교
"""

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Use polynomial features to induce overfitting
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

X = np.linspace(-3, 3, 60).reshape(-1, 1)
y = np.sin(X).ravel() + 0.3 * np.random.randn(60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for name, model in [("Ridge (L2)", Ridge(alpha=1.0)),
                    ("Lasso (L1)", Lasso(alpha=0.01))]:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=10)),
        ("scaler", StandardScaler()),
        ("reg", model)
    ])
    pipe.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse  = mean_squared_error(y_test,  pipe.predict(X_test))
    print(f"{name:15s}  train MSE={train_mse:.3f}  test MSE={test_mse:.3f}")
