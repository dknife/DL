"""
으뜸 딥러닝 — 02장 04절
사이킷런 가우시안 나이브 베이즈
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
# expected output: accuracy: 0.956
