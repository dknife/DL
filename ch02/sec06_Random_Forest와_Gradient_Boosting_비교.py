"""
으뜸 딥러닝 — 02장 06절
Random Forest와 Gradient Boosting 비교
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest: 100 trees, random sqrt(n) features per split
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# Gradient Boosting: 100 trees, depth 3, learning rate 0.1
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
print(f"Gradient Boosting accuracy: {accuracy_score(y_test, gb.predict(X_test)):.4f}")

# Feature importances (Random Forest)
for name, imp in zip(load_iris().feature_names, rf.feature_importances_):
    print(f"  {name}: {imp:.3f}")
