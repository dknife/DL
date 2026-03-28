"""
으뜸 딥러닝 — 02장 04절
사이킷런 결정 트리 — 붓꽃 분류
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# limit max_depth to prevent overfitting
dec_tree = DecisionTreeClassifier(max_depth=3, criterion='gini',
                                  random_state=42)
dec_tree.fit(X_train, y_train)
y_pred = dec_tree.predict(X_test)

print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
# expected output: accuracy: 0.956

# feature importances used for splitting
for name, imp in zip(load_iris().feature_names,
                     dec_tree.feature_importances_):
    print(f"  {name}: {imp:.3f}")
