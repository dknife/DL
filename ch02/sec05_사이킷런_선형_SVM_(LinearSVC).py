"""
으뜸 딥러닝 — 02장 05절
사이킷런 선형 SVM (LinearSVC)
"""

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=2,
                           n_redundant=0, random_state=42)

# SVM is sensitive to scale -> StandardScaler is required
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    LinearSVC(C=1.0, loss='hinge', max_iter=5000))
])
svm_clf.fit(X, y)

print(f"train accuracy: {svm_clf.score(X, y):.3f}")
