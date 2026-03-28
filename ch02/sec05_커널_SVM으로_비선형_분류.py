"""
으뜸 딥러닝 — 02장 05절
커널 SVM으로 비선형 분류
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_circles

# Linearly non-separable circular data
X, y = make_circles(n_samples=500, noise=0.1, random_state=42)

# RBF kernel SVM
rbf_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel='rbf', C=10, gamma=0.5))
])
rbf_svm.fit(X, y)
print(f"RBF accuracy: {rbf_svm.score(X, y):.3f}")

# Polynomial kernel SVM
poly_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel='poly', degree=4, C=10, coef0=2))
])
poly_svm.fit(X, y)
print(f"poly accuracy: {poly_svm.score(X, y):.3f}")
