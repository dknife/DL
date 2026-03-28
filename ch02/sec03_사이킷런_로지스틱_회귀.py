"""
으뜸 딥러닝 — 02장 03절
사이킷런 로지스틱 회귀
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Binary classification: X_train (m, n), y_train (m,) -- 0 or 1
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)          # class prediction (0 or 1)
y_prob = model.predict_proba(X_test)    # probability prediction (m, 2)

print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
      target_names=["negative", "positive"]))
