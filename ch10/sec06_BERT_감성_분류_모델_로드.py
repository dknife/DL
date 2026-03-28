"""
으뜸 딥러닝 — 10장 06절
BERT 감성 분류 모델 로드
"""

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
