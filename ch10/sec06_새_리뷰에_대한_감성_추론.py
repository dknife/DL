"""
으뜸 딥러닝 — 10장 06절
새 리뷰에 대한 감성 추론
"""

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer
)

print(classifier("The acting was brilliant and the story was gripping."))
# [{'label': 'POSITIVE', 'score': 0.9987}]

print(classifier("Terrible movie. Complete waste of time."))
# [{'label': 'NEGATIVE', 'score': 0.9994}]
