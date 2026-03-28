"""
으뜸 딥러닝 — 10장 06절
IMDb 데이터셋 로드
"""

from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset(features: ['text','label'], num_rows: 25000)
#     test:  Dataset(features: ['text','label'], num_rows: 25000)
# })
