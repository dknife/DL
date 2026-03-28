"""
으뜸 딥러닝 — 03장 03절
훈련용 데이터 증강 파이프라인 예시
"""

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),        # random horizontal flip
    T.RandomCrop(32, padding=4),          # random crop with 4px padding
    T.ColorJitter(brightness=0.2,
                  contrast=0.2),          # random brightness/contrast
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2023, 0.1994, 0.2010)),
])
