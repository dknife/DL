"""
으뜸 딥러닝 — 06장 01절
Conv2d 파라미터 수 확인
"""

total = sum(p.numel() for p in conv.parameters())
print(total)    # 896 = 32 * (3 * 3 * 3 + 1)
