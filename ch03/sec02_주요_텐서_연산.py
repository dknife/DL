"""
으뜸 딥러닝 — 03장 02절
주요 텐서 연산
"""

A = torch.randn(3, 4)
B = torch.randn(4, 2)

# Matrix multiply (matmul): cols of A must equal rows of B
C = torch.matmul(A, B)       # (3,4) @ (4,2) -> (3,2)
C = A @ B                    # same result, shorter notation

# Element-wise operations (shapes must match)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(x + y)                 # tensor([5., 7., 9.])
print(x * y)                 # element-wise product: tensor([ 4., 10., 18.])
print(x ** 2)                # element-wise square: tensor([1., 4., 9.])

# Reduction operations
print(x.sum())               # tensor(6.)
print(x.mean())              # tensor(2.)
print(x.max())               # tensor(3.)
