"""
으뜸 딥러닝 — 03장 02절
텐서의 shape·dtype·device
"""

x = torch.randn(2, 3, 4)   # 3-D tensor

print(x.shape)              # torch.Size([2, 3, 4])
print(x.dtype)              # torch.float32  (default)
print(x.device)             # cpu

# Specify dtype: integer, half precision, double precision
i = torch.zeros(3, dtype=torch.int64)
h = torch.ones(3,  dtype=torch.float16)   # half precision (saves GPU memory)
d = torch.ones(3,  dtype=torch.float64)   # double precision

# Specify device: move to cuda if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
