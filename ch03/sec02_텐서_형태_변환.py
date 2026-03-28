"""
으뜸 딥러닝 — 03장 02절
텐서 형태 변환
"""

x = torch.randn(4, 6)

# view: reinterpret shape freely when total elements match (shares memory)
y = x.view(24)               # (4,6) -> (24,)  flatten to 1-D
y = x.view(2, 3, 4)          # (4,6) -> (2,3,4)
y = x.view(-1, 3)            # -1 is inferred automatically: (8,3)

# reshape: similar to view but also works on non-contiguous tensors (may copy)
y = x.reshape(3, 8)

# squeeze / unsqueeze: remove/add size-1 dimensions
z = torch.randn(1, 5, 1)
print(z.squeeze().shape)     # (5,)  -- remove all size-1 dims
print(z.unsqueeze(0).shape)  # (1,1,5,1) -- add dim at position 0
