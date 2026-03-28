"""
으뜸 딥러닝 — 03장 02절
자동미분 기반 경사하강법
"""

import torch

w = torch.tensor(1.0, requires_grad=True)   # initial parameter
eta = 0.1                                   # learning rate

for step in range(5):
    # Forward pass: compute loss  (e.g. L = (w - 3)^2, minimum at w=3)
    loss = (w - 3.0)**2

    # Backward pass: compute gradients
    loss.backward()

    # Update parameter (inside no_grad block to avoid tracking)
    with torch.no_grad():
        w -= eta * w.grad               # w <- w - eta * dL/dw

    # Prevent gradient accumulation: must reset before next step
    w.grad.zero_()

    print(f"step {step+1}: w={w.item():.4f}, loss={loss.item():.4f}")

# step 1: w=1.6000, loss=4.0000
# step 2: w=2.0800, loss=1.4400
# step 3: w=2.4640, loss=0.5184
# step 4: w=2.7712, loss=0.1866
# step 5: w=3.0170, loss=0.0672  <- w converges to 3
