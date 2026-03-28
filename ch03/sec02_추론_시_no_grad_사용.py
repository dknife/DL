"""
으뜸 딥러닝 — 03장 02절
추론 시 no\_grad 사용
"""

model.eval()                            # switch dropout/batchnorm to eval mode

with torch.no_grad():                   # disable gradient tracking
    predictions = model(test_data)      # saves memory and computation
