"""
으뜸 딥러닝 — 10장 06절
모델 평가
"""

results = trainer.evaluate()
print(f"Test accuracy: {results['eval_accuracy']:.3f}")
# Test accuracy: 0.930
