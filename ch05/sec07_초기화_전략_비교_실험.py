"""
으뜸 딥러닝 — 05장 07절
초기화 전략 비교 실험
"""

def apply_init(model, strategy):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if strategy == 'zero':
                nn.init.zeros_(m.weight)
            elif strategy == 'large_random':
                nn.init.normal_(m.weight, std=1.0)
            elif strategy == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif strategy == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

results_init = {}
for name in ['zero', 'large_random', 'xavier', 'he']:
    model = make_mlp()
    apply_init(model, name)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    results_init[name] = run_experiment(model, opt)
    print(f"{name:14s} -> acc={results_init[name]['acc'][-1]:.4f}")
# zero           -> acc=0.1000   (symmetry breaking failure)
# large_random   -> acc=0.8523   (saturation: slow learning)
# xavier         -> acc=0.8871
# he             -> acc=0.8936
