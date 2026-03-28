"""
으뜸 딥러닝 — 05장 07절
옵티마이저 비교 실험
"""

optim_configs = {
    'SGD':      lambda p: optim.SGD(p, lr=0.01),
    'Momentum': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
    'Adam':     lambda p: optim.Adam(p, lr=1e-3),
    'AdamW':    lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-2),
}

results_opt = {}
for name, opt_fn in optim_configs.items():
    model = make_mlp()
    apply_init(model, 'he')
    results_opt[name] = run_experiment(model, opt_fn(model.parameters()))
    print(f"{name:10s} -> acc={results_opt[name]['acc'][-1]:.4f}")
# SGD        -> acc=0.8512
# Momentum   -> acc=0.8714
# Adam       -> acc=0.8936
# AdamW      -> acc=0.8948
