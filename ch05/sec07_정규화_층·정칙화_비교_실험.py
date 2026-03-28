"""
으뜸 딥러닝 — 05장 07절
정규화 층·정칙화 비교 실험
"""

reg_configs = {
    'baseline':   dict(use_bn=False, drop_p=0.0, wd=0),
    'BN':         dict(use_bn=True,  drop_p=0.0, wd=0),
    'Dropout':    dict(use_bn=False, drop_p=0.3, wd=0),
    'WD':         dict(use_bn=False, drop_p=0.0, wd=1e-2),
    'BN+Drop+WD': dict(use_bn=True,  drop_p=0.3, wd=1e-2),
}

results_reg = {}
for name, cfg in reg_configs.items():
    model = make_mlp(use_bn=cfg['use_bn'], drop_p=cfg['drop_p'])
    apply_init(model, 'he')
    opt = optim.AdamW(model.parameters(), lr=1e-3,
                      weight_decay=cfg['wd'])
    results_reg[name] = run_experiment(model, opt)
    print(f"{name:12s} -> acc={results_reg[name]['acc'][-1]:.4f}")
# baseline     -> acc=0.8948
# BN           -> acc=0.8985
# Dropout      -> acc=0.8963
# WD           -> acc=0.8960
# BN+Drop+WD  -> acc=0.9025
