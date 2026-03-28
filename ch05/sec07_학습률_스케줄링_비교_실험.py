"""
으뜸 딥러닝 — 05장 07절
학습률 스케줄링 비교 실험
"""

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR

n_epochs = 30
steps_per_epoch = len(train_loader)
total_steps = n_epochs * steps_per_epoch

def warmup_cosine(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda

sched_configs = {
    'fixed':   lambda opt: None,
    'StepLR':  lambda opt: StepLR(opt, step_size=10, gamma=0.1),
    'Cosine':  lambda opt: CosineAnnealingLR(opt, T_max=n_epochs),
    'WarmCos': lambda opt: LambdaLR(
        opt, warmup_cosine(steps_per_epoch * 2, total_steps)),
}

results_sched = {}
for name, sched_fn in sched_configs.items():
    model = make_mlp(use_bn=True, drop_p=0.3)
    apply_init(model, 'he')
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = sched_fn(opt)
    # StepLR and CosineAnnealingLR are epoch-level
    step_based = (name == 'WarmCos')
    if step_based:
        results_sched[name] = run_experiment(
            model, opt, n_epochs, scheduler=sched)
    else:
        criterion = nn.CrossEntropyLoss()
        history = {'loss': [], 'acc': []}
        for epoch in range(n_epochs):
            loss = train_one_epoch(model, train_loader, criterion, opt)
            acc  = evaluate(model, test_loader)
            history['loss'].append(loss)
            history['acc'].append(acc)
            if sched is not None:
                sched.step()          # epoch-level update
        results_sched[name] = history
    print(f"{name:8s} -> acc={results_sched[name]['acc'][-1]:.4f}")
# fixed    -> acc=0.9025
# StepLR   -> acc=0.9048
# Cosine   -> acc=0.9062
# WarmCos  -> acc=0.9071
