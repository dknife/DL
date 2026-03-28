"""
으뜸 딥러닝 — 13장 04절
RLHF PPO 학습 루프 (의사 코드)
"""

import torch

# Pre-trained models
sft_model = load_sft_model()          # frozen reference
policy = load_sft_model()             # trainable policy
reward_model = load_reward_model()    # frozen RM
beta = 0.1                            # KL penalty weight

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-6)

for prompts in dataloader:
    # Generate responses from current policy
    responses = policy.generate(prompts)

    # Compute reward and KL penalty
    rm_scores = reward_model(prompts, responses)
    log_pi = policy.log_prob(prompts, responses)
    log_pi_ref = sft_model.log_prob(prompts, responses)
    kl = log_pi - log_pi_ref           # per-token KL
    rewards = rm_scores - beta * kl.sum(dim=-1)

    # PPO update with clipped objective
    advantages = compute_advantages(rewards)
    ratio = torch.exp(log_pi - log_pi.detach())
    clipped = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
    loss = -torch.min(ratio * advantages,
                      clipped * advantages).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
