"""
으뜸 딥러닝 — 13장 05절
CartPole 환경 설정
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n              # 2
print(f"State dim: {state_dim}, Action dim: {action_dim}")
# State dim: 4, Action dim: 2
