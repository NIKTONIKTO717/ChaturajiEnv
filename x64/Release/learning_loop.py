import chaturajienv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import threading
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import time
import faulthandler
from network import *
from mcts import *
faulthandler.enable()

# === Hyperparameters ===
LEARNING_RATE = 3e-4       # Recommended: 1e-4 to 3e-4
BATCH_SIZE = 1024          # Recommended: 512-2048 (depending on memory)
L2_REG = 1e-4              # Weight decay (L2 regularization) to prevent overfitting
EPOCHS = 1000              # Adjust as needed

device_training = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device_acting = (torch.device('cpu'))
print(f"Training device: {device_training}, Acting device: {device_acting}")

net_acting = AlphaZeroNet((172,8,8), 4096, 4, 16, 16).to(device_acting)
net_training = AlphaZeroNet((172,8,8), 4096, 4, 16, 16).to(device_training)
optimizer = optim.Adam(net_training.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

torch.set_num_threads(24)  # Use all 24 CPU threads for inference
torch.set_num_interop_threads(4)


mcts = MCTS(net_acting, device_acting, 100000, 24, 100)

mcts.start()

while mcts.get_game_storage_size() < 10000:
    print(f"Game storage size: {mcts.get_game_storage_size()}")
    time.sleep(10)



# training loop
for i in range(1000):
    samples, policies, values = mcts.get_batch(1000)

    samples = torch.tensor(samples, dtype=torch.float32, device=device_training)
    policies = torch.tensor(policies, dtype=torch.float32, device=device_training)
    values = torch.tensor(values, dtype=torch.float32, device=device_training)

    # Forward pass
    policy_pred, value_pred = net_training(samples)

    policy_loss = F.cross_entropy(policy_pred, policies)
    value_loss = F.mse_loss(value_pred.squeeze(), values)

    total_loss = total_loss = policy_loss + value_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {i}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
