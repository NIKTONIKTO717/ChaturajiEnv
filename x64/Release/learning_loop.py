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
from network import AlphaZeroNet
from mcts import MCTS
from evaluation import Eval
import tools
faulthandler.enable()

# === Hyperparameters ===
LEARNING_RATE = 1e-4       # Recommended: 1e-4 to 3e-4
BATCH_SIZE = 2048           # Recommended: 512-2048 (depending on memory)
L2_REG = 1e-4              # Weight decay (L2 regularization) to prevent overfitting
ITERATIONS = 1000          # Number of training iterations per evaluation
PASSES = 10                 # Adjust as needed
CPU_CORES = 22              # Number of CPU cores to use for MCTS
STORAGE_SIZE = 20000      # Number of games to store in the game storage
BUDGET = 200               # Number of MCTS simulations per move (800 in AlphaZero)

def main():
    device_training = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    device_acting = (torch.device('cpu'))
    print(f"Training device: {device_training}, Acting device: {device_acting}")

    # 20 * 8 + 4 + 4 + 4 + 1 = 173
    net_acting = AlphaZeroNet((173,8,8), 4096, 4, 16, 16).to(device_acting)
    net_training = AlphaZeroNet((173,8,8), 4096, 4, 16, 16).to(device_training)
    net_acting.eval()
    net_training.train()
    optimizer = optim.Adam(net_training.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

    mcts = MCTS(net_acting, STORAGE_SIZE, CPU_CORES, BUDGET)
    mcts.process_samples('mcts_games')
    mcts.use_model = True
    # mcts.start()

    sampling_ratio = np.array([0.25, 0.25, 0.25, 0.25])
    # training loop
    for l in range(10):
        for i in range(ITERATIONS): #from alpha zero - checkpoint every 1000 steps
            samples, policies, values = mcts.get_batch(BATCH_SIZE, sampling_ratio)
            samples = torch.tensor(samples, dtype=torch.float32, device=device_training)
            policies = torch.tensor(policies, dtype=torch.float32, device=device_training)
            values = torch.tensor(values, dtype=torch.float32, device=device_training)
            for e in range(PASSES):
                # Forward pass
                policy_pred, value_pred = net_training(samples)

                policy_loss = F.cross_entropy(policy_pred, policies)
                value_loss = F.mse_loss(value_pred.squeeze(), values)
                total_loss = policy_loss + value_loss

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if(e + 1 == PASSES and i % 100 == 0):
                    print(f"Iteration {i}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

        mcts.stop()

        #evaluate the network against random player
        evaluator = Eval(net_acting, net_training, CPU_CORES, 2)

        #evaluator.run_shifting(BUDGET, (2, -1, -1, -1)) # vanilla MCTS vs random
        #evaluator.run_shifting(BUDGET, (2, 0, 0, 0)) # acting_net vs vanilla MCTS
        moves_sum_0, score_sum_0, rewards_sum_0, rank_counts_0 = evaluator.run(BUDGET, (2, 1, 1, 1)) # acting_net vs training_net
        moves_sum_1, score_sum_1, rewards_sum_1, rank_counts_1 = evaluator.run(BUDGET, (1, 2, 1, 1)) # acting_net vs training_net
        moves_sum_2, score_sum_2, rewards_sum_2, rank_counts_2 = evaluator.run(BUDGET, (1, 1, 2, 1)) # acting_net vs training_net
        moves_sum_3, score_sum_3, rewards_sum_3, rank_counts_3 = evaluator.run(BUDGET, (1, 1, 1, 2)) # acting_net vs training_net
        #evaluator.run_shifting(300, (2, 1, 1, 1)) # acting_net vs training_net
        #moves_sum, score_sum, rewards_sum, rank_counts = evaluator.run_shifting(BUDGET, (2, 1, 1, 1)) # acting_net vs training_net

        rewards = np.array([rewards_sum_0[0], rewards_sum_1[1], rewards_sum_2[2], rewards_sum_3[3]])
        tau = 1.0
        rewards_exp = np.exp(-rewards / (tau * CPU_CORES * 2)) # normalize by number of games
        sampling_ratio = rewards_exp / np.sum(rewards_exp)
        print(f"Sampling ratio: {sampling_ratio}")
        
        #copy if better to acting network
        #if rewards_sum[0] > 0:
        if sum(rewards) > 0:
            tools.save_model(net_training)
            with open("log.txt", "a") as file:
                file.write(f"pickle: Training network gen {l} - {tools.get_model_hash(net_training)} pickled\n")
            tools.save_model(net_acting)
            net_acting.load_state_dict(net_training.state_dict())
        else:
            net_training.load_state_dict(net_acting.state_dict())

        mcts.start()
    mcts.stop()


if __name__ == '__main__':
    main()