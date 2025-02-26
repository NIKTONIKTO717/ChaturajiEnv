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
    mcts.start()

    # training loop
    for l in range(10):
        for i in range(ITERATIONS): #from alpha zero - checkpoint every 1000 steps
            samples, policies, values = mcts.get_batch(BATCH_SIZE)
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
        #control group
        evaluator.run(BUDGET, (0, -1, -1, -1)) # vanilla MCTS vs random
        evaluator.run(BUDGET, (-1, 0, -1, -1)) # vanilla MCTS vs random
        evaluator.run(BUDGET, (-1, -1, 0, -1)) # vanilla MCTS vs random
        evaluator.run(BUDGET, (-1, -1, -1, 0)) # vanilla MCTS vs random
        evaluator.run(BUDGET, (2, -1, -1, -1)) # acting_net vs random
        evaluator.run(BUDGET, (-1, 2, -1, -1)) # acting_net vs random
        evaluator.run(BUDGET, (-1, -1, 2, -1)) # acting_net vs random
        evaluator.run(BUDGET, (-1, -1, -1, 2)) # acting_net vs random
        evaluator.run(BUDGET, (2, 0, 0, 0)) # acting_net vs vanilla MCTS
        evaluator.run(BUDGET, (0, 2, 0, 0)) # acting_net vs vanilla MCTS
        evaluator.run(BUDGET, (0, 0, 2, 0)) # acting_net vs vanilla MCTS
        evaluator.run(BUDGET, (0, 0, 0, 2)) # acting_net vs vanilla MCTS
        evaluator.run(BUDGET, (2, 1, 1, 1)) # acting_net vs training_net
        evaluator.run(BUDGET, (1, 2, 1, 1)) # acting_net vs training_net
        evaluator.run(BUDGET, (1, 1, 2, 1)) # acting_net vs training_net
        evaluator.run(BUDGET, (1, 1, 1, 2)) # acting_net vs training_net

        #TODO: add logic to compare acting network with training network
        #copy better network to acting network
        tools.save_model(net_acting)
        with open("log.txt", "a") as file:
            file.write(f"pickle: Actor network gen {l} - {tools.get_model_hash(net_acting)} pickled\n")
        tools.save_model(net_acting)
        net_acting.load_state_dict(net_training.state_dict())

        mcts.start()
    mcts.stop()


if __name__ == '__main__':
    main()