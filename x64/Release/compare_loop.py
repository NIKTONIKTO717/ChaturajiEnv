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
from comparison import Compare
import tools
import pickle
faulthandler.enable()

CPU_CORES = 4              # Number of CPU cores to use for MCTS
BUDGET = 10                # Number of MCTS simulations per move (800 in AlphaZero)
EVAL_GAMES = 2              # Number of games to evaluate per CPU core    

# folders of 4 different models
AGENTS_DIR = [
    'model_0',
    'model_1',
    'model_2',
    'model_3'
]
# names of pickle files, same for each agent
PICKLE_FILES = [
    'acting_net_0',
    'acting_net_1',
    'acting_net_2',
    #'acting_net_3',
    #'acting_net_4',
    #'acting_net_5',
    #'acting_net_6',
    #'acting_net_7',
    #'acting_net_8',
    #'acting_net_9',
    #'acting_net_10',
    #'acting_net_11',
    #'acting_net_12',
]

def main():
    rewards_gens = []
    for file in PICKLE_FILES:
        start_time = time.time()
        networks = []
        rewards_sum_sum = [0,0,0,0]
        for i in range(4):
            networks.append(AlphaZeroNet((173,8,8), 4096, 8, 32, 32))
            networks[i].load_state_dict(torch.load(AGENTS_DIR[i] + '/' + file + '.pickle'))
        compare = Compare(networks, AGENTS_DIR, file, CPU_CORES, EVAL_GAMES)
        for setup in [(0,1,2,3), (0,1,3,2), (0,2,1,3), (0,2,3,1), (0,3,1,2), (0,3,2,1)]:
            print('Comparing agents:', setup)
            moves_sum, score_sum, rewards_sum, rank_counts = compare.run_shifting(BUDGET, setup)
            rewards_sum_reordered = sorted(zip(setup, rewards_sum), key=lambda x: x[0])
            rewards_sum_reordered = [x[1] for x in rewards_sum_reordered]
            print('Rewards sum:', rewards_sum_reordered)
            rewards_sum_sum = np.add(rewards_sum_sum, rewards_sum_reordered)
            print(rewards_sum_reordered)
            print('New sum:')
            print(rewards_sum_sum)
            rewards_gens.append(rewards_sum_reordered)
        print('Time taken:', time.time() - start_time)

    #plot graph with 4 lines, one for each agent
    plt.plot(rewards_gens[0], label='Agent 1')
    plt.plot(rewards_gens[1], label='Agent 2')
    plt.plot(rewards_gens[2], label='Agent 3')
    plt.plot(rewards_gens[3], label='Agent 4')
    plt.xlabel('Games')
    plt.ylabel('Rewards')
    plt.title('Rewards of each agent')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()