import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import faulthandler
from network import AlphaZeroNet
from comparison import Compare
import pickle
from evaluation import Eval
faulthandler.enable()

CPU_CORES = 11                  # Number of CPU cores to use for MCTS
BUDGET = [100, 400, 1600, 6400]   # Number of MCTS simulations per move (800 in AlphaZero)
EVAL_GAMES = 2                  # Number of games to evaluate per CPU core    

def main():
    rewards_sum_sum = [0,0,0,0]
    score_sum_sum = [0,0,0,0]
    networks = [None, None, None, None]
    compare = Compare(networks, networks, 'mcts_only', CPU_CORES, EVAL_GAMES)
    for setup in [(0,1,2,3), (0,1,3,2), (0,2,1,3), (0,2,3,1), (0,3,1,2), (0,3,2,1)]:
            print('Comparing agents:', setup)
            moves_sum, score_sum, rewards_sum, rank_counts = compare.run_shifting(BUDGET, setup)
            rewards_sum_reordered = [rewards_sum[setup.index(i)] for i in range(4)]
            score_sum_reordered = [score_sum[setup.index(i)] for i in range(4)]
            score_sum_sum = np.add(score_sum_sum, score_sum_reordered)
            rewards_sum_sum = np.add(rewards_sum_sum, rewards_sum_reordered)
    print('Rewards:', rewards_sum_sum)
    print('Scores:', score_sum_sum)

if __name__ == '__main__':
    main()