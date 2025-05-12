import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import faulthandler
from network import AlphaZeroNet
from comparison import Compare
from evaluation import Eval
import pickle
faulthandler.enable()

CPU_CORES = 22              # Number of CPU cores to use for MCTS
BUDGET = 400                # Number of MCTS simulations per move (800 in AlphaZero)
EVAL_GAMES = 4              # Number of games to evaluate per CPU core    

# folders of 4 different models
AGENTS_DIR = [
    'models_no_eval',
    'models_eval_all_geq_0',
    'models_eval_sum_geq_16',
    'models_eval_mcts_max'
]
# names of pickle files, same for each agent
PICKLE_FILES = [
    'acting_net_0',
    'acting_net_1',
    'acting_net_2',
    'acting_net_3',
    'acting_net_4',
    'acting_net_5',
    'acting_net_6',
    'acting_net_7',
    'acting_net_8',
    'acting_net_9',
    'acting_net_10',
    'acting_net_11',
    'acting_net_12',
]

def main():
    for agent in AGENTS_DIR:
        rewards_gens_array = []
        moves_sum_array = []
        score_sum_array = []
        for file in PICKLE_FILES:
            net_acting = AlphaZeroNet((173,8,8), 4096, 8, 32, 32).to('cpu')
            net_acting.load_state_dict(torch.load(agent + '/' + file + '.pkl'))
            evaluator = Eval(net_acting, net_acting, CPU_CORES, EVAL_GAMES)
            moves_sum, score_sum, rewards_sum, rank_counts = evaluator.run(BUDGET, (1, 1, 1, 1))
            moves_sum_array.append(moves_sum)
            rewards_gens_array.append(rewards_sum)
            score_sum_array.append(score_sum)
            print('moves_sum:', moves_sum)
            print('rewards_sum:', rewards_sum)
            print('score_sum:', score_sum)
        # Save the results
        with open(f'{agent}_moves_sum_array_results.pkl', 'wb') as f:
            pickle.dump(moves_sum_array, f)
        with open(f'{agent}_rewards_gens_array_results.pkl', 'wb') as f:
            pickle.dump(rewards_gens_array, f)
        with open(f'{agent}_score_sum_array_results.pkl', 'wb') as f:
            pickle.dump(score_sum_array, f)

if __name__ == '__main__':
    main()