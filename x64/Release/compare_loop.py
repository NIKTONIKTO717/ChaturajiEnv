import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import faulthandler
from network import AlphaZeroNet
from comparison import Compare
import pickle
faulthandler.enable()

CPU_CORES = 22                  # Number of CPU cores to use for MCTS
BUDGET = [400, 400, 400, 400]   # Number of MCTS simulations per move (800 in AlphaZero)
EVAL_GAMES = 1                  # Number of games to evaluate per CPU core    

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
    rewards_gens_array = []
    moves_sum_array = []
    score_sum_array = []
    for file in PICKLE_FILES:
        start_time = time.time()
        networks = []
        rewards_sum_sum = [0,0,0,0]
        score_sum_sum = [0,0,0,0]
        for i in range(4):
            if AGENTS_DIR[i] is None:
                networks.append(None)
            else:
                networks.append(AlphaZeroNet((173,8,8), 4096, 8, 32, 32))
                networks[i].load_state_dict(torch.load(AGENTS_DIR[i] + '/' + file + '.pickle'))
        #networks[3] = None # last one is MCTS
        compare = Compare(networks, AGENTS_DIR, file, CPU_CORES, EVAL_GAMES)
        for setup in [(0,1,2,3), (0,1,3,2), (0,2,1,3), (0,2,3,1), (0,3,1,2), (0,3,2,1)]:
            print('Comparing agents:', setup)
            budget_setup = [BUDGET[i] for i in setup]
            moves_sum, score_sum, rewards_sum, rank_counts = compare.run_shifting(budget_setup, setup)
            moves_sum_array.append(moves_sum)
            rewards_sum_reordered = [rewards_sum[setup.index(i)] for i in range(4)]
            score_sum_reordered = [score_sum[setup.index(i)] for i in range(4)]
            score_sum_sum = np.add(score_sum_sum, score_sum_reordered)
            rewards_sum_sum = np.add(rewards_sum_sum, rewards_sum_reordered)
        rewards_gens_array.append(rewards_sum_sum)
        score_sum_array.append(score_sum_sum)
        print('Time taken:', time.time() - start_time)

    with open('rewards_gens_array.pkl', 'wb') as f:
        pickle.dump(rewards_gens_array, f)
    with open('moves_sum_array.pkl', 'wb') as f:
        pickle.dump(moves_sum_array, f)
    with open('score_sum_array.pkl', 'wb') as f:
        pickle.dump(score_sum_array, f)

    rewards_by_agent = list(zip(*rewards_gens_array))
    plt.plot(rewards_by_agent[0], label='no_eval')
    plt.plot(rewards_by_agent[1], label='eval_all_geq_0')
    plt.plot(rewards_by_agent[2], label='eval_sum_geq_16')
    plt.plot(rewards_by_agent[3], label='eval_mcts_max')
    plt.xlabel('Hours of training')
    plt.ylabel('Rewards')
    plt.title('Different agents rewards')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()