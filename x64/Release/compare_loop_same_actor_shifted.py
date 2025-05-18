import torch
import faulthandler
from network import AlphaZeroNet
from evaluation import Eval
import pickle
faulthandler.enable()

CPU_CORES = 22              # Number of CPU cores to use for MCTS
BUDGET = 1600                # Number of MCTS simulations per move (800 in AlphaZero)
EVAL_GAMES = 4              # Number of games to evaluate per CPU core    
SHIFT = 3

# folders of 4 different models
AGENTS_DIR = [
    'models_without_modification_1',
    'models_adjust_sampling_1',
    'models_winrate_sampling_1',
    'models_both_samplings_1'
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
    'acting_net_13',
    'acting_net_14',
    'acting_net_15',
    'acting_net_16',
    'acting_net_17',
    'acting_net_18',
    'acting_net_19',
    'acting_net_20',
    'acting_net_21',
    'acting_net_22',
    'acting_net_23',
]

def main():
    print('Agent 4 is going to be shifted by '+ str(SHIFT) + ' generations')
    for agent in AGENTS_DIR:
        rewards_gens_array = []
        moves_sum_array = []
        score_sum_array = []
        for file, i in zip(PICKLE_FILES, range(len(PICKLE_FILES) - SHIFT)):
            net_acting = AlphaZeroNet((173,8,8), 4096, 8, 32, 32).to('cpu')
            net_acting.load_state_dict(torch.load(agent + '/' + file + '.pkl'))
            net_acting_shifted = AlphaZeroNet((173,8,8), 4096, 8, 32, 32).to('cpu')
            net_acting_shifted.load_state_dict(torch.load(agent + '/' + PICKLE_FILES[i + SHIFT] + '.pkl'))
            evaluator = Eval(net_acting, net_acting_shifted, CPU_CORES, EVAL_GAMES)
            moves_sum, score_sum, rewards_sum, rank_counts = evaluator.run(BUDGET, (1, 1, 1, 2))
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