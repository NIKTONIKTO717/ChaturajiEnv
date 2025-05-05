import chaturajienv
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import time
import faulthandler
import os
import tools
import string
faulthandler.enable()

def random_filename(n = 10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def run_mcts_game(process_id, n_games, search_budget = 800):
    tools.set_process(process_id)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start_time = time.time()
    moves = 0
    total_score = (0,0,0,0)
    total_final_reward = [0.0, 0.0, 0.0, 0.0]
    for game_index in range(n_games):
        game = chaturajienv.game()
        for j in range(10000):
            budget = search_budget #800 in AlphaZero
            while budget > 0:
                 # not necessary to use (8, 0)
                sample = game.get_evaluate_sample(1, 0)
                v = np.zeros(4)
                p = np.ones(4096)
                budget = game.give_evaluated_sample(p, v, budget, 4.0)

            if(game.step_stochastic(1.0)):
                total_score = tuple(map(sum, zip(total_score, game.get(-1).get_score_default())))
                total_final_reward = [sum(x) for x in zip(total_final_reward, game.final_reward)]
                print('Game ', game_index, ' finished after ', j+1, ' moves')
                print(game.get(-1).get_score_default())
                moves += j+1
                break

        if(game_index % 10 == 0):
            print('total score:', total_score)
            print('total final reward:', total_final_reward)

        game.save_game(f'mcts_test_games/{random_filename()}.bin')
    print('Process:', process_id, 'games:', game_index, f'time per move: {((time.time() - start_time) / moves):.5g}')

def main():
    os.makedirs('mcts_games', exist_ok=True)
    processes = []
    for i in [1,2,4,5,7,8,9,10]: # optimization for specific CPU best performing cores (13,15,19,20,11,4,6)
        #2 virtual CPU are used for one physical CPU core, we use only physical CPU per process
        p = multiprocessing.Process(target=run_mcts_game, args=(2*i, 10, 400))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()