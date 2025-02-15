import chaturajienv
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import torch.multiprocessing as multiprocessing
import multiprocessing
import time
import faulthandler
import psutil
import os
from network import AlphaZeroNet
import tools
faulthandler.enable()

#player = -1 -> random_move
#player = 0 -> vanilla MCTS deterministic policy (competitive play)
#player = 1 -> acting_net MCTS deterministic policy (competitive play)
#player = 2 -> training_net MCTS deterministic policy (competitive play)
def play_game(process_id, net_acting, net_training, n_games, search_budget = 800, players = (0,0,0,0)):
    tools.set_process(process_id)
    game_index = 0
    start_time = time.time()
    #moves_sum = 0
    #score_sum = (0,0,0,0)
    #rewards = []
    directory = tools.directory_name(hash(net_acting), hash(net_training), search_budget, players)
    for p in range(4):
        if players[p] not in [-1, 0, 1, 2]:
            raise ValueError('Invalid player value')
    for _ in range(n_games):
        game = chaturajienv.game()
        for j in range(10000):
            budget = search_budget #800 in AlphaZero
            while budget > 0:
                sample = game.get_evaluate_sample(8, 0)
                sample = torch.from_numpy(sample).unsqueeze(0)
                # for random and vanilla MCTS is used vanilla MCTS estimation
                if players[game.get(j).turn] == -1 or players[game.get(j).turn] == 0:
                    p = np.zeros(4096)
                    v = np.zeros(4)
                else:
                    if players[game.get(j).turn] == 1:
                        with torch.no_grad():
                            p, v = net_acting(sample)
                    else: #defaultly also for 
                        with torch.no_grad():
                            p, v = net_training(sample)
                    p = p.squeeze(0).cpu().numpy()
                    v = v.squeeze(0).cpu().numpy()

                budget = game.give_evaluated_sample(p, v, budget)

            if players[game.get(j).turn] == -2:
                game.step_random()
                break
            else:
                game.step_deterministic()
                break

        #moves_sum += game.size()
        #score_sum = tuple(map(sum, zip(score_sum, game.get(j+1).get_score_default())))
        #rewards.append(game.final_reward)
        game.save_game(f'evaluated_games/{directory}/game_{game_index}.bin')

class Eval:
    #acting -> previous network (device = cpu)
    #training -> new network (device = cuda if available)
    def __init__(self, net_acting, net_training, num_processes, n_games):
        self.net_acting = net_acting
        self.net_training = net_training
        self.num_processes = num_processes
        self.n_games = n_games

    def process_games(self, directory):
        moves_sum = 0
        score_sum = (0,0,0,0)
        rewards = []
        for file in os.listdir(f'evaluated_games/{directory}'):
            game = chaturajienv.load_game(f'evaluated_games/{directory}/{file}')
            moves_sum += game.size()
            score_sum = tuple(map(sum, zip(score_sum, game.get(game.size()-1).get_score_default())))
            rewards.append(game.final_reward)
        return moves_sum, score_sum, rewards

    def run(self, search_budget = 800, players = (0,0,0,0)):
        directory = tools.directory_name(hash(self.net), search_budget, players)
        os.makedirs(directory, exist_ok=True)
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                    target=play_game, 
                    args=(
                        i, 
                        self.net_acting, 
                        self.net_training, 
                        self.n_games, 
                        self.search_budget, 
                        self.players
                    )
                )
            p.start()
            self.processes.append(p)
        print(f'Evaluation {search_budget}-{players} started')
        for p in self.processes:
            p.join()
        moves_sum, score_sum, rewards = self.process_games(directory)
        rewards_sum = tuple(map(sum, zip(*rewards)))
        print(f'Moves: {moves_sum}, Score: {score_sum}, Rewards: {rewards_sum:.5g}')
        rank_counts = tools.rank_counts(rewards)
        print('\t\t\t1st\t2nd\t3rd\t4th')
        for i in range(4):
            print(f'Player {i} ({players[i]}): {rank_counts[i][0]}\t{rank_counts[i][1]}\t{rank_counts[i][2]}\t{rank_counts[i][3]}')
        return moves_sum, score_sum, rewards_sum, rank_counts
        
    