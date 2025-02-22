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
def play_game(process_id, net_acting, net_training, directory, n_games, search_budget = 800, players = (0,0,0,0)):
    tools.set_process(process_id)
    for p in range(4):
        if players[p] not in [-1, 0, 1, 2]:
            raise ValueError('Invalid player value')
    for game_index in range(n_games):
        games = [] # every player has own game instance, so they don't use same MCTS tree
        for _ in range(4):
            games.append(chaturajienv.game())
        turn = 0
        for j in range(10000):
            game = games[turn]
            budget = search_budget #800 in AlphaZero
            while budget > 0:
                sample = game.get_evaluate_sample(8, 0)
                sample = torch.from_numpy(sample).unsqueeze(0)
                # for random and vanilla MCTS is used vanilla MCTS estimation
                if players[turn] == -1 or players[turn] == 0:
                    p = np.ones(4096)
                    v = np.zeros(4)
                else:
                    if players[turn] == 1:
                        with torch.no_grad():
                            p, v = net_acting(sample)
                    else: #defaultly also for 
                        with torch.no_grad():
                            p, v = net_training(sample)
                    p = p.squeeze(0).cpu().numpy()
                    v = v.squeeze(0).cpu().numpy()

                budget = game.give_evaluated_sample(p, v, budget)

            if players[turn] == -1:
                if game.step_random():
                    game.save_game(f'{directory}/game_{process_id}_{game_index}.bin')
                    break
            else:
                if game.step_deterministic():
                    game.save_game(f'{directory}/game_{process_id}_{game_index}.bin')
                    break
            last_action = game.get_action(-1)
            for i in range(4):
                if i != turn:
                    games[i].step(last_action)
            turn = game.get(-1).turn
        

class Eval:
    #acting -> previous network (device = cpu)
    #training -> new network (device = cuda if available)
    def __init__(self, net_acting, net_training, num_processes, n_games):
        self.net_acting = net_acting
        self.net_training = net_training
        self.net_acting_hash = tools.get_model_hash(net_acting)
        self.net_training_hash = tools.get_model_hash(net_training)
        self.device_acting = next(self.net_acting.parameters()).device
        self.device_training = next(self.net_training.parameters()).device
        self.num_processes = num_processes
        self.n_games = n_games

    def process_games(self, directory):
        moves_sum = 0
        score_sum = (0,0,0,0)
        rewards = []
        for file in os.listdir(f'{directory}'):
            game = chaturajienv.load_game(f'{directory}/{file}')
            moves_sum += game.size()
            score_sum = tuple(map(sum, zip(score_sum, game.get(-1).get_score_default())))
            rewards.append(game.final_reward)
        for file in os.listdir(f'{directory}'):
            os.remove(f'{directory}/{file}')
        return moves_sum, score_sum, rewards

    def run(self, search_budget = 800, players = (0,0,0,0)):
        directory = tools.directory_name(self.net_acting_hash, self.net_training_hash, search_budget, players)
        os.makedirs(directory, exist_ok=True)
        self.net_training.to(self.device_acting)
        self.net_training.eval()
        processes = []
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                    target=play_game, 
                    args=(
                        i, 
                        self.net_acting, 
                        self.net_training, 
                        directory,
                        self.n_games, 
                        search_budget, 
                        players
                    )
                )
            p.start()
            processes.append(p)
        print(f'Evaluation {search_budget}-{players} started')
        for p in processes:
            p.join()
        self.net_training.to(self.device_training)
        self.net_training.train()
        moves_sum, score_sum, rewards = self.process_games(directory)
        rewards_sum = tuple(map(sum, zip(*rewards)))
        rewards_sum_formatted = ", ".join([f"{item:.5g}" for item in rewards_sum])
        print(f'Moves: {moves_sum}, Score: {score_sum}, Rewards: [{rewards_sum_formatted}]')
        rank_counts = tools.rank_counts(rewards)
        print('\t\t1st\t2nd\t3rd\t4th')
        for i in range(4):
            print(f'Player {i} ({players[i]}):\t{rank_counts[i][0]}\t{rank_counts[i][1]}\t{rank_counts[i][2]}\t{rank_counts[i][3]}')
        return moves_sum, score_sum, rewards_sum, rank_counts
        
    