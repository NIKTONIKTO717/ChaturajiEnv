import chaturajienv
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.multiprocessing as multiprocessing
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
@torch.compile
def play_game(process_id, net_acting, net_training, directories, n_games, search_budget = 800, player_setups = [(0,0,0,0)]):
    tools.set_process(process_id)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for directory, players in zip(directories, player_setups):
        for p in range(4):
            if players[p] not in [-1, 0, 1, 2]:
                raise ValueError('Invalid player value')
        for game_index in range(n_games):
            games = [] # every player has own game instance, so they don't use same MCTS tree
            for _ in range(4):
                games.append(chaturajienv.game())
                games[-1].evaluation_game = True
            turn = 0
            for _ in range(10000):
                # for p in range(4): #each player has same budget for estimation
                game = games[turn]
                budget = search_budget #800 in AlphaZero
                while budget > 0:
                    sample = game.get_evaluate_sample(8, 0)
                    sample = torch.from_numpy(sample).unsqueeze(0)
                    # for random and vanilla MCTS is used vanilla MCTS estimation
                    if players[turn] == -1 or players[turn] == 0:
                        p_out = np.ones(4096)
                        v_out = np.zeros(4)
                    else:
                        if players[turn] == 1:
                            with torch.no_grad():
                                p_net, v_net = net_acting(sample)
                        else:
                            with torch.no_grad():
                                p_net, v_net = net_training(sample)
                        p_out = p_net.squeeze(0).numpy()
                        v_out = v_net.squeeze(0).numpy()
                    budget = game.give_evaluated_sample(p_out, v_out, budget, 4.0) #c_puct = 4.0

                game = games[turn]

                terminaleted = False
                if players[turn] == -1:
                    terminaleted = game.step_random()
                else:
                    terminaleted = game.step_deterministic()

                if terminaleted:
                    game.save_game(f'{directory}/game_{process_id}_{game_index}.bin')
                    break
                else:
                    last_action = game.get_action(-1)
                    for i in range(4):
                        if i != turn:
                            games[i].step_given(last_action)
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
        #zip directory, archive has smaller size than directory
        os.system(f'zip -r {directory}.zip {directory} > {os.devnull} 2>&1')
        for file in os.listdir(f'{directory}'):
            os.remove(f'{directory}/{file}')
        os.rmdir(directory)
        return moves_sum, score_sum, rewards

    def run(self, search_budget = 800, players = (0,0,0,0)):
        directory = tools.directory_name('evaluated', [self.net_acting_hash, self.net_training_hash], search_budget, players)
        os.makedirs(directory, exist_ok=True)
        self.net_training.to(self.device_acting)
        self.net_training.share_memory()
        self.net_training.eval()
        processes = []
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                    target=play_game, 
                    args=(
                        i, 
                        self.net_acting, 
                        self.net_training,
                        [directory],
                        self.n_games, 
                        search_budget, 
                        [players]
                    )
                )
            p.start()
            processes.append(p)
        print(f'Evaluation {search_budget}-{players} started', flush=True)
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
            print(f'Player {i} ({players[i]}):\t{rank_counts[i][0]}\t{rank_counts[i][1]}\t{rank_counts[i][2]}\t{rank_counts[i][3]}', flush=True)
        return moves_sum, score_sum, rewards_sum, rank_counts
    
    def run_shifting(self, search_budget = 800, players = (0,0,0,0)):
        """
        Runs the evaluation of the network with shifting player positions.

        Returns:
            moves_sum (int): Total number of moves made during the evaluation.
            score_sum (tuple): Sum of scores for each player.
            rewards_sum (np.ndarray(float)): Sum of rewards for each player.
            rewards_sum_min (np.ndarray): Minimum rewards for each player across all shifted positions (rewards_sum_min[1] = 2.3 means that for any position sum of rewards from all games played in this position is at least 2.3).
            rank_counts (np.ndarray): Counts of ranks for each player rank_counts[player_index].
        """
        
        directory = tools.directory_name('evaluated', [self.net_acting_hash, self.net_training_hash], search_budget, players)
        directories = [directory + '_0', directory + '_1', directory + '_2', directory + '_3']
        for d in directories:
            os.makedirs(d, exist_ok=True)

        self.net_training.to(self.device_acting)
        self.net_training.share_memory()
        self.net_training.eval()

        setups = [
            players,
            (players[1], players[2], players[3], players[0]),
            (players[2], players[3], players[0], players[1]),
            (players[3], players[0], players[1], players[2])
        ]

        processes = []
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                    target=play_game, 
                    args=(
                        i, 
                        self.net_acting, 
                        self.net_training,
                        directories,
                        self.n_games, 
                        search_budget, 
                        setups,
                    )
                )
            p.start()
            processes.append(p)
        print(f'Evaluation budget: {search_budget}, setup: {players} started (including shifting positions)', flush=True)
        for p in processes:
            p.join()
        
        self.net_training.to(self.device_training)
        self.net_training.train()

        moves_sum = 0
        score_sum = (0,0,0,0)
        rewards_sum_min = (9999,9999,9999,9999)
        rewards = []

        for i in range(4):
            moves, scores, rew = self.process_games(directories[i])
            moves_sum += moves
            score_shift_back = scores[-i:] + scores[:-i]
            rew_shift_back = [r[-i:] + r[:-i] for r in rew]
            rew_sum = tuple(map(sum, zip(*rew_shift_back)))
            rewards_sum_min = tuple(map(min, zip(rewards_sum_min, rew_sum)))
            for r in rew_shift_back:
                rewards.append(r)
            score_sum = tuple(map(sum, zip(score_sum, score_shift_back)))
        rewards_sum = tuple(map(sum, zip(*rewards)))
        rewards_sum_formatted = ", ".join([f"{item:.5g}" for item in rewards_sum])
        print(f'Moves: {moves_sum}, Score: {score_sum}, Rewards: [{rewards_sum_formatted}]')
        rank_counts = tools.rank_counts(rewards)
        print('\t\t1st\t2nd\t3rd\t4th')
        for i in range(4):
            print(f'Player ({players[i]}):\t{rank_counts[i][0]}\t{rank_counts[i][1]}\t{rank_counts[i][2]}\t{rank_counts[i][3]}', flush=True)
        return moves_sum, score_sum, rewards_sum, rewards_sum_min, rank_counts
        
    