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

# if networks passed as a list, they aren't coppied as expected. 
def play_game(process_id, networks, directories, n_games, search_budget = [800,800,800,800], player_setups = [(0,1,2,3)]):
    tools.set_process(process_id)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for directory, players in zip(directories, player_setups):
        for p in range(4):
            if networks[p] == None:
                raise ValueError('Invalid network')
            if search_budget[players[p]] < 0:
                raise ValueError('Invalid search budget')
        for game_index in range(n_games):
            games = [] # every player has own game instance, so they don't use same MCTS tree
            for _ in range(4):
                games.append(chaturajienv.game())
                games[-1].evaluation_game = True
            turn = 0
            for _ in range(10000):
                game = games[turn]
                net = networks[players[turn]]
                budget = search_budget[players[turn]]
                while budget > 0:
                    sample = game.get_evaluate_sample(8, 0)
                    sample = torch.from_numpy(sample).unsqueeze(0)
                    with torch.no_grad():
                        p_net, v_net = net(sample)
                    p_out = p_net.squeeze(0).numpy()
                    v_out = v_net.squeeze(0).numpy()
                    budget = game.give_evaluated_sample(p_out, v_out, budget, 4.0) #c_puct = 4.0

                game = games[turn]

                terminaleted = False
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
        

class Compare:
    #acting -> previous network (device = cpu)
    #training -> new network (device = cuda if available)
    def __init__(self, networks, directories, filename, num_processes, n_games):
        self.networks = networks #[net_1, net_2, net_3, net_4]
        self.directories = directories #[directory_1, directory_2, directory_3, directory_4]
        self.filename = filename #net_1
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
        directory = tools.directory_name('compare', [self.filename] + self.directories, search_budget, players)
        os.makedirs(directory, exist_ok=True)
        for net in self.networks:
            net.share_memory()
            net.eval()
        processes = []
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                    target=play_game, 
                    args=(
                        i, 
                        self.networks,
                        [directory],
                        self.n_games, 
                        [search_budget,search_budget,search_budget,search_budget], 
                        [players]
                    )
                )
            p.start()
            processes.append(p)
        print(f'Comparison {search_budget}-{players} started', flush=True)
        for p in processes:
            p.join()
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
        Runs the Comparison of the network with shifting player positions.

        Returns:
            moves_sum (int): Total number of moves made during the comparison.
            score_sum (tuple): Sum of scores for each player.
            rewards_sum (np.ndarray(float)): Sum of rewards for each player.
            rewards_sum_min (np.ndarray): Minimum rewards for each player across all shifted positions (rewards_sum_min[1] = 2.3 means that for any position sum of rewards from all games played in this position is at least 2.3).
            rank_counts (np.ndarray): Counts of ranks for each player rank_counts[player_index].
        """
        directory = tools.directory_name('compare', [self.filename] + self.directories, search_budget, players)
        directories = [directory + '_0', directory + '_1', directory + '_2', directory + '_3']
        for d in directories:
            os.makedirs(d, exist_ok=True)

        for net in self.networks:
            net.share_memory()
            net.eval()#.to(memory_format=torch.channels_last)
            #net = torch.compile(net, mode='reduce-overhead', fullgraph=True, backend='inductor')

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
                        self.networks,
                        directories,
                        self.n_games, 
                        [search_budget,search_budget,search_budget,search_budget], 
                        setups,
                    )
                )
            p.start()
            processes.append(p)
        print(f'Comparison budget: {search_budget}, setup: {players} started (including shifting positions)', flush=True)
        for p in processes:
            p.join()

        moves_sum = 0
        score_sum = (0,0,0,0)
        rewards = []

        for i in range(4):
            moves, scores, rew = self.process_games(directories[i])
            moves_sum += moves
            score_shift_back = scores[-i:] + scores[:-i]
            rew_shift_back = [r[-i:] + r[:-i] for r in rew]
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
        return moves_sum, score_sum, rewards_sum, rank_counts
        
    