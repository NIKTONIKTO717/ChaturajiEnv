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
import tools
import pybind11
from network import AlphaZeroNet
faulthandler.enable()

def run_mcts_game(process_id, net, stop_event, search_budget, use_model, hash=''):
    tools.set_process(process_id)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    game_index = 0
    start_time = time.time()
    moves = 0
    while not stop_event.is_set():
        game = chaturajienv.game()
        for j in range(10000):
            budget = search_budget #800 in AlphaZero
            while budget > 0:
                sample = game.get_evaluate_sample(8, 0)
                sample = torch.from_numpy(sample).unsqueeze(0)
                if use_model:
                    with torch.no_grad():
                        p, v = net(sample)
                    p = p.squeeze(0).numpy()
                    v = v.squeeze(0).numpy()     
                else:
                    p = np.random.rand(4096)
                    v = np.random.rand(4)

                budget = game.give_evaluated_sample(p, v, budget, 4.0) #c_puct = 4.0

            if(game.step_stochastic(1.0)):
                game_index += 1
                moves += j+1
                break

        game.save_game(f'cache_games/game_{hash}_{process_id}_{game_index}.bin')
    print('Process:', process_id, 'games:', game_index, f'time per move: {((time.time() - start_time) / (moves + 1)):.5g}', flush=True)

class MCTS:
    def __init__(self, net, storage_size = 200000, num_processes = 7, budget = 800, directories = ['cache_games']):
        #network related
        self.net = net # The shared network

        #processes related
        multiprocessing.set_start_method('spawn', force=True)
        self.num_processes = num_processes
        self.stop_event = multiprocessing.Event()
        self.processes = []
        self.directories = directories
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        #mcts related
        self.budget = budget
        self.game_storage = chaturajienv.game_storage(storage_size)
        self.use_model = False
        
        #performance related
        p = psutil.Process(os.getpid())
        p.cpu_affinity([self.num_processes, self.num_processes + 1]) #next 2 cores after last process core

    def process_samples(self, directory):
        for file in os.listdir(directory):
            if file.endswith(".bin"):
                self.game_storage.load_game(f'{directory}/{file}')

    def process_cache(self):
        for directory in self.directories:
            for file in os.listdir(directory):
                if file.endswith(".bin"):
                    self.game_storage.load_game(f'{directory}/{file}')
                    os.remove(f'{directory}/{file}')
        print('Game storage size:', self.game_storage.size(), flush=True)

    def start(self, process_cache = True):
        #make sure cache is clear
        if process_cache:
            self.process_cache()
        self.processes = []
        self.stop_event.clear()
        #set network to eval mode
        self.net.share_memory()
        self.net.eval()

        print('MCTS Acting network hash:', tools.get_model_hash(self.net))
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                    target=run_mcts_game, 
                    args=(i, self.net, self.stop_event, self.budget, self.use_model, tools.get_model_hash(self.net))
            )
            p.start()
            self.processes.append(p)

    def stop(self, process_cache = True):
        self.stop_event.set()
        for p in self.processes:
            p.join()
        if process_cache:
            self.process_cache()
    
    def size(self):
        return self.game_storage.size()
    
    def cache_size(self):
        return len(os.listdir('cache_games'))

    def get_batch(self, batch_size, sampling_ratio, winrate_sampling = True, win_ratio = 0.5):
        if self.game_storage.size() == 0:
            return None, None, None
        #just to find out sample, policy, value shapes
        (sample, policy, value) = self.game_storage.get_random_sample(8, winrate_sampling, win_ratio)
        sample_shape = sample.shape
        policy_shape = policy.shape
        value_shape = value.shape

        samples = np.empty((batch_size, *sample_shape), dtype=np.float32)
        policies = np.empty((batch_size, *policy_shape), dtype=np.float32)
        values = np.empty((batch_size, *value_shape), dtype=np.float32)

        for i in range(batch_size):
            if sampling_ratio is None:
                (sample, policy, value) = self.game_storage.get_random_sample(8, winrate_sampling, win_ratio)
            else:
                (sample, policy, value) = self.game_storage.get_random_sample_distribution(8, sampling_ratio[0], sampling_ratio[1], sampling_ratio[2], sampling_ratio[3], winrate_sampling, win_ratio)
            samples[i] = sample
            policies[i] = policy
            values[i] = value

        return samples, policies, values