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
import torch.multiprocessing as multiprocessing
import time
import faulthandler
import psutil
import os
import tools
from network import AlphaZeroNet
faulthandler.enable()

def run_mcts_game(process_id, net, stop_event, search_budget, use_model = False):
    tools.set_process(process_id, False)
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
                    p = p.squeeze(0).cpu().numpy()
                    v = v.squeeze(0).cpu().numpy()     
                else:
                    p = np.random.rand(4096)
                    v = np.random.rand(4)
                #p = np.exp(p)/np.sum(np.exp(p))
                #p = torch.nn.functional.softmax(p, dim=1)

                budget = game.give_evaluated_sample(p, v, budget)

            if(game.step_stochastic(1.0)):
                game_index += 1
                moves += j+1
                break

        game.save_game(f'cache_games/game_{process_id}_{game_index}.bin')
    print('Process:', process_id, 'games:', game_index, f'time per move: {((time.time() - start_time) / (moves + 1)):.5g}')

class MCTS:
    def __init__(self, net, device, storage_size = 200000, num_processes = 8, budget = 800):
        #network related
        self.net = net # The shared network
        self.net.share_memory()
        self.net.eval()
        self.device = device # The device to run the network on

        #processes related
        multiprocessing.set_start_method('spawn', force=True)
        self.num_processes = num_processes
        self.stop_event = multiprocessing.Event()
        self.processes = []
        os.makedirs('cache_games', exist_ok=True)

        #mcts related
        self.budget = budget
        self.game_storage = chaturajienv.game_storage(storage_size)
        self.use_model = False

    def process_cache(self):
        for file in os.listdir('cache_games'):
            self.game_storage.load_game(f'cache_games/{file}')
            os.remove(f'cache_games/{file}')

    def start(self):
        #make sure cache is clear
        self.process_cache()
        self.processes = []
        self.stop_event.clear()
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                    target=run_mcts_game, 
                    args=(i, self.net, self.stop_event, self.budget, self.use_model)
                )
            p.start()
            self.processes.append(p)

    def stop(self):
        self.stop_event.set()
        for p in self.processes:
            p.join()
        self.process_cache()
    
    def size(self):
        return self.game_storage.size()
    
    def cache_size(self):
        return len(os.listdir('cache_games'))

    def get_batch(self, batch_size):
        if self.game_storage.size() == 0:
            return None, None, None
        #just to find out sample, policy, value shapes
        (sample, policy, value) = self.game_storage.get_random_sample(8)
        sample_shape = sample.shape
        policy_shape = policy.shape
        value_shape = value.shape

        samples = np.empty((batch_size, *sample_shape), dtype=np.float32)
        policies = np.empty((batch_size, *policy_shape), dtype=np.float32)
        values = np.empty((batch_size, *value_shape), dtype=np.float32)

        for i in range(batch_size):
            (sample, policy, value) = self.game_storage.get_random_sample(8)
            samples[i] = sample
            policies[i] = policy
            values[i] = value

        return samples, policies, values