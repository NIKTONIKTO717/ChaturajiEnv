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
import psutil
import os
from network import AlphaZeroNet
faulthandler.enable()

game_storage_shared = chaturajienv.game_storage_shared()

def run_mcts_game(process_id, net, stop_event, search_budget, game_storage):
    """Runs MCTS using the shared model."""
    
    game_index = 0
    while not stop_event.is_set():
        use_model = False # (game_storage.size() > 10000) # first 10000 games are using vanilla MCTS
        game = chaturajienv.game()
        start_time = time.time()
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

                p_mask = game.get_legal_moves_mask()
                if budget == search_budget:
                    #add dirichlet noise same as AlphaZero
                    p = 0.75 * p + 0.25 * np.random.dirichlet([0.03] * 4096)
                p = p * p_mask ## TODO: do this in C++
                p = p / np.sum(p)
                #p = np.exp(p)/np.sum(np.exp(p))
                #p = torch.nn.functional.softmax(p, dim=1)

                budget = game.give_evaluated_sample(p, v, budget)

            if(game.step_stochastic(1.0)):
                print('Game', process_id, '-', game_index, 'finished after', j+1, 'moves')
                game_index += 1
                print(game.final_reward)
                print("time per move:", (time.time() - start_time) / (j+1))
                break
        game_storage.add_game(game)
        print('Game storage size:', game_storage.size())

class MCTS:
    def __init__(self, net, device, num_processes = 8, budget = 800):
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

        #mcts related
        self.budget = budget

    def start(self):
        self.processes = []
        self.stop_event.clear()
        for i in range(self.num_processes):
            p = multiprocessing.Process(
                target=run_mcts_game, 
                args=(
                    i, 
                    self.net, 
                    # game_storage is thread safe. Is inside MCTS, contains mutex lock in C++
                    self.stop_event, 
                    self.budget,
                    game_storage_shared
                    )
                )
            p.start()
            self.processes.append(p)

    def stop(self):
        self.stop_event.set()
        for p in self.processes:
            p.join()
    
def get_game_storage_size(game_storage):
    #safe because mutex implementation in C++
    return game_storage.size()

def get_batch(game_storage, batch_size):
    if len(game_storage) == 0:
        return None, None, None
    #just to find out sample, policy, value shapes
    (sample, policy, value) = game_storage.get_random_sample(8)
    sample_shape = sample.shape
    policy_shape = policy.shape
    value_shape = value.shape

    samples = np.empty((batch_size, *sample_shape), dtype=np.float32)
    policies = np.empty((batch_size, *policy_shape), dtype=np.float32)
    values = np.empty((batch_size, *value_shape), dtype=np.float32)

    for i in range(batch_size):
        (sample, policy, value) = game_storage.get_random_sample(8)
        samples[i] = sample
        policies[i] = policy
        values[i] = value

    return samples, policies, values