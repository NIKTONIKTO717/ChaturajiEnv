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
from network import AlphaZeroNet
faulthandler.enable()

class MCTS:
    def __init__(self, net, device, game_storage_size = 100000, num_threads = 8, budget = 800):
        self.net = net # The shared network
        self.net.share_memory()
        self.net.eval()
        self.device = device # The device to run the network on
        self.game_storage = chaturajienv.game_storage(game_storage_size)
        self.stop_event = threading.Event()  # Stop flag (for termination)
        self.num_threads = num_threads
        self.budget = budget
        self.threads = []

    def start(self):
        self.threads = []
        self.stop_event.clear()
        for i in range(self.num_threads):
            t = threading.Thread(target=self.run_mcts_game, args=(i,))
            t.start()
            self.threads.append(t)

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join()

    def get_game_storage(self):
        self.stop_event.set()
        for t in self.threads:
            t.join()
        return self.game_storage
    
    def get_game_storage_size(self):
        #safe because mutex implementation in C++
        return self.game_storage.size()
    
    def get_batch(self, batch_size):
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

    def run_mcts_game(self, thread_id):
        """Runs MCTS using the shared model."""
        game_index = 0
        while not self.stop_event.is_set():
            use_model = (self.game_storage.size() > 10000) # first 10000 games are using vanilla MCTS
            game = chaturajienv.game()
            for j in range(10000):
                budget = 800 #800 in AlphaZero
                while budget > 0:
                    sample = game.get_evaluate_sample(8, 0)
                    sample = torch.from_numpy(sample).unsqueeze(0)
                    if use_model:
                        with torch.no_grad():
                            p, v = self.net(sample)
                        p = p.squeeze(0).cpu().numpy()
                        v = v.squeeze(0).cpu().numpy()     
                    else:
                        p = np.random.rand(4096)
                        v = np.random.rand(4)

                    p_mask = game.get_legal_moves_mask()
                    if budget == 800:
                        #add dirichlet noise same as AlphaZero
                        p = 0.75 * p + 0.25 * np.random.dirichlet([0.03] * 4096)
                    p = p * p_mask
                    p = p / np.sum(p)
                    #p = np.exp(p)/np.sum(np.exp(p))
                    #p = torch.nn.functional.softmax(p, dim=1)

                    budget = game.give_evaluated_sample(p, v, budget)

                if(game.step_stochastic(1.0)):
                    print('Game ', thread_id, '-', game_index, ' finished after ', j+1, ' moves')
                    game_index += 1
                    print(game.final_reward)
                    break
            self.game_storage.add_game(game)