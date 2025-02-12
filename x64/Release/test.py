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
import os
import psutil
faulthandler.enable()

total_score = (0,0,0,0)

game_storage = chaturajienv.game_storage(1000)

p = psutil.Process(os.getpid())
p.cpu_affinity([1])

for g in range(20):
    game = chaturajienv.game()
    start_time = time.time()
    for j in range(10000):
        budget = 800 #800 in AlphaZero
        while budget > 0:
            sample = game.get_evaluate_sample(8, 0)

            #get flat np array of size 4 with random noise
            v = np.random.rand(4)
            p = np.random.rand(4096)
            #p = p.squeeze(0).cpu().numpy()
            #v = v.squeeze(0).cpu().numpy()
            p_mask = game.get_legal_moves_mask()
            if budget == 800:
                #add dirichlet noise
                p = 0.75 * p + 0.25 * np.random.dirichlet([0.03] * 4096)
            p = p * p_mask
            p = p / np.sum(p)
            #p = np.exp(p)/np.sum(np.exp(p))
            #p = torch.nn.functional.softmax(p, dim=1)
            budget = game.give_evaluated_sample(p, v, budget)

        if(game.get(j).turn == 0):
            
            if(game.step_stochastic(1.0)):
                print('Game ', g, ' finished after ', j+1, ' moves')
                total_score = tuple(map(sum, zip(total_score, game.get(j+1).get_score_default())))
                print(game.final_reward)
                print("time per move:", (time.time() - start_time) / (j+1))
                break
        else:
            
            if(game.step_random()):
                print('Game ', g, ' finished after ', j+1, ' moves')
                #game.get(j+1).printBoard()
                #print(game.get(j+1).getScoreDefault())
                total_score = tuple(map(sum, zip(total_score, game.get(j+1).get_score_default())))
                print(game.final_reward)
                print("time per move:", (time.time() - start_time) / (j+1))
                break
    print('total score:', total_score)

    
    game_storage.add_game(game)

(sample, policy, value) = game_storage.get_random_sample(8)
batch_size = 1000
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

print(samples[0].shape)
print(type(samples[0]))