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

total_score = (0,0,0,0)

game_storage = chaturajienv.game_storage(1000)

device_training = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device_acting = (torch.device('cpu'))

net = AlphaZeroNet((172,8,8), 4096, 4, 16, 16).to(device_acting)
net.share_memory()
net.eval()

num_params = sum(p.data.nelement() for p in net.parameters())
print(f'Number of parameters: {num_params} \n')

def run_mcts_game(thread_id, model, game_storage):
    """Runs MCTS using the shared model."""
    while True:
        use_model = game_storage.size > 10000 # first 10000 games are using vanilla MCTS
        game = chaturajienv.game()
        for j in range(10000):
            budget = 800 #800 in AlphaZero
            while budget > 0:
                sample = game.get_evaluate_sample(8, 0)
                sample = torch.from_numpy(sample).unsqueeze(0)
                if use_model:
                    with torch.no_grad():
                        p, v = model(sample)
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
                print('Game ', thread_id, '-', g, ' finished after ', j+1, ' moves')
                total_score = tuple(map(sum, zip(total_score, game.get(j+1).get_score_default())))
                print(game.final_reward)
                break
        print('total score:', total_score)
        game_storage.add_game(game)


if __name__ == "__main__":
    num_threads = 8
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=run_mcts_game, args=(i, net, game_storage))
        t.start()
        threads.append(t)