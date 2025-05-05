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
import hashlib
faulthandler.enable()

GAME_FILE_FORMAT = 'bin'  # File format for game files

def set_process(process_id, info=False):
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([process_id])  # Pin to specific CPU core
    except Exception as e:
        print(f"Failed to set CPU affinity for process {process_id}: {e}")

    if info:
        print(f"Process {process_id} started")

def directory_name(dir, hashes, search_budget, players):
    name = f'{dir}_games/'
    for hash in hashes:
        name += f'{hash}_'
    name += f'{search_budget}_'
    for player in players:
        name += f'{player}_'
    name = name[:-1]  # Remove the last underscore
    return name

# counts the number of times each player finished at each rank
# if multiple players finish at the same rank, they all get the same rank
# f.e.: [1, 2, 2, 4] -> [1st, 2nd, 2nd, 4]
def rank_counts(data):
    n = 4
    rank_matrix = np.zeros((n, n), dtype=int)

    for row in data:
        sorted_indices = sorted(range(n), key=lambda i: row[i], reverse=True)
        sorted_values = [row[i] for i in sorted_indices]

        ranks = [0] * n 
        current_rank = 0

        for i in range(n):
            if i > 0 and sorted_values[i] != sorted_values[i - 1]:
                current_rank = i  
            ranks[sorted_indices[i]] = current_rank

        for idx, rank in enumerate(ranks):
            rank_matrix[idx][rank] += 1

    return rank_matrix

# generates hash based only on model parameters
def get_model_hash(model, length=8):
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())

    return hasher.hexdigest()[:length]

def save_model(model, directory = 'models', name = None):
    if name is None:
        name = get_model_hash(model)
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), f'{directory}/{name}.pkl')