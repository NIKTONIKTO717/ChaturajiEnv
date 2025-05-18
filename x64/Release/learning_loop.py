import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import faulthandler
from network import AlphaZeroNet
from mcts import MCTS 
from evaluation import Eval
import tools
import pickle
from torch.optim.lr_scheduler import StepLR
faulthandler.enable()

# === Hyperparameters ===
LEARNING_RATE = 1e-2        # Recommended: 1e-4 to 3e-4
BATCH_SIZE = 2048           # Recommended: 512-2048 (depending on memory)
L2_REG = 1e-4               # Weight decay (L2 regularization) to prevent overfitting
ITERATIONS = 1000           # Number of training iterations per evaluation
PASSES = 1                  # Adjust as needed
CPU_CORES = 22              # Number of CPU cores to use for MCTS
STORAGE_SIZE = 5000         # Number of games to store in the game storage
BUDGET = 400                # Number of MCTS simulations per move (800 in AlphaZero)
EVAL_GAMES = 2              # Number of games to evaluate per CPU core    
SELF_PLAY_DIR = ['cache_games', '//n68ad57.mevnode.com/selfplay1/cache_games'] # Directory for self-play games
MODEL_DIR = '//n68ad57.mevnode.com/selfplay1/models' # Directory for model storage
SELF_PLAY = False           # True if you want to generate self-play games
PRELOAD_MODEL_FILE = 'no_eval_preload.pkl' # Preload model file for acting network
ADJUST_SAMPLING_RATIO = True # Adjust sampling ratio based on rewards
WINRATE_SAMPLING = True      # Adjust sampling ratio based on winrate
EXPERIMENT_ID = 'both_samplings_' # Experiment ID for logging

def main():
    start_time = time.time()
    device_training = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    device_acting = (torch.device('cpu'))
    print(f"Training device: {device_training}, Acting device: {device_acting}")

    # 20 * 8 + 4 + 4 + 4 + 1 = 173
    net_acting = AlphaZeroNet((173,8,8), 4096, 8, 32, 32).to(device_acting)
    net_training = AlphaZeroNet((173,8,8), 4096, 8, 32, 32).to(device_training)
    if PRELOAD_MODEL_FILE:
        print(f"Loading model {PRELOAD_MODEL_FILE}")
        state_dict = torch.load(PRELOAD_MODEL_FILE, map_location="cpu")
        net_acting.load_state_dict(state_dict)
        net_training.load_state_dict(state_dict)
    net_acting.eval()
    net_training.train()
    optimizer = optim.Adam(net_training.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.2)

    mcts = MCTS(net_acting, STORAGE_SIZE, CPU_CORES, BUDGET, SELF_PLAY_DIR)
    mcts.use_model = True
    mcts.process_samples('mcts_games')

    sampling_ratio = np.array([0.25, 0.25, 0.25, 0.25])
    moves_array = []
    score_array = []
    rewards_array = []
    rank_counts_array = []
    policy_loss_array = []
    value_loss_array = []
    # training loop
    for l in range(1000):
        print(f"Training network gen {l}", flush=True)
        mcts.process_cache()
        for i in range(ITERATIONS): #from alpha zero - checkpoint every 1000 steps
            if ADJUST_SAMPLING_RATIO:
                samples, policies, values = mcts.get_batch(BATCH_SIZE, sampling_ratio, WINRATE_SAMPLING, 0.5) # sampling_ratio
            else:
                samples, policies, values = mcts.get_batch(BATCH_SIZE, None, WINRATE_SAMPLING, 0.5)
            samples = torch.tensor(samples, dtype=torch.float32, device=device_training)
            policies = torch.tensor(policies, dtype=torch.float32, device=device_training)
            values = torch.tensor(values, dtype=torch.float32, device=device_training)
            for e in range(PASSES):
                # Forward pass
                policy_pred, value_pred = net_training(samples)

                policy_loss = F.cross_entropy(policy_pred, policies)
                value_loss = F.mse_loss(value_pred.squeeze(), values)
                total_loss = policy_loss + value_loss

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if(e + 1 == PASSES and i % 100 == 0):
                    policy_loss_array.append(policy_loss.item())
                    value_loss_array.append(value_loss.item())
                    print(f"Iteration {i}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {total_loss.item():.4f}", flush=True)

        scheduler.step()
        mcts.stop()

        #evaluate the network against random player
        evaluator = Eval(net_acting, net_training, CPU_CORES, EVAL_GAMES)

        #evaluator.run_shifting(BUDGET, (2, -1, -1, -1)) # vanilla MCTS vs random
        #### MCTS ####
        #evaluator.run_shifting(BUDGET, (2, 0, 0, 0)) # acting_net vs vanilla MCTS
        #### SEPARATELY EVAL ####
        #moves_sum_0, score_sum_0, rewards_sum_0, rank_counts_0 = evaluator.run(BUDGET, (2, 1, 1, 1)) # acting_net vs training_net
        #moves_sum_1, score_sum_1, rewards_sum_1, rank_counts_1 = evaluator.run(BUDGET, (1, 2, 1, 1)) # acting_net vs training_net
        #moves_sum_2, score_sum_2, rewards_sum_2, rank_counts_2 = evaluator.run(BUDGET, (1, 1, 2, 1)) # acting_net vs training_net
        #moves_sum_3, score_sum_3, rewards_sum_3, rank_counts_3 = evaluator.run(BUDGET, (1, 1, 1, 2)) # acting_net vs training_net
        #rewards = np.array([rewards_sum_0[0], rewards_sum_1[1], rewards_sum_2[2], rewards_sum_3[3]])
        #### SHIFTING EVAL ####
        #moves_sum, score_sum, rewards_sum, rewards_sum_min, rank_counts = evaluator.run_shifting(BUDGET, (2, 1, 1, 1)) # acting_net vs training_net
        #moves_array.append(moves_sum)
        #score_array.append(score_sum)
        #rewards_array.append(rewards_sum)
        #rank_counts_array.append(rank_counts)

        if ADJUST_SAMPLING_RATIO:
            #### EVAL AGAINST PREVIOUS ####
            moves_sum_0, score_sum_0, rewards_sum_0, rank_counts_0 = evaluator.run(BUDGET, (2, 1, 1, 1)) # acting_net vs training_net
            moves_sum_1, score_sum_1, rewards_sum_1, rank_counts_1 = evaluator.run(BUDGET, (1, 2, 1, 1)) # acting_net vs training_net
            moves_sum_2, score_sum_2, rewards_sum_2, rank_counts_2 = evaluator.run(BUDGET, (1, 1, 2, 1)) # acting_net vs training_net
            moves_sum_3, score_sum_3, rewards_sum_3, rank_counts_3 = evaluator.run(BUDGET, (1, 1, 1, 2)) # acting_net vs training_net
            rewards = np.array([rewards_sum_0[0], rewards_sum_1[1], rewards_sum_2[2], rewards_sum_3[3]])
            #### EVAL AGAINST ITSELF ####
            #moves_sum, score_sum, rewards_sum, rank_counts = evaluator.run(BUDGET, (2, 2, 2, 2)) # acting_net vs itself
            #rewards = np.array([rewards_sum[0], rewards_sum[1], rewards_sum[2], rewards_sum[3]])
            tau = 1.0
            rewards_exp = np.exp(-rewards / (tau * CPU_CORES * EVAL_GAMES)) # normalize by number of games
            sampling_ratio = rewards_exp / np.sum(rewards_exp)
            print(f"Sampling ratio: {sampling_ratio}")       
        
        #copy if better to acting network
        #if rewards_sum[0] >= 6: # maximum is +60
        #if sum(rewards) > 0:
        #if min(rewards_sum_0[0], rewards_sum_1[1], rewards_sum_2[2], rewards_sum_3[3]) > 0:
        #if rewards_sum_min[0] > 0:
        if True:
            net_acting.load_state_dict(net_training.state_dict())
            tools.save_model(net_acting, MODEL_DIR)
            with open(f'{EXPERIMENT_ID}log.txt', "a") as file:
                file.write(f"pickle: Training network gen {l} - {tools.get_model_hash(net_training)} pickled\n")
        else:
            net_training.load_state_dict(net_acting.state_dict())
        if SELF_PLAY:
            mcts.start()

        #pickle python arrays
        pickle.dump(moves_array, open(f'{EXPERIMENT_ID}moves_array.pkl', "wb"))
        pickle.dump(score_array, open(f'{EXPERIMENT_ID}score_array.pkl', "wb"))
        pickle.dump(rewards_array, open(f'{EXPERIMENT_ID}rewards_array.pkl', "wb"))
        pickle.dump(rank_counts_array, open(f'{EXPERIMENT_ID}rank_counts_array.pkl', "wb"))
        pickle.dump(policy_loss_array, open(f'{EXPERIMENT_ID}policy_loss_array.pkl', "wb"))
        pickle.dump(value_loss_array, open(f'{EXPERIMENT_ID}value_loss_array.pkl', "wb"))
        tools.save_model(net_acting, MODEL_DIR, f'last_in_hour/{EXPERIMENT_ID}acting_net_{int((time.time() - start_time) // 3600)}')
        time.sleep(600) # generate more games

    mcts.stop()


if __name__ == '__main__':
    main()