import torch
import time
import faulthandler
from network import AlphaZeroNet
from mcts import MCTS 
import tools
import os
faulthandler.enable()

# === Hyperparameters ===
CPU_CORES = 30              # Number of CPU cores to use for MCTS
BUDGET = 400                # Number of MCTS simulations per move (800 in AlphaZero)
SELF_PLAY_DIR = ['cache_games'] # Directory for self-play games
MODEL_DIR = 'models' # Directory for model storage
CHECK_INTERVAL = 60  # seconds

def get_latest_model_path(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not models:
        return None
    models.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    return os.path.join(model_dir, models[0])

def main():
    device_acting = (torch.device('cpu'))
    # 20 * 8 + 4 + 4 + 4 + 1 = 173
    net_acting = AlphaZeroNet((173,8,8), 4096, 8, 32, 32).to(device_acting)
    net_acting.eval()

    mcts = MCTS(net_acting, 1, CPU_CORES, BUDGET, SELF_PLAY_DIR)
    mcts.use_model = True

    last_loaded_model = None

    while True:
        latest_model = get_latest_model_path(MODEL_DIR)
        if latest_model and latest_model != last_loaded_model:
            mcts.stop(False)
            latest_model = get_latest_model_path(MODEL_DIR) # meanwhile another model could be saved
            print(f"New model detected: {latest_model}")
            try:
                state_dict = torch.load(latest_model, map_location="cpu")
                net_acting.load_state_dict(state_dict)
                last_loaded_model = latest_model
                print("Model updated to:", tools.get_model_hash(net_acting))
            except Exception as e:
                print(f"Error loading model: {e}")
            mcts.start(False)

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()