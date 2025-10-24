# File: train_dqn.py
import os
import torch
import numpy as np
import math
from envs.env_utils import create_mario_env
from models.dqn_model import DuelingDQN_CNN
from buffers.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent

# --- 1. THIẾT LẬP SIÊU THAM SỐ ---
TOTAL_TIMESTEPS = 10_000_000
LEVEL_NAME = 'SuperMarioBros-1-1-v3'

# DQN
REPLAY_BUFFER_CAPACITY = 100_000 # Kích thước buffer
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4

# Epsilon-Greedy (Mục 4.6)
EPS_START = 1.0     # 100% ngẫu nhiên
EPS_END = 0.01      # 1% ngẫu nhiên
EPS_DECAY = 1_000_000 # Số bước để giảm từ START xuống END

LEARN_START_STEP = 10_000 # Bắt đầu học sau khi buffer có 10k mẫu
LEARN_EVERY_STEP = 4      # Học mỗi 4 bước
TARGET_UPDATE_FREQ = 5000 # Cập nhật target net mỗi 5k bước

SAVE_CHECKPOINT_FREQ = 100_000 

def get_epsilon(step):
    """Tính epsilon giảm dần (linear decay)."""
    if step > EPS_DECAY:
        return EPS_END
    return EPS_START - (EPS_START - EPS_END) * (step / EPS_DECAY)

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 2. KHỞI TẠO ---
    env = create_mario_env(LEVEL_NAME)
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    agent = DQNAgent(
        input_channels=obs_shape[0],
        num_actions=num_actions,
        device=device,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ
    )
    
    MODEL_PATH = "TrainDQN/dqn_mario_5000000+SuperMarioBros-1-1-v3.pth" 
    try:
        # Tải trọng số vào "policy_net"
        agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # QUAN TRỌNG: Đồng bộ "target_net" ngay lập tức
        agent.update_target_net() 
        
        print(f"--- Đã tải model từ '{MODEL_PATH}'. Tiếp tục huấn luyện. ---")
    except FileNotFoundError:
        print(f"--- Không tìm thấy model. Bắt đầu huấn luyện từ đầu. ---")

    global_step = 5000000
    last_checkpoint_step = 5000000
    state = env.reset()

    # --- 3. VÒNG LẶP HUẤN LUYỆN CHÍNH ---
    while global_step < TOTAL_TIMESTEPS:
        
        # 1. Tính Epsilon
        epsilon = get_epsilon(global_step)
        
        # 2. Chọn hành động (Epsilon-Greedy)
        action = agent.policy_net.get_action(state, epsilon, device)
        
        # 3. Tương tác với môi trường
        next_state, reward, done, info = env.step(action)
        
        # 4. Lưu vào Replay Buffer
        buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        global_step += 1
        
        if done:
            print(f"Bước {global_step}: Hoàn thành episode. Flag: {info.get('flag_get')}")
            state = env.reset()

        # 5. Học (nếu đủ điều kiện)
        if global_step > LEARN_START_STEP and global_step % LEARN_EVERY_STEP == 0:
            agent.learn(buffer, BATCH_SIZE)

        # 6. Lưu Model
        if global_step - last_checkpoint_step >= SAVE_CHECKPOINT_FREQ:
            checkpoint_path = os.path.join("TrainDQN", f"dqn_mario_{global_step}+{LEVEL_NAME}.pth")
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            print(f"Đã lưu model tại: {checkpoint_path}")
            last_checkpoint_step = global_step

    env.close()
    print("Hoàn tất huấn luyện DQN!")

if __name__ == "__main__":
    main()