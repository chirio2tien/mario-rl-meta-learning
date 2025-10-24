# File: train.py
import os
import torch
import numpy as np
from envs.env_utils import create_mario_env
from models.model import ActorCriticCNN
from buffers.rollout_buffer import RolloutBuffer
from agents.ppo_agent import PPOAgent

# --- 1. THIẾT LẬP SIÊU THAM SỐ ---
TOTAL_TIMESTEPS = 10_000_000
LEVEL_NAME = 'SuperMarioBros-1-1-v3'

# PPO
ROLLOUT_STEPS = 512 # Số bước thu thập dữ liệu trước mỗi lần update
PPO_EPOCHS = 10
MINI_BATCH_SIZE = 256 # Kích thước 1 mini-batch (ROLLOUT_STEPS / MINI_BATCH_SIZE = 2 batches)
LEARNING_RATE = 2.5e-4
CLIP_EPSILON = 0.1
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.1
VALUE_LOSS_COEF = 0.5

SAVE_CHECKPOINT_FREQ = 100_000 # Lưu model mỗi 100k bước

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 2. KHỞI TẠO ---
    env = create_mario_env(LEVEL_NAME)
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape # (4, 84, 84)

    model = ActorCriticCNN(input_channels=obs_shape[0], num_actions=num_actions).to(device)

    MODEL_PATH = "TrainPPO/ppo_mario_8830876+SuperMarioBros-1-1-v3.pth" 
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"--- Đã tải model từ '{MODEL_PATH}'. Tiếp tục huấn luyện. ---")
    except FileNotFoundError:
        print(f"--- Không tìm thấy model. Bắt đầu huấn luyện từ đầu. ---")
    
    buffer = RolloutBuffer(ROLLOUT_STEPS, gamma=GAMMA, gae_lambda=GAE_LAMBDA)
    agent = PPOAgent(
        model=model,
        learning_rate=LEARNING_RATE,
        clip_epsilon=CLIP_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        value_loss_coef=VALUE_LOSS_COEF,
        entropy_coef=ENTROPY_COEF
    )

    global_step = 0
    num_updates = 0
    last_checkpoint_step = 0
    
    # Khởi tạo state
    state = env.reset() # shape (4, 84, 84)

    # --- 3. VÒNG LẶP HUẤN LUYỆN CHÍNH ---
    while global_step < TOTAL_TIMESTEPS:
        
        # --- 3.1 THU THẬP DỮ LIỆU (ROLLOUT) ---
        model.eval() # Chuyển sang chế độ eval khi rollout
        
        for step in range(ROLLOUT_STEPS):
            # Chuyển state sang tensor
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = model.get_action(state_tensor)
            
            # Tương tác với môi trường
            next_state, reward, done, info = env.step(action.cpu().numpy()[0])
            
            # Lưu vào buffer
            buffer.add(
                state_tensor.cpu(),
                action.cpu(),
                reward,
                done,
                log_prob.cpu(),
                value.cpu()
            )
            
            state = next_state
            global_step += 1
            
            if done:
                print(f"Bước {global_step}: Hoàn thành episode. Flag: {info.get('flag_get')}")
                state = env.reset()

        # --- 3.2 TÍNH ADVANTAGE & CẬP NHẬT ---
        model.train() # Chuyển sang chế độ train
        
        # Tính value của state cuối cùng
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value = model(state_tensor)
            last_value = last_value.cpu().item()
            last_done = done
            
        # 1. Tính GAE và Returns
        buffer.compute_advantages(last_value, last_done)
        
        # 2. Cập nhật PPO
        agent.learn(buffer)
        
        # 3. Xóa buffer
        buffer.clear()
        
        num_updates += 1
        print(f"Hoàn tất PPO Update #{num_updates} tại bước {global_step}")

        # --- 3.3 LƯU MODEL ---
        if global_step - last_checkpoint_step >= SAVE_CHECKPOINT_FREQ:
            checkpoint_path = os.path.join("TrainPPO", f"ppo_mario_{global_step}+{LEVEL_NAME}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Đã lưu model tại: {checkpoint_path}")
            last_checkpoint_step = global_step

    env.close()
    print("Hoàn tất huấn luyện!")

if __name__ == "__main__":
    main()