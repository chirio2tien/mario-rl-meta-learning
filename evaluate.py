# File: evaluate.py (Phiên bản "All-in-One")
import torch
import time
import numpy as np
from envs.env_utils import create_mario_env

# === 1. SỬA IMPORT: Tải CẢ HAI kiến trúc model ===
from models.model import ActorCriticCNN       # Model cho PPO và Meta
from models.dqn_model import DuelingDQN_CNN  # Model cho DQN


# 1. Chọn loại model bạn muốn xem: "PPO", "DQN", hoặc "META"
# MODEL_TYPE = "DQN"

# # 2. Sửa đường dẫn đến file model
# MODEL_PATH = "TrainDQN/dqn_mario_10000000+SuperMarioBros-1-1-v3.pth"

# Nếu muốn xem PPO:
MODEL_TYPE = "PPO"
MODEL_PATH = "TrainPPO/ppo_mario_8630272+SuperMarioBros-1-1-v3.pth"

# Nếu muốn xem META:
# MODEL_TYPE = "META"
# MODEL_PATH = "checkpoints_meta/meta_policy_50.pth"
# ----------------------------------------------------------------------

LEVEL_NAME = 'SuperMarioBros-1-1-v0' # Cứ để v0 cho tất cả

def main():
    device = torch.device("cpu") # Đánh giá thì không cần GPU
    print(f"Sử dụng thiết bị: {device}")
    print(f"Đang tải model loại: {MODEL_TYPE} từ: {MODEL_PATH}")

    # 1. Khởi tạo môi trường
    env = create_mario_env(LEVEL_NAME)
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # === 2. SỬA MODEL: Tải đúng kiến trúc dựa trên MODEL_TYPE ===
    model = None
    if MODEL_TYPE == "PPO" or MODEL_TYPE == "META":
        model = ActorCriticCNN(input_channels=obs_shape[0], num_actions=num_actions).to(device)
    elif MODEL_TYPE == "DQN":
        model = DuelingDQN_CNN(input_channels=obs_shape[0], num_actions=num_actions).to(device)
    else:
        print(f"LỖI: Không biết MODEL_TYPE là '{MODEL_TYPE}'. Vui lòng chọn 'PPO', 'DQN', hoặc 'META'.")
        env.close()
        return

    # Tải trọng số (code này giữ nguyên)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file model '{MODEL_PATH}'.")
        env.close()
        return

    model.eval() 
    print("Bắt đầu chạy đánh giá... (Nhấn Ctrl+C trong terminal để dừng)")

    try:
        while True:
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                env.render()
                time.sleep(0.016) 
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
                
                action_to_take = 0 # Khởi tạo một số int

                # === 3. SỬA CÁCH LẤY ACTION (Quan trọng) ===
                with torch.no_grad():
                    if MODEL_TYPE == "PPO" or MODEL_TYPE == "META":
                        # PPO trả về (tensor, ..., ...)
                        action_tensor, _, _ = model.get_action(state_tensor, deterministic=False)
                        action_to_take = action_tensor.cpu().numpy()[0] # Lấy số int
                    
                    elif MODEL_TYPE == "DQN":
                        # DQN trả về 1 số int
                        action_to_take = model.get_action(state_tensor, epsilon=0.05, device=device) 
                
                # 5. Tương tác (luôn dùng số int)
                next_state, reward, done, info = env.step(action_to_take)
                
                state = next_state
                total_reward += reward

            print(f"Episode kết thúc. Tổng thưởng: {total_reward}. Mario qua cờ: {info.get('flag_get')}")

    except KeyboardInterrupt:
        print("\nĐã dừng đánh giá.")
    
    env.close()

if __name__ == "__main__":
    main()