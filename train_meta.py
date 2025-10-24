# File: train_meta.py
import os
import torch
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import copy # Dùng để copy model

# Import các file PPO của bạn
from envs.env_utils import create_mario_env
from models.model import ActorCriticCNN
from buffers.rollout_buffer import RolloutBuffer
from agents.ppo_agent import PPOAgent

# --- 1. THIẾT LẬP SIÊU THAM SỐ META ---
TOTAL_META_ITERATIONS = 5000
META_LR = 1e-4        # Learning rate VÒNG NGOÀI (Reptile)
INNER_LR = 2.5e-4     # Learning rate VÒNG TRONG (PPO)
META_BATCH_SIZE = 5   # Số lượng task (level) mỗi lần update
INNER_UPDATES = 10    # Số lần update PPO cho mỗi task
ROLLOUT_STEPS = 512   # (Giống PPO)
PPO_EPOCHS = 4        # (Giống PPO)
MINI_BATCH_SIZE = 256 # (Giống PPO)

# Danh sách các task (level) để học (Mục 4.7)
TASK_LIST = [
    'SuperMarioBros-1-1-v3',
    'SuperMarioBros-1-2-v3',
    'SuperMarioBros-1-3-v3',
    'SuperMarioBros-2-1-v3',
    'SuperMarioBros-2-2-v3',
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 2. KHỞI TẠO META-POLICY ---
    # Đây là model "cha" (theta)
    meta_policy_net = ActorCriticCNN(input_channels=4, num_actions=7).to(device)
    
    # Optimizer cho Reptile
    meta_optimizer = optim.Adam(meta_policy_net.parameters(), lr=META_LR)
    
    print("Bắt đầu Meta-Training (Reptile + PPO)...")

    # --- 3. VÒNG LẶP NGOÀI (META-LEARNING) ---
    for meta_iter in range(TOTAL_META_ITERATIONS):
        
        # 1. Lấy trọng số "cha" (theta)
        meta_weights = copy.deepcopy(meta_policy_net.state_dict())
        adapted_weights_list = [] # Lưu trọng số "con" (phi)

        # 2. Sample các task
        tasks = np.random.choice(TASK_LIST, size=META_BATCH_SIZE, replace=False)
        print(f"\n--- Meta-Iteration {meta_iter} --- Tasks: {tasks} ---")

        # --- 4. VÒNG LẶP TRONG (TASK-SPECIFIC LEARNING) ---
        for task_level in tasks:
            
            # Tạo agent "con" (phi)
            task_net = ActorCriticCNN(input_channels=4, num_actions=7).to(device)
            task_net.load_state_dict(meta_weights) # Copy trọng số "cha"
            
            task_agent = PPOAgent(
                model=task_net,
                learning_rate=INNER_LR,
                ppo_epochs=PPO_EPOCHS,
                mini_batch_size=MINI_BATCH_SIZE
            )
            
            env = create_mario_env(task_level)
            buffer = RolloutBuffer(ROLLOUT_STEPS)
            
            state = env.reset()
            
            # Huấn luyện PPO N bước (Inner updates)
            for _ in range(INNER_UPDATES):
                # (Đây là code y hệt train.py)
                for step in range(ROLLOUT_STEPS):
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob, value = task_net.get_action(state_tensor)
                    
                    next_state, reward, done, info = env.step(action.cpu().numpy()[0])
                    buffer.add(state_tensor.cpu(), action.cpu(), reward, done, log_prob.cpu(), value.cpu())
                    state = next_state
                    if done:
                        state = env.reset()

                # Tính GAE và Update
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
                    _, last_value = task_net(state_tensor)
                    last_value = last_value.cpu().item()
                
                buffer.compute_advantages(last_value, done)
                task_agent.learn(buffer)
                buffer.clear()
            
            # 5. Lưu lại trọng số "con" (phi) đã học
            adapted_weights_list.append(copy.deepcopy(task_net.state_dict()))
            env.close()

        # --- 5. CẬP NHẬT META (REPTILE UPDATE) ---
        
        # Tính trung bình các trọng số "con" (avg(phi))
        avg_adapted_weights = OrderedDict()
        for key in meta_weights.keys():
            avg_adapted_weights[key] = torch.stack(
                [weights[key] for weights in adapted_weights_list]
            ).mean(dim=0)

        # Tính hướng cập nhật (phi_avg - theta)
        update_direction = OrderedDict()
        for key in meta_weights.keys():
            update_direction[key] = avg_adapted_weights[key] - meta_weights[key]
            
        # Cập nhật Reptile (gán pseudo-gradient)
        meta_optimizer.zero_grad()
        for name, param in meta_policy_net.named_parameters():
            # grad = -(phi_avg - theta) = theta - phi_avg
            # param_new = param - lr * grad
            # param_new = param - lr * (theta - phi_avg)
            # param_new = (1 - lr)*theta + lr*phi_avg  (Đây chính là Reptile)
            
            # (Cách gán grad đơn giản hơn)
            # theta_new = theta + meta_lr * (avg_phi - theta)
            param.grad = -update_direction[name]
            
        meta_optimizer.step()
        
        print(f"--- Meta-Update {meta_iter} hoàn tất. ---")

        # Lưu meta-model
        if meta_iter % 50 == 0:
            checkpoint_path = os.path.join("TrainMeta", f"meta_policy_{meta_iter}+{task_level}.pth")
            torch.save(meta_policy_net.state_dict(), checkpoint_path)
            print(f"Đã lưu meta-model tại: {checkpoint_path}")

if __name__ == "__main__":
    main()