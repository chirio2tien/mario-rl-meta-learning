# File: ppo_agent.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.model import ActorCriticCNN
from buffers.rollout_buffer import RolloutBuffer

class PPOAgent:
    def __init__(self,
                 model: ActorCriticCNN,
                 learning_rate=2.5e-4,
                 clip_epsilon=0.1,  # (Mục 4.8)
                 ppo_epochs=4,
                 mini_batch_size=32,
                 value_loss_coef=0.5,
                 entropy_coef=0.1): # (Mục 4.6 - Entropy bonus)
        
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Siêu tham số PPO
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        self.device = next(model.parameters()).device

    def learn(self, buffer: RolloutBuffer):
        """
        Thực hiện MỘT bước cập nhật PPO (gồm nhiều epochs) 
        dựa trên dữ liệu trong buffer.
        """
        
        # Chuẩn hóa advantages (rất quan trọng để ổn định)
        
        # Vòng lặp PPO Epochs
        for _ in range(self.ppo_epochs):
            
            # Lấy các mini-batch ngẫu nhiên
            for batch in buffer.get_batches(self.mini_batch_size, self.device):
                states, actions, old_log_probs, advantages, returns = batch

# BÊN TRONG VÒNG LẶP:
# DÒNG ĐÃ SỬA:
                advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)                # Đánh giá lại hành động (lấy new_log_probs, new_values, entropy)
                new_log_probs, new_values, entropy = self.model.evaluate_action(states, actions)
                
                new_values = new_values.squeeze()

                # --- 1. Tính Policy Loss (PPO-Clip) ---
                # Tỷ lệ (ratio)
                ratio = (new_log_probs - old_log_probs).exp()
                
                # Hàm mục tiêu surrogate 1 (không clip)
                surr1 = ratio * advantages_normalized
                # Hàm mục tiêu surrogate 2 (có clip)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_normalized
                
                # PPO loss là min của 2 cái trên, lấy mean và đổi dấu (vì ta dùng gradient descent)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # --- 2. Tính Value Loss (MSE) ---
                # (Mục 4.8 - Tối ưu)
                value_loss = F.mse_loss(new_values, returns)
                
                # --- 3. Tính Loss tổng ---
                # Trừ entropy loss để khuyến khích khám phá (Mục 4.6)
                loss = (policy_loss + 
                        self.value_loss_coef * value_loss - 
                        self.entropy_coef * entropy.mean())
                
                # --- 4. Cập nhật ---
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # Clip gradient
                self.optimizer.step()