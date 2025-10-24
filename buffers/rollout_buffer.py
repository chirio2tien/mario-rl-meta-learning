# File: rollout_buffer.py
import torch
import numpy as np

class RolloutBuffer:
    """
    Buffer để lưu trữ các trajectories (trải nghiệm) cho PPO.
    """
    def __init__(self, buffer_size, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Khởi tạo các list để lưu trữ
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        
        self.pos = 0 # Vị trí con trỏ
        self.full = False

    def add(self, state, action, reward, done, log_prob, value):
        # Thêm dữ liệu vào buffer
        if len(self.states) < self.buffer_size:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.dones.append(None)
            self.log_probs.append(None)
            self.values.append(None)
        
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def compute_advantages(self, last_value, last_done):
        """
        Tính toán Advantage (GAE) và Returns (Mục 4.8).
        Phải được gọi sau khi thu thập đủ 1 batch (rollout).
        """
        # Chuyển list sang tensor
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze()

        gae = 0
        self.advantages = torch.zeros_like(rewards)
        self.returns = torch.zeros_like(rewards)
        
        # Loop ngược từ cuối về đầu
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            # Tính delta (TD Error)
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            
            # Tính GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            
            # Lưu Advantage và Return (Return = GAE + Value)
            self.advantages[t] = gae
            self.returns[t] = gae + values[t]
            
        # Lưu lại để dùng trong PPO update
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.returns = self.returns
        self.advantages = self.advantages

    def get_batches(self, mini_batch_size, device):
        """
        Một generator để chia buffer thành các mini-batch ngẫu nhiên.
        """
        # Lấy index ngẫu nhiên
        indices = np.random.permutation(self.buffer_size)
        
        for start_idx in range(0, self.buffer_size, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                self.states[batch_indices].to(device),
                self.actions[batch_indices].to(device),
                self.log_probs[batch_indices].to(device),
                self.advantages[batch_indices].to(device),
                self.returns[batch_indices].to(device),
            )

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.pos = 0
        self.full = False