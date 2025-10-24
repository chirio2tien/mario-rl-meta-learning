# File: buffers/replay_buffer.py
import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Buffer lưu trữ (s, a, r, s', done) cho DQN.
    """
    def __init__(self, capacity):
        # deque (double-ended queue) tự động xóa phần tử cũ khi đầy
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        """Thêm một transition vào buffer."""
        # Chuyển về CPU và kiểu dữ liệu cơ bản để tiết kiệm bộ nhớ
        state = np.array(state)
        next_state = np.array(next_state)
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """Lấy một batch ngẫu nhiên từ buffer."""
        # Lấy ngẫu nhiên batch_size mẫu
        batch = random.sample(self.buffer, batch_size)
        
        # Tách batch thành các tensor riêng biệt
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=device).unsqueeze(1) # (batch_size, 1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1) # (batch_size, 1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1) # (batch_size, 1)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)