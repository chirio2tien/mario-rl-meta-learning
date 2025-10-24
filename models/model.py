# File: model.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticCNN(nn.Module):
    """
    Kiến trúc mạng Actor-Critic dùng chung (CNN-base).
    Input: (batch_size, 4, 84, 84) - 4 frame ảnh xám
    """
    def __init__(self, input_channels, num_actions):
        super(ActorCriticCNN, self).__init__()
        
        # Lớp CNN cơ sở (trích xuất đặc trưng)
        # Đây là kiến trúc "Nature CNN" chuẩn
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Tính toán kích thước đầu ra của CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            cnn_out_dim = self.cnn_base(dummy_input).shape[1]
            
        # Lớp FC (fully-connected)
        self.fc_base = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
        )
        
        # 1. Đầu ra cho Actor (Policy Head)
        # Quyết định hành động nào sẽ làm
        self.actor_head = nn.Linear(512, num_actions)
        
        # 2. Đầu ra cho Critic (Value Head)
        # Đánh giá trạng thái hiện tại "tốt" như thế nào
        self.critic_head = nn.Linear(512, 1)

    def forward(self, x):
        # Chuẩn hóa ảnh 
        x = x / 255.0
        
        cnn_out = self.cnn_base(x)
        fc_out = self.fc_base(cnn_out)
        
        policy_logits = self.actor_head(fc_out) # Logits cho action
        value = self.critic_head(fc_out)        # Giá trị (value) của state
        
        return policy_logits, value

    def get_action(self, state, deterministic=False):
        """
        Hàm helper để CHỌN HÀNH ĐỘNG (dùng khi rollout).
        """
        logits, value = self.forward(state)
        
        # Tạo phân phối xác suất
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample() # Sample hành động
            
        log_prob = dist.log_prob(action)
        
        return action.detach(), log_prob.detach(), value.detach()

    def evaluate_action(self, state, action):
        """
        Hàm helper để ĐÁNH GIÁ HÀNH ĐỘNG (dùng khi update PPO).
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy() # (Mục 4.6 - Entropy bonus)
        
        return log_prob, value, entropy