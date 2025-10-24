# File: models/dqn_model.py
import torch
import torch.nn as nn
import numpy as np
class DuelingDQN_CNN(nn.Module):
    """
    Kiến trúc mạng Dueling DQN.
    Input: (batch_size, 4, 84, 84)
    Output: Q-values (batch_size, num_actions)
    """
    def __init__(self, input_channels, num_actions):
        super(DuelingDQN_CNN, self).__init__()
        
        self.num_actions = num_actions
        
        # Lớp CNN cơ sở
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            cnn_out_dim = self.cnn_base(dummy_input).shape[1]

        # Lớp FC (fully-connected)
        # 1. Đầu ra cho Value Stream (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Chỉ 1 giá trị (Value)
        )
        
        # 2. Đầu ra cho Advantage Stream (A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions) # Mỗi hành động 1 giá trị (Advantage)
        )

    def forward(self, x):
        x = x / 255.0
        cnn_out = self.cnn_base(x)
        
        # Tính V(s) và A(s, a)
        value = self.value_stream(cnn_out)
        advantage = self.advantage_stream(cnn_out)
        
        # Kết hợp theo công thức Dueling DQN:
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

    # models/dqn_model.py

    def get_action(self, state, epsilon, device):
        """
        Hàm helper để chọn hành động (epsilon-greedy)
        (ĐÃ SỬA: Xử lý cả state là array (từ train) và tensor (từ evaluate))
        """
        if torch.rand(1).item() < epsilon:
            # Khám phá: chọn hành động ngẫu nhiên
            return torch.randint(0, self.num_actions, (1,)).item()
        else:
            # Khai thác: chọn hành động tốt nhất
            
            state_tensor = None # Khởi tạo
            
            # Kiểm tra xem state là array hay tensor
            if not isinstance(state, torch.Tensor):
                # Nếu là array (từ train_dqn.py), thì mới convert
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
            else:
                # Nếu đã là tensor (từ evaluate.py), dùng luôn
                state_tensor = state.to(device) # Chỉ cần đảm bảo nó đúng device
            
            with torch.no_grad():
                q_values = self.forward(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action