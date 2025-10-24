# File: agents/dqn_agent.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.dqn_model import DuelingDQN_CNN
from buffers.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self,
                 input_channels,
                 num_actions,
                 device,
                 learning_rate=1e-4,
                 gamma=0.99,
                 target_update_freq=1000):
        
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # Khởi tạo 2 mạng: Policy (chính) và Target (bản sao)
        self.policy_net = DuelingDQN_CNN(input_channels, num_actions).to(self.device)
        self.target_net = DuelingDQN_CNN(input_channels, num_actions).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Sao chép trọng số ban đầu
        self.update_target_net()
        self.target_net.eval() # Target net chỉ dùng để dự đoán, không train

    def learn(self, buffer: ReplayBuffer, batch_size):
        """
        Thực hiện MỘT bước cập nhật DQN.
        """
        if len(buffer) < batch_size:
            return # Chưa đủ dữ liệu để học

        self.learn_step_counter += 1

        # 1. Lấy 1 batch dữ liệu
        states, actions, rewards, next_states, dones = buffer.sample(batch_size, self.device)
        
        # 2. Tính Q(s, a) (Current Q values)
        # Lấy Q-value của các hành động (actions) mà ta đã thực sự làm
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # 3. Tính Q_target(s', a') (Next Q values) - Đây là logic của Double DQN
        with torch.no_grad():
            # a. Dùng policy_net để chọn hành động tốt nhất (argmax) ở state kế tiếp
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            
            # b. Dùng target_net để lấy giá trị Q của hành động đó
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            
            # c. Tính target Q value
            # Q_target = reward + gamma * next_q_value * (1 - done)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        # 4. Tính Loss (dùng Huber Loss thay vì MSE cho ổn định)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 5. Cập nhật
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters(): # Clip gradient
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # 6. Cập nhật Target Net (nếu đến lúc)
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_net()

    def update_target_net(self):
        """Sao chép trọng số từ Policy Net sang Target Net."""
        print("--- Cập nhật Target Net ---")
        self.target_net.load_state_dict(self.policy_net.state_dict())