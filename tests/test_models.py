# File: tests/test_models.py
import unittest
import torch

# Quan trọng: Đảm bảo bạn có thể import từ thư mục gốc
try:
    from models.model import ActorCriticCNN
    from models.dqn_model import DuelingDQN_CNN
except ImportError:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from models.model import ActorCriticCNN
    from models.dqn_model import DuelingDQN_CNN

class TestModels(unittest.TestCase):

    def test_actor_critic_cnn_shapes(self):
        """Kiểm tra input/output shape của ActorCriticCNN (PPO/Meta)."""
        input_channels = 4
        num_actions = 7
        batch_size = 5 # Test với batch size > 1

        model = ActorCriticCNN(input_channels, num_actions)
        # Tạo input giả (Batch, Channels, Height, Width)
        dummy_input = torch.randn(batch_size, input_channels, 84, 84)

        # Chạy forward pass
        policy_logits, value = model(dummy_input)

        # Kiểm tra output shape
        self.assertEqual(policy_logits.shape, (batch_size, num_actions), "Sai shape của policy_logits!")
        self.assertEqual(value.shape, (batch_size, 1), "Sai shape của value!")

    def test_dueling_dqn_cnn_shapes(self):
        """Kiểm tra input/output shape của DuelingDQN_CNN (DQN)."""
        input_channels = 4
        num_actions = 7
        batch_size = 5

        model = DuelingDQN_CNN(input_channels, num_actions)
        dummy_input = torch.randn(batch_size, input_channels, 84, 84)

        # Chạy forward pass
        q_values = model(dummy_input)

        # Kiểm tra output shape (phải là Q-values cho mỗi action)
        self.assertEqual(q_values.shape, (batch_size, num_actions), "Sai shape của q_values!")

if __name__ == '__main__':
    unittest.main()