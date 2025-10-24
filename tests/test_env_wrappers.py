# File: tests/test_env_wrappers.py
import unittest
import numpy as np
import gym  # Cần import gym để kiểm tra kiểu dữ liệu

# Quan trọng: Đảm bảo bạn có thể import từ thư mục gốc
# Nếu chạy từ thư mục gốc (mario-rl-meta-learning), import này sẽ hoạt động
try:
    from envs.env_utils import create_mario_env, MarioRewardWrapper
except ImportError:
    # Nếu chạy trực tiếp file test, cần thêm đường dẫn thư mục gốc
    import sys
    import os
    # Lấy đường dẫn thư mục cha của thư mục 'tests'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from envs.env_utils import create_mario_env, MarioRewardWrapper

class TestEnvWrappers(unittest.TestCase):

    def test_create_mario_env_output_shape_and_type(self):
        """Kiểm tra shape và dtype của observation sau khi qua tất cả wrapper."""
        env = create_mario_env('SuperMarioBros-1-1-v3')
        obs = env.reset()
        env.close()

        # Kiểm tra shape (Channels, Height, Width)
        expected_shape = (4, 84, 84)
        self.assertEqual(obs.shape, expected_shape, f"Sai shape! Mong đợi {expected_shape}, nhận được {obs.shape}")

        # Kiểm tra kiểu dữ liệu (phải là uint8 để tiết kiệm bộ nhớ)
        # Lưu ý: FrameStack trả về LazyFrames, cần chuyển thành array để kiểm tra dtype
        obs_array = np.array(obs)
        self.assertEqual(obs_array.dtype, np.uint8, f"Sai dtype! Mong đợi uint8, nhận được {obs_array.dtype}")

    def test_reward_wrapper_go_right(self):
        """Kiểm tra MarioRewardWrapper có thưởng khi đi sang phải không."""
        env = gym.make('SuperMarioBros-1-1-v3')
        env = MarioRewardWrapper(env) # Chỉ test wrapper này

        _ = env.reset()
        initial_x = env.current_x

        # Giả lập đi sang phải (action = 1 trong SIMPLE_MOVEMENT)
        _, reward, _, info = env.step(1)
        env.close()

        # x_pos phải tăng
        self.assertGreater(info['x_pos'], initial_x, "Đi phải nhưng x_pos không tăng?")
        # Phải có phần thưởng dương nhỏ (vì phá kỷ lục max_x)
        self.assertGreater(reward, 0, "Đi phải lần đầu mà không được thưởng?")

    def test_reward_wrapper_stand_still_penalty(self):
        """Kiểm tra MarioRewardWrapper có phạt khi đứng yên quá lâu không."""
        env = gym.make('SuperMarioBros-1-1-v3')
        env = MarioRewardWrapper(env)

        _ = env.reset()
        total_reward = 0
        penalty_received = False

        # Giả lập đứng yên (action = 0 là NOOP) 150 lần
        for _ in range(150):
            _, reward, _, _ = env.step(0)
            total_reward += reward
            # Kiểm tra xem có bị phạt không (reward < 0)
            if reward < 0:
                penalty_received = True
                break
        env.close()

        self.assertTrue(penalty_received, "Đứng yên 150 frames mà không bị phạt?")

    def test_reward_wrapper_death_penalty(self):
        """Kiểm tra MarioRewardWrapper có phạt nặng khi chết không."""
        env = gym.make('SuperMarioBros-1-1-v3')
        env = MarioRewardWrapper(env)

        _ = env.reset()
        done = False
        final_reward = 0

        # Chơi cho đến khi chết (tìm cách chết nhanh, ví dụ rơi xuống hố)
        # Lưu ý: Test này có thể không ổn định 100%
        # Cách đơn giản là chạy sang phải và hy vọng rơi hố đầu tiên
        for _ in range(300): # Giới hạn số bước để tránh treo
             # Action 1: RIGHT
            _, reward, done, info = env.step(1)
            if done:
                final_reward = reward
                # Đảm bảo chết thực sự (life = 0)
                self.assertEqual(info['life'], 0, f"Done=True nhưng life={info['life']}?")
                break
        env.close()

        self.assertTrue(done, "Chạy 300 bước mà chưa chết để test?")
        # Kiểm tra xem có bị phạt nặng không (reward << 0)
        self.assertLess(final_reward, -5.0, f"Chết mà reward chỉ là {final_reward}, không đủ phạt?")

if __name__ == '__main__':
    unittest.main()