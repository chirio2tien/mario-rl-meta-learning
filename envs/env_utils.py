# File: env_utils.py
import gym
import numpy as np
import cv2  # Dùng cv2 cho resize nhanh hơn, cài bằng 'pip install opencv-python'
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT




class ResizeObservation(gym.ObservationWrapper):
    """
    Wrapper để resize observation về kích thước mong muốn.
    """
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            self.shape_tuple = (shape, shape)
        else:
            self.shape_tuple = tuple(shape)
        
        # Output (observation space) phải là 2D
        self.observation_space = Box(low=0, high=255, shape=self.shape_tuple, dtype=np.uint8)

    def observation(self, observation):
        # observation là 2D (từ GrayScale)
        # cv2.resize có (width, height)
        resized_obs = cv2.resize(observation, self.shape_tuple, interpolation=cv2.INTER_AREA)
        # resized_obs vẫn là 2D (84, 84)
        return resized_obs

class SkipFrame(gym.Wrapper):
    """
    Wrapper để lặp lại hành động trong `skip` frames.
    Trả về frame cuối cùng và tổng reward.
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def create_mario_env(level_name='SuperMarioBros-1-1-v0', skip_frames=4, frame_stack=4, resize_shape=84):
    """
    Hàm factory để tạo và bọc môi trường Mario.
    """
    env = gym.make(level_name)
    
    # 1. Rút gọn không gian hành động (Mục 4.2)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    env = MarioRewardWrapper(env)
    # 2. Lặp lại hành động (Frame Skipping)
    env = SkipFrame(env, skip=skip_frames)
    
    # 3. Biểu diễn trạng thái (Mục 4.1)
    env = GrayScaleObservation(env, keep_dim=False)       # (240, 256) -> (240, 256)
    env = ResizeObservation(env, (resize_shape, resize_shape)) # (240, 256) -> (84, 84)
    env = FrameStack(env, num_stack=frame_stack)         # (84, 84) -> (4, 84, 84)
                                                         # (FrameStack tự động đổi channel lên trước)
    return env
# (Dán code này vào CUỐI file envs/env_utils.py)

class MarioRewardWrapper(gym.Wrapper):
    """
    Wrapper để "dạy" AI (Reward Shaping).
    - Thưởng cho việc tiến lên (tăng max_x_pos).
    - Phạt vì đứng im hoặc đi lùi (hèn nhát).
    """
    def __init__(self, env):
        super(MarioRewardWrapper, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.max_x = 0
        self.stuck_counter = 0 # Bộ đếm "bị kẹt"

    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.max_x = 0
        self.stuck_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        
        # 1. PHẠT VÌ "HÈN NHÁT" (Đứng im hoặc đi lùi)
        if info['x_pos'] <= self.current_x:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        if self.stuck_counter > 100: # Nếu bị kẹt quá 100 frames
            reward -= 0.1 # Phạt -0.1
            self.stuck_counter = 0 # Reset

        # 2. THƯỞNG VÌ "TIẾN BỘ" (Phá kỷ lục x_pos)
        if info['x_pos'] > self.max_x:
            reward += 0.5 # Thưởng +0.5
            self.max_x = info['x_pos']

        # 3. Phạt nặng khi chết (để nó sợ chết)
        if done and info['life'] == 0:
            reward -= 10.0
            
        self.current_x = info['x_pos']
        
        return next_state, reward, done, info