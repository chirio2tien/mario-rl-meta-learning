# Mario RL Meta-Learning

Deep Reinforcement Learning cho Super Mario Bros sử dụng Meta-Learning (Reptile/MAML), PPO và DQN với PyTorch.

## 🎯 Tính năng

- ✅ Hỗ trợ nhiều thuật toán RL: **PPO**, **DQN/Rainbow**, **Reptile Meta-Learning**
- ✅ Kiến trúc modular, dễ mở rộng
- ✅ Xử lý ảnh đầy đủ: grayscale, resize, frame stacking
- ✅ Quản lý train/test levels riêng biệt
- ✅ Logging với TensorBoard
- ✅ Vectorized environments cho training song song
- ✅ Unit tests

## 📁 Cấu trúc dự án

```
mario-rl-meta-learning/
├── configs/           # Cấu hình YAML và danh sách levels
├── envs/             # Environment wrappers
├── agents/           # Các thuật toán RL
├── models/           # Neural network architectures
├── buffers/          # Replay và rollout buffers
├── meta/             # Meta-learning components
├── utils/            # Utilities (logger, scheduler...)
├── tests/            # Unit tests
├── train.py          # Script huấn luyện chính
└── evaluate.py       # Script đánh giá
```

## 🚀 Cài đặt

### Yêu cầu
- Python 3.8+
- CUDA 11+ (khuyến nghị cho GPU)

### Các bước cài đặt

```bash
# Clone repository
git clone https://github.com/chirio2tien/mario-rl-meta-learning.git
cd mario-rl-meta-learning

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📖 Sử dụng

### Huấn luyện PPO
```bash
python train.py --config configs/ppo.yaml
```

### Huấn luyện DQN
```bash
python train.py --config configs/dqn.yaml
```

### Huấn luyện Reptile Meta-Learning
```bash
python train.py --config configs/reptile.yaml
```

### Đánh giá model
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --levels test
```

## ⚙️ Cấu hình

Chỉnh sửa file YAML trong `configs/`:

**PPO** (`configs/ppo.yaml`):
- Learning rate, clip range, entropy coefficient
- Number of parallel environments
- Rollout length, minibatch size

**DQN** (`configs/dqn.yaml`):
- Epsilon-greedy schedule
- Replay buffer size, batch size
- Target network update frequency

**Reptile** (`configs/reptile.yaml`):
- Inner/outer learning rates
- Number of inner steps
- Task batch size

**Levels** (`configs/levels.py`):
- Train levels: dùng cho meta-training
- Test levels: đánh giá generalization
- Validation levels: early stopping

## 📊 Kết quả

_Thêm kết quả thực nghiệm, đồ thị learning curves, video demo tại đây._

## 🏗️ Kiến trúc chi tiết

### Environment Pipeline
```
Raw Frame → Grayscale → Resize(84x84) → FrameStack(4) → SkipFrame(4) → Policy Network
```

### Meta-Learning Flow (Reptile)
```
Outer Loop (over task distribution):
  Sample batch of levels
  For each level:
    Inner Loop: Adapt policy (5 gradient steps)
    Store adapted weights
  Meta-update: Aggregate weights
```

## 🧪 Testing

```bash
# Chạy tất cả tests
pytest tests/

# Test cụ thể
pytest tests/test_env.py -v
```

## 📚 Tài liệu tham khảo

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Rainbow DQN](https://arxiv.org/abs/1710.02298)
- [Reptile Meta-Learning](https://arxiv.org/abs/1803.02999)
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

## 📝 License

MIT License

## 👤 Tác giả

**chirio2tien**
- GitHub: [@chirio2tien](https://github.com/chirio2tien)

## 🤝 Đóng góp

Contributions, issues và feature requests đều được hoan nghênh!

---

⭐ Nếu project hữu ích, hãy cho một star nhé!