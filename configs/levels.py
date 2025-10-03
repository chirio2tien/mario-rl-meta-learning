# Danh sách levels cho training, validation và testing

# Levels dùng cho meta-training (inner loop)
TRAIN_LEVELS = [
    'SuperMarioBros-1-1-v0',
    'SuperMarioBros-1-2-v0',
    'SuperMarioBros-1-3-v0',
    'SuperMarioBros-2-1-v0',
    'SuperMarioBros-2-2-v0',
    'SuperMarioBros-3-1-v0',
    'SuperMarioBros-3-2-v0',
    'SuperMarioBros-4-1-v0',
]

# Levels dùng cho validation (chọn checkpoint tốt nhất)
VAL_LEVELS = [
    'SuperMarioBros-2-3-v0',
    'SuperMarioBros-3-3-v0',
]

# Levels dùng cho meta-testing (unseen, đánh giá generalization)
TEST_LEVELS = [
    'SuperMarioBros-1-4-v0',
    'SuperMarioBros-4-2-v0',
    'SuperMarioBros-5-1-v0',
    'SuperMarioBros-6-1-v0',
    'SuperMarioBros-7-1-v0',
]

# Random level (tất cả stages)
RANDOM_LEVEL = 'SuperMarioBros-v0'