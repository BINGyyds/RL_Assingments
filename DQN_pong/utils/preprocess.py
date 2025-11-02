import numpy as np
import cv2

# def greyscale(state):
#     """
#     Preprocess state (210, 160, 3) image into
#     a (80, 80, 1) image in grey scale
#     """
#     state = np.reshape(state, [210, 160, 3]).astype(np.float32)

#     # grey scale
#     state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

#     # karpathy
#     state = state[35:195]  # crop
#     state = state[::2,::2] # downsample by factor of 2

#     state = state[:, :, np.newaxis]

#     return state.astype(np.uint8)

def greyscale(state):
    """
    Preprocesses a state to 84x84 grayscale.
    Handles both RGB (H, W, 3) and grayscale (H, W) inputs.
    """
    # 1. 确保 state 是浮点数类型以便计算
    state = state.astype(np.float32)

    # 2. 如果 state 是彩色的 (3个维度)，则转为灰度
    if state.ndim == 3 and state.shape[2] == 3:
        state = np.mean(state, axis=2)

    # 3. 归一化 (可选，但通常是好习惯)
    # state = state / 255.0

    # 4. 使用 OpenCV 进行缩放，确保输出是 84x84
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)

    # 5. 增加一个通道维度，使其形状变为 (84, 84, 1)
    state = np.expand_dims(state, axis=2)

    # 6. 将类型转回 uint8 以便存储在经验池中
    return state.astype(np.uint8)


def blackandwhite(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    # erase background
    state[state==144] = 0
    state[state==109] = 0
    state[state!=0] = 1

    # karpathy
    state = state[35:195]  # crop
    state = state[::2,::2, 0] # downsample by factor of 2

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)