# Deep Reinforcement Learning: DQN Experiments
## 深度强化学习：DQN 实验

This repository contains the coursework for a deep reinforcement learning course. The experiments involve implementing and comparing DQN with a Target Network on the Flappy Bird game, and applying a Nature DQN model to play the Atari Pong game using OpenAI Gym.

（本代码库包含了深度强化学习课程的作业。实验内容包括在 Flappy Bird 游戏上实现并对比带有 Target Network 的 DQN，以及应用 Nature DQN 模型在 OpenAI Gym 环境中玩 Atari Pong 游戏。）

## Project Structure (项目结构)

```
.
├── Flappybird-pytorch/       # 第一部分: 在 Flappy Bird 上实现 DQN
│   ├── assets/               # 游戏资源文件
│   ├── game/                 # 游戏环境代码
│   ├── pretrained_model/     # 预训练模型存放目录
│   └── dqn.py                # DQN 模型与训练脚本
├── DQN_pong/                 # 第二部分: 在 Atari Pong 上实现 DQN
│   ├── configs/              # 配置文件
│   ├── results/              # 实验结果 (得分图、游戏视频)
│   ├── utils/                # 工具函数 (预处理、环境包装器)
│   ├── weights/              # 预训练权重存放目录
│   ├── q6_nature_torch.py    # Nature DQN 模型定义与简单任务训练
│   └── q7_train_atari_nature.py  # 加载模型玩 Pong 游戏
└── README.md                 # 本说明文件
```

## Environment Setup (环境配置)

This project is developed and tested on a **Linux (Ubuntu-like)** system with an **NVIDIA GPU**. The following steps will guide you to create a fully functional Conda environment.

（本项目在 **Linux (类 Ubuntu) 系统** 和 **NVIDIA GPU** 上开发与测试。以下步骤将指导你创建一个完整的 Conda 环境。）

### Prerequisites (先决条件)

- **Conda**: Make sure you have Anaconda or Miniconda installed. You can download it from the [official website](https://www.anaconda.com/products/distribution).
  （请确保你已安装 Anaconda 或 Miniconda。可从 [官方网站](https://www.anaconda.com/products/distribution) 下载。）
- **NVIDIA Driver**: A compatible NVIDIA driver must be installed on your system. You can check this by running `nvidia-smi` in your terminal.
  （必须已安装兼容的 NVIDIA 显卡驱动。你可以在终端运行 `nvidia-smi` 来检查。）

### Step 1: Create a Clean Conda Environment (第一步：创建纯净的 Conda 环境)

We will create a new Conda environment named `rl_env` with Python 3.10. Using a dedicated environment prevents conflicts with other projects.

（我们将创建一个名为 `rl_env` 并使用 Python 3.10 的新环境。使用独立环境可以避免与其他项目的包冲突。）

```bash
conda create -n rl_env python=3.10```

### Step 2: Activate the Environment (第二步：激活环境)

Before installing packages, activate the newly created environment. You will need to do this every time you work on the project.

（在安装包之前，请激活新创建的环境。每次开始进行项目时都需要执行此操作。）

```bash
conda activate rl_env
```

### Step 3: Install PyTorch with CUDA Support (via Pip) (第三步：通过 Pip 安装支持 CUDA 的 PyTorch)

To ensure maximum compatibility with the NVIDIA driver, we will install PyTorch and its related libraries using `pip` and specifying the official download index for CUDA 12.1.

（为确保与 NVIDIA 驱动的最佳兼容性，我们将使用 `pip` 并指定 CUDA 12.1 的官方下载源来安装 PyTorch 及其相关库。）

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

After installation, you can verify that PyTorch can see your GPU by running:
（安装后，你可以通过运行以下命令来验证 PyTorch 是否能识别到你的 GPU：）

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
The output should be `CUDA available: True`. （输出应为 `CUDA available: True`。）

### Step 4: Install Project Dependencies (第四步：安装项目依赖)

Next, install all other required libraries, including OpenAI Gym for the Atari environment and a specific version of NumPy for compatibility.

（接下来，安装所有其他必需的库，包括用于 Atari 环境的 OpenAI Gym，以及一个用于兼容性的特定版本的 NumPy。）

```bash
pip install "gym[atari,accept-rom-license]" opencv-python numpy==1.26.4
```
**Note (注意):** We explicitly install `numpy==1.26.4` to avoid compatibility issues between the older `gym` library and NumPy 2.0+. （我们明确指定安装 `numpy==1.26.4` 是为了避免旧版 `gym` 库与 NumPy 2.0+ 版本之间的兼容性问题。）

## How to Run the Experiments (如何运行实验)

### Part 1: DQN on Flappy Bird (第一部分：DQN 玩 Flappy Bird)

1.  Navigate to the project directory for the first part:
    （进入第一部分的实验目录：）
    ```bash
    cd Flappybird-pytorch
    ```
2.  To train the DQN model (either original or with Target Network), run:
    （若要训练 DQN 模型（无论是原始版还是带 Target Network 的版本），请运行：）
    ```bash
    python dqn.py train
    ```
3.  Training progress and reward logs will be printed to the console and saved to a `.csv` file.
    （训练进度和奖励日志将打印在控制台，并保存到一个 `.csv` 文件中。）

### Part 2: DQN on Atari Pong (第二部分：DQN 玩 Atari Pong)

**Before you start (开始前):**
-   You need to complete the code in `q6_nature_torch.py`. （你需要补全 `q6_nature_torch.py` 中的代码。）
-   Make sure the pre-trained model weights file (`5score.weights`) is placed inside the `DQN_pong/weights/` directory. （请确保预训练权重文件 `5score.weights` 已放置在 `DQN_pong/weights/` 目录下。）

1.  Navigate to the project directory for the second part:
    （进入第二部分的实验目录：）
    ```bash
    cd DQN_pong
    ```

2.  To train the model on a simple task (as required by the assignment), run:
    （若要在简单任务上训练模型（作业要求），请运行：）
    ```bash
    python q6_nature_torch.py
    ```
    This will generate a `score.png` in the `results` folder. （这将在 `results` 文件夹下生成一个 `score.png` 得分图。）

3.  To load the pre-trained model and watch it play Pong, run:
    （若要加载预训练模型并观看它玩 Pong 游戏，请运行：）
    ```bash
    python q7_train_atari_nature.py
    ```
    This will save a video of the gameplay in the `results` folder. （这将在 `results` 文件夹下保存一段游戏演示视频。）

## Important Code Fixes (重要代码修正)

If you are using the original provided code, you might encounter a `ModuleNotFoundError`. Please apply the following fix:

（如果你使用的是原始提供的框架代码，可能会遇到 `ModuleNotFoundError` 错误。请进行以下修正：）

-   **File (文件)**: `DQN_pong/q6_nature_torch.py`
-   **Original Line (原始代码行)**: `from torch.tensor import Tensor`
-   **Corrected Line (修正后代码行)**: `from torch import Tensor`

This change is necessary because `Tensor` is a class directly under the `torch` module.
（此修正是必需的，因为 `Tensor` 是一个直接位于 `torch` 模块下的类。）