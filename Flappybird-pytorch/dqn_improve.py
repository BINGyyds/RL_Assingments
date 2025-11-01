import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 500000
        self.replay_memory_size = 50000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data



def train(q_network, target_network, start):

    TOTAL_TRAINING_STEPS = 1000000  # 训练总步数
    TARGET_UPDATE_FREQUENCY = 1000  # 目标网络更新频率（步数）

    # define Adam optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([q_network.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = q_network.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(q_network.initial_epsilon, q_network.final_epsilon, q_network.number_of_iterations)

    # ==================== 重要：修改奖励日志文件名 ====================
    episode_number = 0
    total_episode_reward = 0.0
    # 修改文件名，防止覆盖原始DQN的实验数据！
    reward_log_file = "rewards_target_dqn.csv" 
    with open(reward_log_file, 'w') as f:
        f.write("episode,total_reward\n")
    # =================================================================

    # main infinite loop
    while iteration < TOTAL_TRAINING_STEPS:
        # get output from the neural network
        output = q_network(state)[0]

        # initialize action
        action = torch.zeros([q_network.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(q_network.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)

        # ++++++++++++++++++++++++++++ 新增代码：步骤2 ++++++++++++++++++++++++++++
        # 累加当前回合的奖励
        total_episode_reward += reward
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > q_network.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), q_network.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # # get output for the next state
        # output_1_batch = q_network(state_1_batch)

        # ==================== 核心修改点 1: 计算目标Q值 ====================
        # 使用 Target Network 来计算下一状态(state_1)的Q值
        # with torch.no_grad() 可以阻止梯度计算，节省资源
        with torch.no_grad():
            output_1_batch = target_network(state_1_batch)
        # ===================================================================

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + q_network.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(q_network(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        # ==================== 核心修改点 2: 定期更新目标网络 ====================
        # 每隔 TARGET_UPDATE_FREQUENCY 次迭代，就将 q_network 的权重复制给 target_network
        if iteration % TARGET_UPDATE_FREQUENCY == 0:
            print(f"--- Iteration {iteration}: Updating Target Network ---")
            target_network.load_state_dict(q_network.state_dict())
        # ======================================================================

        if iteration % 25000 == 0:
            torch.save(q_network, "pretrained_model/current_model_" + str(iteration) + ".pth")

        # ++++++++++++++++++++++++++++ 新增代码：步骤3 ++++++++++++++++++++++++++++
        # 检查回合是否结束
        if terminal:
            print(f"--- Episode {episode_number} Finished --- Total Reward: {total_episode_reward} ---")
            
            # 将回合数据写入CSV文件
            with open(reward_log_file, 'a') as f:
                f.write(f"{episode_number},{total_episode_reward}\n")
            
            # 为下一个回合重置变量
            total_episode_reward = 0.0
            episode_number += 1
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 为了避免刷屏，降低打印频率
        if iteration % 100 == 0: # 每100次迭代打印一次状态
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                np.max(output.cpu().detach().numpy()))


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            # 'pretrained_model/current_model_2000000.pth',
            'pretrained_model/current_model_75000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        # 1. 将 'model' 重命名为 'q_network'，以明确其角色
        q_network = NeuralNetwork()
        if cuda_is_available:
            q_network = q_network.cuda()
        q_network.apply(init_weights)
        
        # 2. 创建一个结构完全相同的 Target Network
        target_network = NeuralNetwork()
        if cuda_is_available:
            target_network = target_network.cuda()
            
        # 3. 在训练开始前，将 Q 网络的权重复制给 Target Network
        print("Initializing Target Network...")
        target_network.load_state_dict(q_network.state_dict())
        
        # 4. 将 Target Network 设置为评估模式（因为它不参与训练）
        target_network.eval()

        start = time.time()
        
        # 5. 将两个网络都传递给 train 函数
        train(q_network, target_network, start)


if __name__ == "__main__":
    main(sys.argv[1])
