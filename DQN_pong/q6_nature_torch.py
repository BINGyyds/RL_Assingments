import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.test_env import EnvTest
from q4_schedule import LinearExploration, LinearSchedule
from core.deep_q_learning_torch import DQN

from configs.q6_nature import config


class NatureQN(DQN):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        input_channels = n_channels * self.config.state_history

        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################
       
        # 根据 Nature DQN 论文定义网络结构
        # 输入维度: (batch, input_channels, img_height, img_width)
        # 例如: (batch, 4, 84, 84)
        
        # 主网络 (Q-Network)
        self.q_network = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 第二层卷积
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 第三层卷积
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # 展平层
            nn.Flatten(),
            # 第一个全连接层
            # 注意: in_features=3136 是针对 84x84 输入计算的。如果你的 env 是 (8,8,6),
            # 这里的输入尺寸需要重新计算，但对于 Atari Pong, 84x84 是标准。
            # 假设 config 中会提供正确的尺寸，或者我们在这里假设是 Atari 标准输入
            # (84-8)/4+1 = 20 -> (20-4)/2+1 = 9 -> (9-3)/1+1 = 7.  7*7*64 = 3136
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            # 输出层
            nn.Linear(in_features=512, out_features=num_actions)
        )

        # 目标网络 (Target Network) - 结构完全相同
        self.target_network = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )
        
        # 将网络移动到正确的设备 (CPU or GPU)
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        # 初始化时，将主网络的权重复制到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        state = state.permute(0,3,1,2)
        # print(f'Input shape after flattening = {input.shape}')
        if network == 'q_network':
            out = self.q_network(state)
        elif network == 'target_network':
            out = self.target_network(state)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out

    def update_target(self):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights.

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.

        Hint:
            1. look up saving and loading pytorch models
        """

        ##############################################################
        ################### YOUR CODE HERE - 1-2 lines ###############
        self.target_network.load_state_dict(self.q_network.state_dict())
        ##############################################################
        ######################## END YOUR CODE #######################


    def calc_loss(self, q_values : Tensor, target_q_values : Tensor,
                    actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        """
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a')
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')
            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            done_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

        Hint:
            You may find the following functions useful
                - torch.max
                - torch.sum
                - torch.nn.functional.one_hot
                - torch.nn.functional.mse_loss
        """
        # you may need this variable
        num_actions = self.env.action_space.n
        gamma = self.config.gamma

        ##############################################################
        ##################### YOUR CODE HERE - 3-5 lines #############
        
        # 1. 计算贝尔曼目标 (Bellman Target)
        # 首先，从 target_q_values 中找到下一状态的最大Q值: max_a' Q_target(s', a')
        next_q_values = torch.max(target_q_values, dim=1)[0]
        
        # done_mask 是一个布尔张量，我们需要把它转成 0 和 1
        # (1.0 - done_mask.float()) 的作用是：如果 done=True, 结果是0; 如果 done=False, 结果是1
        # 这样就能实现：如果游戏结束，未来奖励为0
        targets = rewards + gamma * next_q_values * (1.0 - done_mask.float())

        # 2. 从 q_values 中选出实际执行动作的Q值: Q(s, a)
        # actions 是一个 (batch_size,) 的索引，我们需要把它变成 (batch_size, 1) 的形状
        action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 3. 计算损失 (Loss)
        # 推荐使用 Huber Loss (Smooth L1 Loss) 来增加稳定性
        loss = F.smooth_l1_loss(action_q_values, targets)
        
        # 如果作业严格要求 MSE loss，也可以用下面的代码
        # loss = F.mse_loss(action_q_values, targets)

        return loss
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer(self):
        """
        Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
        parameters

        Hint:
            - Look up torch.optim.Adam
            - What are the input to the optimizer's constructor?
        """
        ##############################################################
        #################### YOUR CODE HERE - 1 line #############
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.00025)
        # self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-6)
        ##############################################################
        ######################## END YOUR CODE #######################


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((84, 84, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
