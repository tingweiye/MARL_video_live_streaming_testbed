import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from algorithms.components.replay_memory import ReplayMemory, Transition

USE_CUDA = torch.cuda.is_available()
GAMMA = 0.8
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


class MetaController(nn.Module):
    def __init__(self, in_features=4, out_features=6):
        """
        Initialize a Meta-Controller of Hierarchical DQN network
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(MetaController, self).__init__()
        self.r_fc = nn.Linear(3, 128)
        self.w_fc = nn.Linear(1, 128)
        self.rw_fc_0 = nn.Linear(256, 128)
        # self.rw_fc_1 = nn.Linear(256, 128)
        
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, out_features)

    def forward(self, x):
        rates_batch   = x[:, 0:3]
        weights_batch = x[:, 3:4]
        
        x_r = F.relu(self.r_fc(rates_batch))
        x_w = F.relu(self.w_fc(weights_batch))
        x_rw = torch.cat([x_r, x_w], 1)
        x_o = F.relu(self.rw_fc_0(x_rw))
        # x_o = F.relu(self.rw_fc_1(x_o))
        
        V = self.V(x_o)
        A = self.A(x_o)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)
        
        return Q
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
 
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class Controller(nn.Module):
    def __init__(self, in_features=4, out_features=6, lookback=10):
        """
        Initialize a Controller(given goal) of h-DQN for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(Controller, self).__init__()
        self.input_channel = 1
        self.action_space = out_features
        kernel_size = 4
        channel_cnn = 128
        channel_fc = 128
        collaborate_fc = 256

        # self.bn = nn.BatchNorm1d(self.input_channel)

        self.conv1 = nn.Conv1d(self.input_channel, channel_cnn, kernel_size) # rate
        self.conv2 = nn.Conv1d(self.input_channel, channel_cnn, kernel_size) # bw
        self.conv3 = nn.Conv1d(self.input_channel, channel_cnn, kernel_size) # buffer
        self.conv4 = nn.Conv1d(self.input_channel, channel_cnn, kernel_size) # goal
        # self.b_fc = nn.Linear(self.input_channel, channel_fc) # buffer
        # self.g_fc_0 = nn.Linear(self.input_channel, channel_fc) # goal

        incoming_size = 3*channel_cnn*(lookback-kernel_size+1)  # rate, bw, buffer
        # incoming_size = 2*channel_cnn*(lookback-kernel_size+1) + channel_fc  # rate, bw, buffer

        self.s_fc = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        self.g_fc_1 = nn.Linear(channel_cnn*(lookback-kernel_size+1), channel_fc) # goal

        self.fc_0 = nn.Linear(in_features=channel_fc*2, out_features=collaborate_fc)
        self.V = nn.Linear(in_features=collaborate_fc, out_features=1)
        self.A = nn.Linear(in_features=collaborate_fc, out_features=self.action_space)

    def forward(self, inputs):
        # rates_batch = inputs[:, 0:1, :]
        # rates_batch = self.bn(rates_batch)
        
        # bandwitdh_batch = inputs[:, 1:2, :]
        # bandwitdh_batch = self.bn(bandwitdh_batch)

        x_r = F.relu(self.conv1(inputs[:, 0:1, :]))
        x_bw = F.relu(self.conv2(inputs[:, 1:2, :]))
        # x_3 = F.relu(self.actor_conv3(download_time_batch))
        x_b = F.relu(self.conv3(inputs[:, 2:3, :]))
        x_g = F.relu(self.conv4(inputs[:, 3:4, :]))

        x_r = x_r.view(-1, self.num_flat_features(x_r))
        x_bw = x_bw.view(-1, self.num_flat_features(x_bw))
        # x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_b = x_b.view(-1, self.num_flat_features(x_b))
        x_g = x_g.view(-1, self.num_flat_features(x_g))

        x_s = torch.cat([x_r, x_bw, x_b], 1)
        x_s = F.relu(self.s_fc(x_s))        
        x_g = F.relu(self.g_fc_1(x_g))
        x_s = x_s.view(-1, self.num_flat_features(x_s))
        x_g = x_g.view(-1, self.num_flat_features(x_g))
        
        x = torch.cat([x_s, x_g], 1)
        x = F.relu(self.fc_0(x))
        
        V = self.V(x)
        A = self.A(x)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q
    
    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
 
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
    

class hDQN():
    """
    The Hierarchical-DQN Agent
    Parameters
    ----------
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        num_goal: int
            The number of goal that agent can choose from
        num_action: int
            The number of action that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
    """
    def __init__(self,
                 optimizer_spec,
                 num_goal=6,
                 num_action=6,
                 replay_memory_size=10000,
                 batch_size=64,
                 tau=0.01):
        ###############
        # BUILD MODEL #
        ###############
        self.num_goal = num_goal
        self.num_action = num_action
        self.batch_size = batch_size
        self.tau = tau
        # Construct meta-controller and controller
        self.meta_controller = MetaController().type(dtype)
        self.target_meta_controller = MetaController().type(dtype)
        self.controller = Controller().type(dtype)
        self.target_controller = Controller().type(dtype)
        # Construct the optimizers for meta-controller and controller
        self.meta_optimizer = optimizer_spec.constructor(self.meta_controller.parameters(), **optimizer_spec.kwargs)
        self.ctrl_optimizer = optimizer_spec.constructor(self.controller.parameters(), **optimizer_spec.kwargs)
        # Construct the replay memory for meta-controller and controller
        self.meta_replay_memory = ReplayMemory(replay_memory_size)
        self.ctrl_replay_memory = ReplayMemory(replay_memory_size)
        # Training control parameters
        self.meta_count = 0
        self.local_count = 0

    def select_goal(self, state, epilson):
        sample = random.random()
        if sample > epilson:
            # state = torch.from_numpy(state).type(dtype)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                return self.meta_controller(Variable(state)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(self.num_goal)])

    def select_action(self, joint_state_goal, epilson):
        sample = random.random()
        if sample > epilson:
            # joint_state_goal = torch.from_numpy(joint_state_goal).type(dtype)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                return self.controller(Variable(joint_state_goal)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(self.num_action)])
        
    def update_meta_controller_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
 
        for q_target_params, q_eval_params in zip(self.target_meta_controller.parameters(), self.meta_controller.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
            
    def update_controller_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
 
        for q_target_params, q_eval_params in zip(self.target_controller.parameters(), self.controller.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
            
    def save_controller_model(self, epoch):
        add_str = 'dqn'
        model_save_path = "./models/hmarl/local/%s_%s_%d_ctrl.model" %(str('abr'), add_str, int(epoch))
        self.controller.save_checkpoint(model_save_path)
    
    def save_meta_controller_model(self, epoch):
        add_str = 'dqn'
        model_save_path = "./models/hmarl/meta/%s_%s_%d_meta.model" %(str('abr'), add_str, int(epoch))
        self.meta_controller.save_checkpoint(model_save_path)
        
    def load_controller_model(self, epoch):
        add_str = 'dqn'
        model_load_path = "./models/hmarl/local/%s_%s_%d_ctrl.model" %(str('abr'), add_str, int(epoch))
        self.controller.load_checkpoint(model_load_path)
        
    def load_meta_controller_model(self, epoch):
        add_str = 'dqn'
        model_load_path = "./models/hmarl/meta/%s_%s_%d_meta.model" %(str('abr'), add_str, int(epoch))
        self.meta_controller.load_checkpoint(model_load_path)
        
    def update_meta_controller(self):
        if len(self.meta_replay_memory) < self.batch_size:
            return
        state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask = \
            self.meta_replay_memory.sample(self.batch_size)
        state_batch = Variable(torch.from_numpy(state_batch).type(dtype))
        goal_batch = Variable(torch.from_numpy(goal_batch).long())
        next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype))
        ex_reward_batch = Variable(torch.from_numpy(ex_reward_batch).type(dtype)).unsqueeze(1)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype).unsqueeze(1)
        done_mask = Variable(torch.from_numpy(done_mask)).type(dtype).unsqueeze(1)
        if USE_CUDA:
            goal_batch = goal_batch.cuda()
        # Compute current Q value, meta_controller takes only state and output value for every state-goal pair
        # We choose Q based on goal chosen.
        current_Q_values = self.meta_controller(state_batch).gather(1, goal_batch)

        # Double DQN update
        with torch.no_grad():
            q_ = self.target_meta_controller(next_state_batch)
            # Compute next Q value based on which goal gives max Q values
            next_max_actions = self.meta_controller(next_state_batch).detach().max(1)[1].unsqueeze(1)
            next_max_q = q_.gather(1, next_max_actions)
            # next_Q_values = q_.gather(1, next)
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = ex_reward_batch + (GAMMA * next_Q_values)
            
        # Compute Bellman error (using Huber loss)
        loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

        # Copy Q to target Q before updating parameters of Q
        self.update_meta_controller_parameters()

        # Optimize the model
        self.meta_optimizer.zero_grad()
        loss.backward()
        for param in self.meta_controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step()

    def update_controller(self):
        if len(self.ctrl_replay_memory) < self.batch_size:
            return
        state_goal_batch, action_batch, next_state_goal_batch, in_reward_batch, done_mask = \
            self.ctrl_replay_memory.sample(self.batch_size)
        state_goal_batch = Variable(torch.from_numpy(state_goal_batch).type(dtype))
        action_batch = Variable(torch.from_numpy(action_batch).long())
        next_state_goal_batch = Variable(torch.from_numpy(next_state_goal_batch).type(dtype))
        in_reward_batch = Variable(torch.from_numpy(in_reward_batch).type(dtype)).unsqueeze(1)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype).unsqueeze(1)
        done_mask = Variable(torch.from_numpy(done_mask)).type(dtype).unsqueeze(1)
        if USE_CUDA:
            action_batch = action_batch.cuda()
        # Compute current Q value, controller takes only (state, goal) and output value for every (state, goal)-action pair
        # We choose Q based on action taken.
        current_Q_values = self.controller(state_goal_batch).gather(1, action_batch)
        # Double DQN update 
        with torch.no_grad():
            q_ = self.target_controller(next_state_goal_batch)
            # Compute next Q value based on which goal gives max Q values
            next_max_actions = self.controller(next_state_goal_batch).detach().max(1)[1].unsqueeze(1)
            next_max_q = q_.gather(1, next_max_actions)
            # next_Q_values = q_.gather(1, next)
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = in_reward_batch + (GAMMA * next_Q_values)
        # Compute Bellman error (using Huber loss)
        loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

        # Soft update Q to target Q before updating parameters of Q
        self.update_controller_parameters()
        # count = 0
        # print()
        # for q_target_params, q_eval_params in zip(self.target_controller.parameters(), self.controller.parameters()):
        #     count += 1
        #     if (count == 3):
        #         print(q_target_params[-1])
        #         break
        # Optimize the model
        self.ctrl_optimizer.zero_grad()
        loss.backward()
        for param in self.controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.ctrl_optimizer.step()
        
        
# if __name__ == '__main__':
#     mc = MetaController()
#     print(mc(torch.from_numpy(np.array([1, 2, 3, 4])).unsqueeze(0).type(dtype)))
#     c = Controller()
#     a = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
#                 [1, 2, 3, 4, 5, 6, 7, 8],
#                 [1, 2, 3, 4, 5, 6, 7, 8],
#                 [1, 2, 3, 4, 5, 6, 7, 8]])
#     print(c(torch.from_numpy(a).unsqueeze(0).type(dtype)))
#     optimizer_spec = OptimizerSpec(
#         constructor=optim.RMSprop,
#         kwargs=dict(lr=0.00025, alpha=0.95, eps=0.01),
#     )
#     hdqn = hDQN(optimizer_spec)
    
#     for e in range(500):
#         state, next_state = a, a
#         action = hdqn.select_action(torch.from_numpy(a).unsqueeze(0).type(dtype), 0.1)
#         hdqn.ctrl_replay_memory.push(a, action, a, 1, 0)
#     hdqn.update_controller()
#     for e in range(500):
#         state, next_state = np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])
#         goal = hdqn.select_goal(torch.from_numpy(np.array([1, 2, 3, 4])).unsqueeze(0).type(dtype), 0.1)
#         hdqn.meta_replay_memory.push(state, goal, next_state, 1, 0)
#     hdqn.update_meta_controller()

