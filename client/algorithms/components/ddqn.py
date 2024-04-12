import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from algorithms.components.replay_memory import ReplayBuffer, Transition

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

USE_CUDA = torch.cuda.is_available()
GAMMA = 0.8
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

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
        self.conv3 = nn.Conv1d(self.input_channel, channel_cnn, kernel_size) # idle
        self.conv4 = nn.Conv1d(self.input_channel, channel_cnn, kernel_size) # buffer
        # self.b_fc = nn.Linear(self.input_channel, channel_fc) # buffer
        # self.g_fc_0 = nn.Linear(self.input_channel, channel_fc) # goal

        incoming_size = 4*channel_cnn*(lookback-kernel_size+1)  # rate, bw, idle, buffer
        # incoming_size = 2*channel_cnn*(lookback-kernel_size+1) + channel_fc  # rate, bw, buffer

        self.s_fc = nn.Linear(in_features=incoming_size, out_features=channel_fc)

        self.fc_0 = nn.Linear(in_features=channel_fc, out_features=collaborate_fc)
        self.V = nn.Linear(in_features=collaborate_fc, out_features=1)
        self.A = nn.Linear(in_features=collaborate_fc, out_features=self.action_space)

    def forward(self, inputs):
        # rates_batch = inputs[:, 0:1, :]
        # rates_batch = self.bn(rates_batch)
        
        # bandwitdh_batch = inputs[:, 1:2, :]
        # bandwitdh_batch = self.bn(bandwitdh_batch)

        x_r = F.relu(self.conv1(inputs[:, 0:1, :]))
        x_bw = F.relu(self.conv2(inputs[:, 1:2, :]))
        x_i = F.relu(self.conv3(inputs[:, 2:3, :]))
        x_b = F.relu(self.conv4(inputs[:, 3:4, :]))

        x_r = x_r.view(-1, self.num_flat_features(x_r))
        x_bw = x_bw.view(-1, self.num_flat_features(x_bw))
        x_i = x_i.view(-1, self.num_flat_features(x_i))
        x_b = x_b.view(-1, self.num_flat_features(x_b))

        x_s = torch.cat([x_r, x_bw, x_i, x_b], 1)
        x_s = F.relu(self.s_fc(x_s))        
        x_s = x_s.view(-1, self.num_flat_features(x_s))
        
        # x = torch.cat([x_s, x_g], 1)
        x = F.relu(self.fc_0(x_s))
        
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
        
class ddqn():
    
    def __init__(self,
                 optimizer_spec,
                 num_action=6,
                 replay_memory_size=10000,
                 batch_size=64,
                 tau=0.01):
        ###############
        # BUILD MODEL #
        ###############
        self.num_action = num_action
        self.batch_size = batch_size
        self.tau = tau
        # Construct controller
        self.controller = Controller().type(dtype)
        self.target_controller = Controller().type(dtype)
        # Construct the optimizers for meta-controller and controller
        self.ctrl_optimizer = optimizer_spec.constructor(self.controller.parameters(), **optimizer_spec.kwargs)
        # Construct the replay memory for meta-controller and controller
        self.ctrl_replay_memory = ReplayBuffer(replay_memory_size)
        self.local_count = -1

    def select_action(self, joint_state_goal, epilson):
        sample = random.random()
        if sample > epilson:
            # joint_state_goal = torch.from_numpy(joint_state_goal).type(dtype)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                return self.controller(Variable(joint_state_goal)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(self.num_action)])
        
    def update_controller_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
 
        for q_target_params, q_eval_params in zip(self.target_controller.parameters(), self.controller.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
            
    def save_controller_model(self, epoch):
        add_str = 'dqn'
        model_save_path = "./models/pensieve/%s_%s_%d_ctrl.model" %(str('abr'), add_str, int(epoch))
        self.controller.save_checkpoint(model_save_path)
        
    def load_controller_model(self, file_path):
        self.controller.load_checkpoint(file_path)
        
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
        # count = 0
        # print()
        # for q_target_params, q_eval_params in zip(self.target_controller.parameters(), self.controller.parameters()):
        #     count += 1
        #     if (count == 3):
        #         print(q_target_params[-1])
        #         break
        # Optimize the model
        self.update_controller_parameters()
        
        self.ctrl_optimizer.zero_grad()
        loss.backward()
        for param in self.controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.ctrl_optimizer.step()