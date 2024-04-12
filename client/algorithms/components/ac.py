"""_summary_
Reference: https://github.com/linnaeushuang/pensieve-pytorch/tree/master
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(torch.nn.Module):
    def __init__(self, action_space):
        super(Actor, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128 
        channel_fc = 128

        # self.bn = nn.BatchNorm1d(self.input_channel)
        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4) # rate
        self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4) # bw
        self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # freeze
        self.actor_conv4 = nn.Conv1d(self.input_channel, channel_cnn, 4) # idle
        self.actor_conv5 = nn.Conv1d(self.input_channel, channel_cnn, 4) # buffer
        # self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4) # rate
        # self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4) # bw
        # # self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # download time
        # # self.actor_fc_1 = nn.Linear(self.input_channel, channel_fc) # latency
        # self.actor_fc_2 = nn.Linear(self.input_channel, channel_fc) # freeze
        # self.actor_fc_3 = nn.Linear(self.input_channel, channel_fc) # idle
        # self.actor_fc_4 = nn.Linear(self.input_channel, channel_fc) # buffer len

        #===================Hide layer=========================
        incoming_size = 5*channel_cnn*5 #+ 3 * channel_fc  #+ 1 * channel_cnn*3

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=256)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=256, out_features=self.action_space)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):
        # rates_batch = inputs[:, 0:1, :] ## refer to env_train.py
        # rates_batch = self.bn(rates_batch)
        
        # bandwitdh_batch = inputs[:, 1:2, :]
        # bandwitdh_batch = self.bn(bandwitdh_batch)

        # download_time_batch = inputs[:, 2:3, :]
        # download_time_batch = self.bn(download_time_batch)

        x_1 = F.relu(self.actor_conv1(inputs[:, 0:1, :]))
        x_2 = F.relu(self.actor_conv2(inputs[:, 1:2, :]))
        x_3 = F.relu(self.actor_conv2(inputs[:, 2:3, :]))
        x_4 = F.relu(self.actor_conv2(inputs[:, 3:4, :]))
        x_5 = F.relu(self.actor_conv2(inputs[:, 4:5, :]))
        # x_3 = F.relu(self.actor_conv3(inputs[:, 2:3, :]))
        # x_4 = F.relu(self.actor_fc_1(inputs[:, 3:4, -1]))
        # x_5 = F.relu(self.actor_fc_2(inputs[:, 4:5, -1]))
        # x_6 = F.relu(self.actor_fc_3(inputs[:, 5:6, -1]))
        # x_7 = F.relu(self.actor_fc_4(inputs[:, 6:7, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        # x_6 = x_6.view(-1, self.num_flat_features(x_6))
        # x_7 = x_7.view(-1, self.num_flat_features(x_7))

        x = torch.cat([x_1, x_2, x_3, x_4, x_5], 1)
        x = F.relu(self.fc1(x))
        # actor
        # actor = F.relu(self.fc1(x))
        # actor = F.relu(self.fc2(actor))
        actor = F.softmax(self.fc3(x), dim=1)
        return actor

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

class Critic(torch.nn.Module):
    def __init__(self, action_space):
        super(Critic, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128 
        channel_fc = 128

        # self.bn = nn.BatchNorm1d(self.input_channel)

        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4) # rate
        self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4) # bw
        self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # freeze
        self.actor_conv4 = nn.Conv1d(self.input_channel, channel_cnn, 4) # idle
        self.actor_conv5 = nn.Conv1d(self.input_channel, channel_cnn, 4) # buffer
        # self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # download time
        # self.actor_fc_1 = nn.Linear(self.input_channel, channel_fc) # latency
        # self.actor_fc_2 = nn.Linear(self.input_channel, channel_fc) # freeze
        # self.actor_fc_3 = nn.Linear(self.input_channel, channel_fc) # idle
        # self.actor_fc_4 = nn.Linear(self.input_channel, channel_fc) # buffer len

        #===================Hide layer=========================
        incoming_size = 5*channel_cnn*5 #+ 3 * channel_fc  #+ 1 * channel_cnn*3

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=256)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

    def forward(self, inputs):
        # rates_batch = inputs[:, 0:1, :] ## refer to env_train.py
        # rates_batch = self.bn(rates_batch)
        
        # bandwitdh_batch = inputs[:, 1:2, :]
        # bandwitdh_batch = self.bn(bandwitdh_batch)

        # download_time_batch = inputs[:, 2:3, :]
        # download_time_batch = self.bn(download_time_batch)

        x_1 = F.relu(self.actor_conv1(inputs[:, 0:1, :]))
        x_2 = F.relu(self.actor_conv2(inputs[:, 1:2, :]))
        # x_3 = F.relu(self.actor_conv3(inputs[:, 2:3, :]))
        # x_4 = F.relu(self.actor_fc_1(inputs[:, 3:4, -1]))
        x_3 = F.relu(self.actor_conv3(inputs[:, 2:3, :]))
        x_4 = F.relu(self.actor_conv4(inputs[:, 3:4, :]))
        x_5 = F.relu(self.actor_conv5(inputs[:, 4:5, :]))
        # x_5 = F.relu(self.actor_fc_2(inputs[:, 4:5, -1]))
        # x_6 = F.relu(self.actor_fc_3(inputs[:, 5:6, -1]))
        # x_7 = F.relu(self.actor_fc_4(inputs[:, 6:7, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        # x_6 = x_6.view(-1, self.num_flat_features(x_6))
        # x_7 = x_7.view(-1, self.num_flat_features(x_7))

        x = torch.cat([x_1, x_2, x_3, x_4, x_5], 1)
        x = F.relu(self.fc1(x))
        # critic
        # critic = F.relu(self.fc1(x))
        # critic = F.relu(self.fc2(critic))
        critic = self.fc3(x)
        return critic

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features


class MActor(torch.nn.Module):
    def __init__(self, action_space):
        super(MActor, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128 
        channel_fc = 128

        self.bn = nn.BatchNorm1d(self.input_channel)

        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4) # rate
        self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4) # bw
        # self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # download time
        self.actor_fc_1 = nn.Linear(self.input_channel, channel_fc) # latency
        self.actor_fc_2 = nn.Linear(self.input_channel, channel_fc) # freeze
        self.actor_fc_3 = nn.Linear(self.input_channel, channel_fc) # idle
        self.actor_fc_4 = nn.Linear(self.input_channel, channel_fc) # buffer
        self.actor_fc_5 = nn.Linear(self.input_channel, channel_fc) # suggestion

        #===================Hide layer=========================
        incoming_size = 2*channel_cnn*5 + 5 * channel_fc  #+ 1 * channel_cnn*3

        self.fc1 = nn.Linear(in_features=incoming_size, out_features= channel_fc)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=channel_fc, out_features=self.action_space)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):
        rates_batch = inputs[:, 0:1, :] ## refer to env_train.py
        rates_batch = self.bn(rates_batch)
        
        bandwitdh_batch = inputs[:, 1:2, :]
        bandwitdh_batch = self.bn(bandwitdh_batch)

        # download_time_batch = inputs[:, 2:3, :]
        # download_time_batch = self.bn(download_time_batch)

        x_1 = F.relu(self.actor_conv1(rates_batch))
        
        x_2 = F.relu(self.actor_conv2(bandwitdh_batch))
        # x_3 = F.relu(self.actor_conv3(download_time_batch))
        x_4 = F.relu(self.actor_fc_1(inputs[:, 3:4, -1]))
        x_5 = F.relu(self.actor_fc_2(inputs[:, 4:5, -1]))
        x_6 = F.relu(self.actor_fc_3(inputs[:, 5:6, -1]))
        x_7 = F.relu(self.actor_fc_4(inputs[:, 6:7, -1]))
        x_8 = F.relu(self.actor_fc_5(inputs[:, 7:8, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        # x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        x_6 = x_6.view(-1, self.num_flat_features(x_6))
        x_7 = x_7.view(-1, self.num_flat_features(x_7))
        x_8 = x_8.view(-1, self.num_flat_features(x_8))

        x = torch.cat([x_1, x_2, x_4, x_5, x_6, x_7, x_8], 1)
        x = F.relu(self.fc1(x))
        # actor
        # actor = F.relu(self.fc1(x))
        # actor = F.relu(self.fc2(actor))
        actor = F.softmax(self.fc3(x), dim=1)
        return actor

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

class MCritic(torch.nn.Module):
    def __init__(self, action_space):
        super(MCritic, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128 
        channel_fc = 128

        self.bn = nn.BatchNorm1d(self.input_channel)

        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4) # rate
        self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4) # bw
        # self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # download time
        self.actor_fc_1 = nn.Linear(self.input_channel, channel_fc) # latency
        self.actor_fc_2 = nn.Linear(self.input_channel, channel_fc) # freeze
        self.actor_fc_3 = nn.Linear(self.input_channel, channel_fc) # idle
        self.actor_fc_4 = nn.Linear(self.input_channel, channel_fc) # buffer len
        self.actor_fc_5 = nn.Linear(self.input_channel, channel_fc) # suggestion

        #===================Hide layer=========================
        incoming_size = 2*channel_cnn*5 + 5 * channel_fc  #+ 1 * channel_cnn*3

        self.fc1 = nn.Linear(in_features=incoming_size, out_features= channel_fc)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):
        rates_batch = inputs[:, 0:1, :] ## refer to env_train.py
        rates_batch = self.bn(rates_batch)
        
        bandwitdh_batch = inputs[:, 1:2, :]
        bandwitdh_batch = self.bn(bandwitdh_batch)

        # download_time_batch = inputs[:, 2:3, :]
        # download_time_batch = self.bn(download_time_batch)

        x_1 = F.relu(self.actor_conv1(rates_batch))
        x_2 = F.relu(self.actor_conv2(bandwitdh_batch))
        # x_3 = F.relu(self.actor_conv3(download_time_batch))
        x_4 = F.relu(self.actor_fc_1(inputs[:, 3:4, -1]))
        x_5 = F.relu(self.actor_fc_2(inputs[:, 4:5, -1]))
        x_6 = F.relu(self.actor_fc_3(inputs[:, 5:6, -1]))
        x_7 = F.relu(self.actor_fc_4(inputs[:, 6:7, -1]))
        x_8 = F.relu(self.actor_fc_5(inputs[:, 7:8, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        # x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        x_6 = x_6.view(-1, self.num_flat_features(x_6))
        x_7 = x_7.view(-1, self.num_flat_features(x_7))
        x_8 = x_8.view(-1, self.num_flat_features(x_8))

        x = torch.cat([x_1, x_2, x_4, x_5, x_6, x_7, x_8], 1)
        x = F.relu(self.fc1(x))
        # critic
        # critic = F.relu(self.fc1(x))
        # critic = F.relu(self.fc2(critic))
        critic = self.fc3(x)
        return critic

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features
