import random
import sys
import threading
import torch
import torch.optim as optim
import numpy as np
from algorithms.client_info import client_info
from algorithms.components.hd3qn import hDQN, OptimizerSpec
sys.path.append("..")
from utils.config import Config
from utils.utils import Logger
from algorithms.pesudo_server import pesudo_server

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

VIDEO_BIT_RATE = Config.BITRATE  # Kbps
MAX_RATE = float(np.max(VIDEO_BIT_RATE))

MAX_EP_LEN = 10
TRAIN_START_META = 2000
TRAIN_START_LOCAL = 3000
TRAIN_INTERVAL = 50
TRAIN_TIMES = 50
MODEL_SAVE_INTERVAL = 5000
META_MODEL_SAVE_INTERVAL = 2000
# MAX_EP_LEN = 10
# TRAIN_START_META = 1
# TRAIN_START_LOCAL = 1
# TRAIN_INTERVAL = 5
# TRAIN_TIMES = 5
# MODEL_SAVE_INTERVAL = 1


class hmarl_server(pesudo_server):
    
    def __init__(self, train_local=True, train_meta=True):
        super().__init__()
        
        optimizer_spec = OptimizerSpec(
            constructor=optim.RMSprop,
            kwargs=dict(lr=0.00025, alpha=0.95, eps=0.01),
        )

        self.agent = hDQN(optimizer_spec)
        self.train_local = train_local
        self.train_meta = train_meta
        
        self.local_data = 0
        self.meta_data = 0
        
        
    def get_others_sum_mean_rate(self, client:client_info, steps_taken):
        sum = 0
        for _, x in self.client_list.items():
            if x.client_idx != client.client_idx:
                sum += x.get_mean_rate_pivot(client.startTime_his.get(-steps_taken))
        return sum
    
    def get_meta_state(self, client:client_info, steps_taken=1):
        meta_state = np.zeros(4)
        mean_total_bw = 21.5
        # calculate meta state members
        mean_rate = np.mean(client.rate_his.get_last_mean(steps_taken)) / mean_total_bw
        sum_others_rate = self.get_others_sum_mean_rate(client, steps_taken) / mean_total_bw
        last_goal = client.goal / MAX_RATE
        weight = client.weight / self.sum_weights
        
        meta_state[0] = mean_rate
        meta_state[1] = sum_others_rate
        meta_state[2] = last_goal
        meta_state[3] = weight
        return meta_state
    
    def get_extrinsic_reward(self, client:client_info):
        return self.get_propotional_fairness()
    
    def get_propotional_fairness(self):
        fairness = 0
        # li = []
        for _, client in self.client_list.items():
            qoe = client.get_qoe()
            weight = client.weight
            fairness += weight * np.log(max(qoe, 1))
            # li.append(qoe)
        # print(li)
        # print(fairness)
        return fairness
    
    def get_maxmin_fairness(self):
        fairness = 10000
        for _, client in self.client_list.items():
            qoe = client.get_qoe()
            weight = client.weight
            fairness = min(qoe / weight, fairness)
        return fairness
    
    def select_goal(self, meta_state, epsilon):
        goal = self.agent.select_goal(torch.from_numpy(meta_state).unsqueeze(0).type(dtype), epsilon)
        self.agent.meta_count += 1
        return VIDEO_BIT_RATE[goal.item()], goal
    
    def select_rate(self, state_goal, epsilon):
        action = self.agent.select_action(torch.from_numpy(state_goal).unsqueeze(0).type(dtype), epsilon)
        self.agent.local_count += 1
        return VIDEO_BIT_RATE[action.item()], action
    
    def server_goal_estimation(self, client:client_info):
        a = np.array([x.get_smooth_bw_idle() for _, x in self.client_list.items()])
        b = np.array([x.get_smooth_bw() for _, x in self.client_list.items()])
        esTotalBW = b.sum()
        
        target_bw = 0
        sum_weights = self.sum_weights
        bottleneck = 0
        fair_bw = 0
        print("++++++++++++++")
        print(b, esTotalBW)
        
        for _, c in self.client_list.items():
            bottleneck, std = c.get_bottleneck()
            weight = c.weight
            fair_bw =  (weight / sum_weights) * esTotalBW
            target_bw = min(fair_bw, bottleneck)
            if c.client_idx == client.client_idx:
                break
            sum_weights -= weight
            esTotalBW -= fair_bw

        print(bottleneck, std + bottleneck, fair_bw)
        print("++++++++++++++")
        goal = VIDEO_BIT_RATE[0]
        for i in reversed(range(len(VIDEO_BIT_RATE))):
            if VIDEO_BIT_RATE[i] < target_bw:
                goal = VIDEO_BIT_RATE[i]
                break
        
        return goal
    
    def pesudo_select_goal(self, client:client_info):
        sample = random.random()
        if client.client_idx == 0:
            if sample < 0.1:
                client.goal = 4.0
            elif sample < 0.2:
                client.goal = 6.5
            elif sample < 0.25:
                client.goal = 5.0
            else:
                client.goal = 8.0
        elif client.client_idx == 1:
            if sample < 0.1:
                client.goal = 6.5
            elif sample < 0.2:
                client.goal = 10.0
            else:
                client.goal = 8.0
        elif client.client_idx == 2:
            if sample < 0.1:
                client.goal = 2.5
            elif sample < 0.2:
                client.goal = 5.0
            else:
                client.goal = 4.0
        return client.goal
    
    def solve(self, idx):
        client = self.client_list[idx]
        done = False
        end = False
        # print(client.client_idx, client.controller_epsilon, client.meta_controller_epsilon)
        steps_taken = client.episode_step if client.episode_step != -1 else 0
        # Goal reached, max steps reached or client started
        if client.goal_reached() or steps_taken == MAX_EP_LEN or client.hmarl_step == -1:  
            # Select goals
            # Initialize steps
            self.update_meta_lock.acquire()
            done = True
            if not client.goal_reached() and steps_taken == MAX_EP_LEN:
                end = True
            if client.hmarl_step == -1:
                done = False
                client.hmarl_step = 0
            # Get meta state
            meta_state = self.get_meta_state(client, steps_taken)
            # Push data to meta controller
            if self.train_meta and client.hmarl_step > 0:
                F = client.accumulative_extrinsic_reawad / steps_taken
                print(f"Client{client.client_idx} gets Reward F: {F}")
                self.agent.meta_replay_memory.push(client.last_meta_state, client.goal_idx, meta_state, F, False)
            client.last_meta_state = meta_state.copy()
            
            client.episode_step = 0
            client.accumulative_extrinsic_reawad = 0
            
            
            meta_epsilon = client.meta_controller_epsilon
            client.goal, client.goal_idx = self.select_goal(meta_state, meta_epsilon)
            # select pesudo goal
            client.goal = self.server_goal_estimation(client)
            Logger.log(f"Client {client.client_idx} gets goal {client.goal} with epsilon {client.meta_controller_epsilon}")
            
            # Train meta controller
            if self.train_meta and self.agent.meta_count >= TRAIN_START_META and self.agent.meta_count % TRAIN_INTERVAL == 0:
                Logger.log("Training meta controller...")
                for t in range(TRAIN_TIMES):
                    self.agent.update_meta_controller()
                Logger.log("Meta controller training completed")
            if self.train_meta and self.agent.meta_count >= TRAIN_START_META and self.agent.meta_count % META_MODEL_SAVE_INTERVAL == 0:
                Logger.log("Meta controller model saved")
                self.agent.save_meta_controller_model(self.agent.meta_count)
            self.update_meta_lock.release()
            client.epsilon_decay()
            
            # Logger.log("Controller model loaded")
            # self.agent.load_controller_model(client.hmarl_step)
            # self.agent.load_meta_controller_model(client.hmarl_step)
            
        # Get extrinsic and intrinsic rewards
        intrinsic_reward = client.get_intrinsic_reward(end, steps_taken)
        client.last_rate = client.rate
        extrinsic_reward = self.get_extrinsic_reward(client)
        client.accumulative_extrinsic_reawad += extrinsic_reward
        
        client.episode_step += 1
        client.hmarl_step += 1
        state_goal = client.get_state_goal()
        
        self.update_local_lock.acquire()
        if self.train_local:
            # Push data to local controller
            if client.hmarl_step > 1:
                self.agent.ctrl_replay_memory.push(client.last_state, client.rate_idx, client.get_state_goal(), intrinsic_reward, done)
            # Train local controller
            if self.agent.local_count >= TRAIN_START_LOCAL and self.agent.local_count % TRAIN_INTERVAL == 0:
                Logger.log("Training local controller...")
                for t in range(TRAIN_TIMES):
                    self.agent.update_controller()
                Logger.log("Local controller training completed")
            # Save controller model
            if self.agent.local_count >= TRAIN_START_LOCAL and self.agent.local_count % MODEL_SAVE_INTERVAL == 0:
                Logger.log("Local controller model saved")
                self.agent.save_controller_model(self.agent.local_count)
        # Select new rate
        epsilon = client.controller_epsilon
        client.rate, client.rate_idx = self.select_rate(state_goal, epsilon)
        self.update_local_lock.release()
        Logger.log(f"Client {client.client_idx} gets rate {client.rate} with epsilon {client.controller_epsilon}") 
        
        return client.rate, client.goal, intrinsic_reward, extrinsic_reward
        
    
    
