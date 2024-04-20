import random
import sys
import time
import threading
import torch
import torch.optim as optim
import numpy as np
from algorithms.client_info import client_info
from algorithms.components.hd3qn import hDQN, OptimizerSpec
sys.path.append("..")
from utils.config import Config
from utils.utils import Logger, get_allocation, get_allocation2
from algorithms.pesudo_server import pesudo_server

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

VIDEO_BIT_RATE = Config.BITRATE  # Kbps
MAX_RATE = float(np.max(VIDEO_BIT_RATE))

MODEL_PATH = 'models/hmarl/local/ddqn_test_0.model'

MAX_EP_LEN = 10
TRAIN_START_META = 2000
TRAIN_START_LOCAL = 5000
TRAIN_INTERVAL = 100
TRAIN_TIMES = 70
MODEL_SAVE_INTERVAL = 5000
META_MODEL_SAVE_INTERVAL = 2000
DECAY_INTERVAL = 10
# MAX_EP_LEN = 10
# TRAIN_START_META = 1
# TRAIN_START_LOCAL = 1
# TRAIN_INTERVAL = 15
# TRAIN_TIMES = 1
# MODEL_SAVE_INTERVAL = 1
# DECAY_INTERVAL = 1


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
        
        self.train_local_event = threading.Event()
        
        if not train_local:
            self.load_model_local(MODEL_PATH)
            
        self.esThread = threading.Thread(target=self.periodic_estimation)
        # self.esThread.start()
    
    def load_model_local(self, file_path):
        self.agent.load_controller_model(file_path)
        Logger.log("Local controller model loaded")
        
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
    
    def select_goal(self, meta_state, epsilon):
        goal = self.agent.select_goal(torch.from_numpy(meta_state).unsqueeze(0).type(dtype), epsilon)
        self.agent.meta_count += 1
        return VIDEO_BIT_RATE[goal.item()], goal
    
    def select_rate(self, state_goal, epsilon):
        action = self.agent.select_action(torch.from_numpy(state_goal).unsqueeze(0).type(dtype), epilson=epsilon)
        self.agent.local_count += 1
        return VIDEO_BIT_RATE[action.item()], action
    
    def estimate_total_bw(self):
        esUpper, esLower = 0, 0
        for _, c in self.client_list.items():
            esUpper += c.get_smooth_bw()
            esLower += c.get_smooth_bw_idle()
        esTotalBW = Config.UPPER_PORTION * esUpper + (1 - Config.UPPER_PORTION) * esLower
        return esTotalBW
    
    def server_goal_estimation(self, client:client_info):
        # a = np.array([x.get_smooth_bw_idle() for _, x in self.client_list.items()])
        esUpper, esLower = 0, 0
        bottlenecks = {}
        weights = {}
        last_rates = {}
        for idx, c in self.client_list.items():
            esUpper += c.get_smooth_bw()
            esLower += c.get_smooth_bw_idle()
            bottlenecks[idx] = c.get_bottleneck()[0]
            weights[idx] = c.weight
            last_rates[idx] = c.rate
        esTotalBW = Config.UPPER_PORTION * esUpper + (1 - Config.UPPER_PORTION) * esLower
        print(f"ESTotalBW: {esTotalBW:.3f}, {esUpper}, {esLower}")
        # return get_allocation(bottlenecks=bottlenecks, weights=weights, totalBw=esTotalBW, client_idx=client.client_idx)
        return get_allocation2(bottlenecks=bottlenecks, weights=weights, last_rates=last_rates, totalBw=esTotalBW, client_idx=client.client_idx)
        # target_bw = 0
        # sum_weights = self.sum_weights
        # bottleneck = 0
        # fair_bw = 0
        # for _, c in self.client_list.items():
        #     bottleneck, std = c.get_bottleneck()
        #     weight = c.weight
        #     fair_bw =  (weight / sum_weights) * esTotalBW
        #     target_bw = min(fair_bw, bottleneck)
        #     if c.client_idx == client.client_idx:
        #         break
        #     sum_weights -= weight
        #     esTotalBW -= target_bw
        # goal = VIDEO_BIT_RATE[0]
        # for i in reversed(range(len(VIDEO_BIT_RATE))):
        #     if VIDEO_BIT_RATE[i] < target_bw:
        #         goal = VIDEO_BIT_RATE[i]
        #         break
        # return goal
    
    def train_meta_controller(self):
        if self.train_meta and self.agent.meta_count >= TRAIN_START_META and self.agent.meta_count % TRAIN_INTERVAL == 0:
            Logger.log("Training meta controller...")
            for _ in range(TRAIN_TIMES):
                self.agent.update_meta_controller()
            Logger.log("Meta controller training completed")
            
    def train_local_controller(self):
        Logger.log("Training local controller...")
        t1 = time.time()
        for _ in range(TRAIN_TIMES):
            self.agent.update_controller()
        t2 = time.time()
        print(f"Pass: {t2-t1}")
        Logger.log("Local controller training completed")
    
    def solve(self, idx):
        client = self.client_list[idx]
        done = False
        # print(client.client_idx, client.controller_epsilon, client.meta_controller_epsilon)
        # steps_taken = client.episode_step if client.episode_step != -1 else 0
        # Goal reached, max steps reached or client started
        # if steps_taken == MAX_EP_LEN or client.hmarl_step == -1:  
        #     # Select goals
        #     # Initialize steps
        #     self.update_meta_lock.acquire()
        #     done = True
        #     if client.hmarl_step == -1:
        #         done = False
        #         client.hmarl_step = 0
        #     # Get meta state
        #     meta_state = self.get_meta_state(client, steps_taken)
        #     # Push data to meta controller
        #     if self.train_meta and client.hmarl_step > 0:
        #         F = client.accumulative_extrinsic_reawad / steps_taken
        #         print(f"Client{client.client_idx} gets Reward F: {F}")
        #         self.agent.meta_replay_memory.push(client.last_meta_state, client.goal_idx, meta_state, F, False)
        #     client.last_meta_state = meta_state.copy()
            
        #     client.episode_step = 0
        #     client.accumulative_extrinsic_reawad = 0
            
            
        #     meta_epsilon = client.meta_controller_epsilon
        #     # client.goal, client.goal_idx = self.select_goal(meta_state, meta_epsilon)
        #     # t4 = time.time()
        #     client.goal = self.server_goal_estimation(client)
        #     # t5 = time.time()
        #     # print("!!!!!!!!!", t5 - t4)
        #     Logger.log(f"Client {client.client_idx} gets goal {client.goal} with epsilon {client.meta_controller_epsilon}")
            
        #     # Train meta controller
        #     self.train_meta_controller()
        #     if self.train_meta and self.agent.meta_count >= TRAIN_START_META and self.agent.meta_count % META_MODEL_SAVE_INTERVAL == 0:
        #         Logger.log("Meta controller model saved")
        #         self.agent.save_meta_controller_model(self.agent.meta_count)
        #     self.update_meta_lock.release()
        if client.hmarl_step == -1:
            client.hmarl_step += 1
        if self.train_local and self.agent.local_count >= TRAIN_START_LOCAL and self.agent.local_count % DECAY_INTERVAL == 0:
            client.epsilon_decay()
        client.goal = self.server_goal_estimation(client)
        # client.goal = self.assigned_rate[client.client_idx]
            # Logger.log("Controller model loaded")
            # self.agent.load_controller_model(client.hmarl_step)
            # self.agent.load_meta_controller_model(client.hmarl_step)
            
        # Get extrinsic and intrinsic rewards
        intrinsic_reward = client.get_intrinsic_reward()
        client.last_rate = client.rate
        extrinsic_reward = self.get_extrinsic_reward(client)
        client.accumulative_extrinsic_reawad += extrinsic_reward
        
        client.episode_step += 1
        client.hmarl_step += 1
        state_goal = client.get_state_goal()
        # print(state_goal)
        
        
        self.update_local_lock.acquire()
        if self.train_local:
            # Push data to local controller
            if client.hmarl_step > 1:
                self.agent.ctrl_replay_memory.push(client.last_state, client.rate_idx, client.get_state_goal(), intrinsic_reward, False)
            # Train local controller
            if self.agent.local_count >= TRAIN_START_LOCAL and self.agent.local_count % TRAIN_INTERVAL == 0:
                threading.Thread(target=self.train_local_controller).start()
            # self.train_local_controller()
            
            # Save controller model
            if self.agent.local_count >= TRAIN_START_LOCAL and self.agent.local_count % MODEL_SAVE_INTERVAL == 0:
                Logger.log("Local controller model saved")
                self.agent.save_controller_model(self.agent.local_count)
        # Select new rate
        epsilon = client.controller_epsilon if self.train_local else 0
        
        client.rate, client.rate_idx = self.select_rate(state_goal, epsilon)
        # client.rate = client.goal
        # client.rate = 4.0
        self.update_local_lock.release()
        Logger.log(f"Client {client.client_idx} gets goal {client.goal} and rate {client.rate} with epsilon {epsilon}") 
        
        return client.rate, client.goal, intrinsic_reward, extrinsic_reward, self.estimate_total_bw()
        
    
    
