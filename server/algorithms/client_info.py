import sys
import numpy as np
sys.path.append("..")
from utils.config import Config
from utils.utils import MovingQueue

S_INFO = 5
S_META = 4
S_LEN = 10
VIDEO_BIT_RATE = Config.BITRATE  # Kbps
MAX_RATE = float(np.max(VIDEO_BIT_RATE))
NUM_ACTIONS = len(VIDEO_BIT_RATE)
INITIAL_RATE = VIDEO_BIT_RATE[int(len(VIDEO_BIT_RATE) // 2)]

MAX_EPSILON_C = 0.9
MAX_EPSILON_M = 0.9
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.9995

QUALTITY_COEF = 5
FREEZE_PENALTY = 50
LATENCY_PENALTY = 1
JUMP_PENALTY = 3
SMOOTH_PENALTY = 8

QOE_QUALTITY_COEF = 2
QOE_SMOOTH_PENALTY = 2
QOE_FREEZE_PENALTY = 12
QOE_LATENCY_PENALTY = 0.2

class client_info:
    
    def __init__(self, idx, weight=1):
        self.client_idx = idx
        self.weight = weight
        self.rate, self.last_rate = INITIAL_RATE, INITIAL_RATE
        self.rate_idx = VIDEO_BIT_RATE.index(INITIAL_RATE)
        self.bw, self.idle, self.bw_idle, self.buffer, self.freeze, self.latency, self.jump, self.startTime, self.goal = 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.rate_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.bw_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.bottleneck = MovingQueue(50) 
        self.bw_idle_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.idle_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.buffer_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.freeze_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN)
        self.latency_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.jump_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.startTime_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        
        # Hierarchical RL parameters
        self.goal = MAX_RATE
        self.goal_idx = len(VIDEO_BIT_RATE) - 1
        self.hmarl_step = -1
        self.episode_step = -1
        self.accumulative_extrinsic_reawad = 0
        self.controller_epsilon = MAX_EPSILON_C
        self.meta_controller_epsilon = MAX_EPSILON_M
        
        self.state = np.zeros((S_INFO,S_LEN))
        self.last_state = np.zeros((S_INFO,S_LEN))
        self.last_meta_state = np.zeros((1, S_META))
        
        self.reach_reward = {2.5:50,
                             4.0:45,
                             5.0:40,
                             6.5:35,
                             8.0:35,
                             10.0:30}
        
    def getLen(self):
        return len(self.rate_his)
        
    def get_smooth_bw(self):
        # return self.bw_his.avg()
        return self.bw_his.get_last_mean(6)
    
    def get_smooth_bw_idle(self):
        return self.bw_idle_his.avg()
    
    def get_bottleneck(self):
        return self.bottleneck.max(), self.bottleneck.std()
    
    def get_traffic_low_high(self, pivot):
        low, high = 0, 0
        for i in range(len(self.rate_his)-1,-1,-1):
            if self.startTime_his.get(i) >= pivot - Config.SERVER_ESTIMATION_LEN and \
                self.startTime_his.get(i) < pivot:
                low += self.rate_his.get(i)
                high = low
            elif self.startTime_his.get(i) < pivot - Config.SERVER_ESTIMATION_LEN:
                high = low + self.rate_his.get(i)
                break
        return low, high
    
    def get_mean_rate_pivot(self, pivot):
        if len(self.startTime_his) == 0:
            return 0
        sum, n = 0, 0
        for i in range(len(self.rate_his)-1,-1,-1):
            if self.startTime_his.get(i) >= pivot:
                sum += self.rate_his.get(i)
                n += 1
            else:
                sum += self.rate_his.get(i)
                n += 1
                break
        return sum / n
    
    def update(self, info):
        if info["rate"] != 0:
            self.rate = info["rate"]
        self.bw = info["bw"]
        self.idle = info["idle"]
        self.bw_idle = self.rate / (self.rate / self.bw + self.idle)
        self.buffer = info["buffer"]
        self.freeze = info["freeze"]
        self.latency = info["latency"]
        self.jump = info["jump"]
        self.startTime = info["startTime"]
        
        self.rate_his.add(self.rate)
        self.bw_his.add(self.bw)
        self.bottleneck.add(self.bw)
        self.idle_his.add(self.idle)
        self.bw_idle_his.add(self.bw_idle)
        self.buffer_his.add(self.buffer)
        self.freeze_his.add(self.freeze)
        self.latency_his.add(self.latency)
        self.jump_his.add(self.jump)
        self.startTime_his.add(self.startTime)
        
        self.last_state = self.get_state_goal().copy()
        self.state = np.roll(self.state, -1, axis=1)
        self.state[0, -1] = self.rate / MAX_RATE
        self.state[1, -1] = self.get_smooth_bw() / 10
        self.state[2, -1] = (self.buffer-1) / Config.CLIENT_MAX_BUFFER_LEN
        self.state[3, -1] = self.idle

    def epsilon_decay(self):
        self.controller_epsilon *= EPSILON_DECAY
        self.meta_controller_epsilon *= EPSILON_DECAY
        
    def goal_reached(self):
        return self.goal == self.rate
    
    def get_state_goal(self):
        self.state[4, -1] = self.goal / MAX_RATE
        return self.state
    
    def get_qoe(self):
        log_rate = np.log(self.rate)
        log_last_rate = np.log(self.last_rate)
        QoE =     QOE_QUALTITY_COEF  * log_rate \
                - QOE_FREEZE_PENALTY * self.freeze \
                - QOE_LATENCY_PENALTY* self.latency \
                - QOE_SMOOTH_PENALTY * np.abs(log_rate - log_last_rate)
        return QoE
    
    def get_intrinsic_reward(self, done, steps_taken, reward_file=""):
        # -- log scale reward --
        log_rate = np.log(self.rate)
        log_goal = np.log(self.goal)
        log_last_rate = np.log(self.last_rate)
        
        reward  = QUALTITY_COEF  * log_rate \
                - (FREEZE_PENALTY * max(0.75, self.freeze) if self.freeze > 0.001 else 0) \
                - LATENCY_PENALTY* self.latency \
                - JUMP_PENALTY   * self.jump \
                - SMOOTH_PENALTY * np.abs(log_rate - log_last_rate)
                
        if self.goal_reached():
            reward += self.reach_reward[self.goal]
        elif self.goal < self.rate:
            reward -= QUALTITY_COEF * (log_rate - log_goal)
        else:
            reward += QUALTITY_COEF * log_rate
                                       
        if steps_taken >= 5: # If not following goal and causes freeze, give pentalty
            reward -= QOE_FREEZE_PENALTY * max(0.75, self.freeze) if self.freeze > 0.001 else 0
        
        return reward
    
    