import sys
import numpy as np
sys.path.append("..")
from utils.config import Config

class client_info:
    
    def __init__(self, idx, weight=1):
        self.client_idx = idx
        self.weight = weight
        self.rate, self.bw, self.buffer, self.latency = 0, 0, 0, 0
        self.rate_his, self.bw_his, self.buffer_his, self.latency_his = [], [], [], []
        self.moving_bw = 0
        
    def getLen(self):
        return len(self.rate_his)
        
    def update(self, rate, bw, buffer, latency):
        self.rate = rate
        self.bw = bw
        self.buffer = buffer
        self.latency = latency
        self.moving_bw += bw
        
        self.rate_his.append(rate)
        self.bw_his.append(bw)
        self.buffer_his.append(buffer)
        self.latency_his.append(latency)
        
        if len(self.rate_his) > Config.SERVER_ALGO_BUFFER_LEN:
            self.moving_bw -= self.bw_his[0]
            self.rate_his.pop(0)
            self.bw_his.pop(0)
            self.buffer_his.pop(0)
            self.latency_his.pop(0)
            
    def get_smooth_bw(self):
        if len(self.rate_his) < Config.SERVER_ALGO_BUFFER_LEN:
            return self.moving_bw / len(self.rate_his)
        else:
            return self.moving_bw / Config.SERVER_ALGO_BUFFER_LEN