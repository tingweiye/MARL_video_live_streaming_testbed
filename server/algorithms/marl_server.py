import sys
import numpy as np
from algorithms.client_info import client_info
sys.path.append("..")
from utils.config import Config
from utils.utils import zfun

class marl_server:
    
    def __init__(self):
        self.client_list = {}
        self.num_agent = 0
        self.sum_weights = 0
        
    def add_client(self, idx, weight=1, rate=0, buffer=0, latency=0):
        info = client_info(idx, weight)
        self.client_list[idx] = info
        self.num_agent += 1
        self.sum_weights += weight
        print(self.sum_weights)
    
    def remove_client(self, idx):
        self.sum_weights -= self.client_list[idx].weight
        self.client_list.pop(idx)
        self.num_agent -= 1
        
    def update_info(self, idx, rate, bw, buffer, latency, startTime):
        self.client_list[idx].update(rate, bw, buffer, latency, startTime)
        
    def bw_prediction(self, startTime):
        low_all, high_all = 0, 0
        for _, c in self.client_list.items():
            low, high = c.get_traffic_low_high(startTime)
            low_all += low
            high_all += high
        return (low_all + high_all) / Config.SERVER_ESTIMATION_LEN / 2
        
    def orchestrate(self, idx):
        a = np.array([x.get_smooth_bw() for _, x in self.client_list.items()])
        esTotalBW = a.sum()
        print(a)
        client_bw = self.client_list[idx].get_smooth_bw()
        client_rate = self.client_list[idx].rate
        client_weight = self.client_list[idx].weight
        fair_bw =  (client_weight / self.sum_weights) * 14.5
        faircoe = abs(client_bw - fair_bw) / esTotalBW
        
        instruction = zfun(client_rate - fair_bw, 0, 6)
            
        exCoef = 1 - faircoe
        # print(instruction, client_bw, fair_bw, esTotalBW)
        
        return instruction, fair_bw, exCoef
    
