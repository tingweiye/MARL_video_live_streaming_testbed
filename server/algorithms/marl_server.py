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
        self.client_list.pop(idx)
        self.num_agent -= 1
        
    def update_info(self, idx, rate, bw, buffer, latency):
        self.client_list[idx].update(rate, bw, buffer, latency)
        
    def orchestrate(self, idx):
        esTotalBW = np.array([x.bw for _, x in self.client_list.items()]).sum()
        client_bw = self.client_list[idx].bw
        client_weight = self.client_list[idx].weight
        fair_bw =  (client_weight / self.sum_weights) * esTotalBW
        faircoe = max(client_bw - fair_bw, 0) / esTotalBW
        
        instruction = zfun(client_bw - fair_bw, 0.5, 5)
        if abs(client_bw - fair_bw) < 0.5:
            instruction = 0
        elif client_bw - fair_bw >= 0.5:
            instruction = client_bw - fair_bw - 0.5
        else:
            instruction = client_bw - fair_bw + 0.5
            
        exCoef = 1 - faircoe
        # print(instruction, client_bw, fair_bw, esTotalBW)
        
        return instruction, fair_bw, exCoef
    
