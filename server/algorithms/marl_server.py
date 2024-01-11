import sys
import numpy as np
from algorithms.client_info import client_info
sys.path.append("..")
from utils.config import Config

class marl_server:
    
    def __init__(self):
        self.client_list = {}
        self.num_agent = 0
        
    def add_client(self, idx, rate=0, buffer=0, latency=0):
        info = client_info(idx)
        self.client_list[idx] = info
        self.num_agent += 1
    
    def remove_client(self, idx):
        self.client_list.pop(idx)
        self.num_agent -= 1
        
    def update_info(self, idx, rate, bw, buffer, latency):
        self.client_list[idx].update(rate, bw, buffer, latency)
        
    def orchestrate(self, idx):
        esTotalBW = np.array([x.bw for _, x in self.client_list.items()]).sum()
        client_bw = self.client_list[idx].bw
        fair_bw = esTotalBW / self.num_agent
        if abs(client_bw - fair_bw) < 1:
            instruction = 0
        elif client_bw - fair_bw >= 1:
            instruction = -1
        else:
            instruction = 1
        
        return instruction, 0
    
