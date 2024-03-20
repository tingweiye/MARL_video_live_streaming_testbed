import sys
import numpy as np
from algorithms.client_info import client_info
sys.path.append("..")
from utils.config import Config
from utils.utils import zfun
from algorithms.pesudo_server import pesudo_server

class marl_server(pesudo_server):
    
    def __init__(self):
        super().__init__()
        
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
    
