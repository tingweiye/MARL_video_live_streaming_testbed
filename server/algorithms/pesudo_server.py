from algorithms.client_info import client_info
from utils.config import Config
import threading
import numpy as np

class pesudo_server:
    
    def __init__(self):
        
        self.client_list = {}
        self.assigned_rate = {}
        self.num_agent = 0
        self.sum_weights = 0
        
        self.update_meta_lock = threading.Lock()
        self.update_local_lock = threading.Lock()
        
    def add_client(self, idx, weight=1, rate=0, buffer=0, latency=0):
        self.update_local_lock.acquire()
        info = client_info(idx, weight)
        self.client_list[idx] = info
        self.num_agent += 1
        self.sum_weights += weight
        print(self.sum_weights)
        self.update_local_lock.release()
    
    def remove_client(self, idx):
        self.update_local_lock.acquire()
        self.sum_weights -= self.client_list[idx].weight
        self.client_list.pop(idx)
        self.num_agent -= 1
        self.update_local_lock.release()
        
    def update_info(self, idx, info):
        self.client_list[idx].update(info)
        
    def bw_prediction(self, startTime):
        low_all, high_all = 0, 0
        for _, c in self.client_list.items():
            low, high = c.get_traffic_low_high(startTime)
            low_all += low
            high_all += high
        return (low_all + high_all) / Config.SERVER_ESTIMATION_LEN / 2
    
    def get_propotional_fairness(self):
        fairness = 0
        for _, client in self.client_list.items():
            qoe = client.get_qoe()
            weight = client.weight
            fairness += weight * np.log(max(qoe, 0.1))
        return fairness
    
    def get_maxmin_fairness(self):
        fairness = 10000
        for _, client in self.client_list.items():
            qoe = client.get_qoe()
            weight = client.weight
            fairness = min(qoe / weight, fairness)
        return fairness
    
    def get_client_qoe(self, idx):
        return self.client_list[idx].get_qoe()