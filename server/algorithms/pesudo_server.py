from algorithms.client_info import client_info
from utils.config import Config
import threading

class pesudo_server:
    
    def __init__(self):
        
        self.client_list = {}
        self.num_agent = 0
        self.sum_weights = 0
        
        self.meta_lock = threading.Lock()
        
    def add_client(self, idx, weight=1, rate=0, buffer=0, latency=0):
        self.meta_lock.acquire()
        info = client_info(idx, weight)
        self.client_list[idx] = info
        self.num_agent += 1
        self.sum_weights += weight
        print(self.sum_weights)
        self.meta_lock.release()
    
    def remove_client(self, idx):
        self.meta_lock.acquire()
        self.sum_weights -= self.client_list[idx].weight
        self.client_list.pop(idx)
        self.num_agent -= 1
        self.meta_lock.release()
        
    def update_info(self, idx, info):
        self.client_list[idx].update(info)
        
    def bw_prediction(self, startTime):
        low_all, high_all = 0, 0
        for _, c in self.client_list.items():
            low, high = c.get_traffic_low_high(startTime)
            low_all += low
            high_all += high
        return (low_all + high_all) / Config.SERVER_ESTIMATION_LEN / 2