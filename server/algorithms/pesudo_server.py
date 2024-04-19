from algorithms.client_info import client_info
from utils.config import Config
import threading
import numpy as np
import time

class pesudo_server:
    
    def __init__(self):
        
        self.client_list = {}
        self.assigned_rate = {}
        self.num_agent = 0
        self.sum_weights = 0
        self.esTotalBw = 5
        
        self.update_meta_lock = threading.Lock()
        self.update_local_lock = threading.Lock()
        
    def add_client(self, idx, weight=1, rate=0, buffer=0, latency=0):
        self.update_local_lock.acquire()
        info = client_info(idx, weight)
        self.client_list[idx] = info
        self.assigned_rate[idx] = Config.MID_RATE
        self.num_agent += 1
        self.sum_weights += weight
        print(self.sum_weights)
        self.update_local_lock.release()
    
    def remove_client(self, idx):
        self.update_local_lock.acquire()
        self.sum_weights -= self.client_list[idx].weight
        self.client_list.pop(idx)
        self.assigned_rate.pop(idx)
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
    
    def periodic_estimation(self):
        while(True):
            t1 = time.time()
            self.server_allocation_estimation()
            t2 = time.time()
            # print("????????", t2 - t1)
            time.sleep(0.95)
    
    def server_allocation_estimation(self):
        esUpper, esLower = 0, 0
        bottlenecks = {}
        weights = {}
        # last_rates = []
        for idx, c in self.client_list.items():
            esUpper += c.get_smooth_bw()
            esLower += c.get_smooth_bw_idle()
            bottlenecks[idx] = c.get_bottleneck()[0]
            weights[idx] = c.weight
            # last_rates.append(c.rate)
        self.esTotalBW = Config.UPPER_PORTION * esUpper + (1 - Config.UPPER_PORTION) * esLower
        print(f"ESTotalBW: {self.esTotalBW:.3f}, {esUpper}, {esLower}")
        self.assigned_rate = self.get_allocation_all(bottlenecks=bottlenecks, weights=weights, totalBw=self.esTotalBW)

    def get_allocation_all(self, bottlenecks, weights, totalBw):
        buffer = {key:2.5 for key, _ in weights.items()}
        result = {key:2.5 for key, _ in weights.items()}
        clients = [key for key, _ in weights.items()]
        maxScore = 0

        def backtrace(bottlenecks, weights, totalBw, client, score):
            if client == len(weights):
                nonlocal maxScore
                nonlocal result
                if score > maxScore:
                    maxScore = score
                    print(score, buffer)
                    result = buffer.copy()
                return
            client_idx = clients[client]
            for r in Config.REVERSED_BITRATE:
                if r <= bottlenecks[client_idx] and r <= totalBw:
                    buffer[client_idx] = r
                    fair_contribution = weights[client_idx] * np.log(r)
                    backtrace(bottlenecks, weights, totalBw-r, client+1, score + fair_contribution)
                else:
                    continue
        backtrace(bottlenecks, weights, totalBw, 0, 0)
        return result