import sys
import numpy as np
sys.path.append("..")
from utils.config import Config
from utils.utils import MovingQueue

class client_info:
    
    def __init__(self, idx, weight=1):
        self.client_idx = idx
        self.weight = weight
        self.rate, self.bw, self.buffer, self.latency, self.startTime = 0, 0, 0, 0, 0
        self.rate_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.bw_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.buffer_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.latency_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.startTime_his = MovingQueue(Config.SERVER_ALGO_BUFFER_LEN) 
        self.moving_bw = 0
        
    def getLen(self):
        return len(self.rate_his)
        
    def update(self, rate, bw, buffer, latency, startTime):
        self.rate = rate
        self.bw = bw
        self.buffer = buffer
        self.latency = latency
        self.startTime = startTime
        self.moving_bw += bw
        
        self.rate_his.add(rate)
        self.bw_his.add(bw)
        self.buffer_his.add(buffer)
        self.latency_his.add(latency)
        self.startTime_his.add(startTime)

    def get_smooth_bw(self):
        return self.bw_his.avg()
    
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