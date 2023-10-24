import numpy as np
import math 
import sys
sys.path.append("..")
from utils.utils import Logger
from utils.config import Config

# SEG_DURATION = 1.0
# ENCODING_TIME = 1.0
# CHUNK_DURATION = 0.2
# CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
# BITRATE = [0.3, 0.5, 1.0, 2.0, 3.0, 6.0]
# SPEED = [0.9, 1.0, 1.1]
# MAX_RATE = BITRATE[-1]

class stallion_solver(object):
    def __init__(self, initial_latency):
        # For new traces
        self.bw_f = 1.0
        self.latency_f = 1.25
        self.n_step = 10
        self.target_latency = 4
        self.speed_buffer_tth = 0.6
        self.bw_history = []
        self.latency_history = []
        self.initial_latency = initial_latency
        self.seg_duration = Config.SEG_DURATION
        self.chunk_duration = Config.CHUNK_DURATION

    def reset(self):
        self.bw_history = []
        self.latency_history = []

    def update_bw_latency(self, bw, latency):
        self.bw_history += [bw]
        self.latency_history += [latency]
        if len(self.bw_history) > self.n_step:
            self.bw_history.pop(0)
        if len(self.latency_history) > self.n_step:
            self.latency_history.pop(0)

    def choose_rate(self, bw):
        i = 0
        for i in reversed(range(len(Config.BITRATE))):
            if Config.BITRATE[i] < bw:
                return i
        return i

    def solve(self, buffer_length, curr_latency):
        # First of all, get speed
        rate, speed = None, None
        if curr_latency >= self.target_latency and buffer_length >= self.speed_buffer_tth:
            speed = 2
        else:
            speed = 1

        # Get rate
        mean_bw, mean_latency = np.mean(self.bw_history), np.mean(self.latency_history)
        std_bw, std_latency = np.std(self.bw_history), np.std(self.latency_history)
        predict_bw = mean_bw - self.bw_f*std_bw
        predict_latency = mean_latency + self.latency_f*std_latency
        overhead = max(predict_latency - self.target_latency, 0)
        if overhead >= self.initial_latency + self.seg_duration + Config.ENCODING_TIME:
            rate = 0
        else:
            dead_time = self.seg_duration - overhead
            ratio = dead_time/self.seg_duration
            predict_bw *= ratio
            rate = self.choose_rate(predict_bw)
            # print(predict_bw, predict_latency, self.latency_history[-1], overhead, ratio)
            
        # print(rate,speed)
        return Config.BITRATE[rate], Config.SPEED[speed]