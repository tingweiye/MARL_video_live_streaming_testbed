from algorithms.marl_server import marl_server
import time
import os
import sys
sys.path.append("..")
from utils.utils import Logger
from utils.config import Config

class Server:
    
    def __init__(self, algo=None):
        
        # client info
        self.client_num = 0
        self.max_client_num = 100
        self.client_list = {}
        if algo == 'marl':      
            self.algo = marl_server()
        self.max_idx = 0
        self.suggestion_diff = 0
        assert self.suggestion_diff >= 0 and self.suggestion_diff < Config.SERVER_MAX_BUFFER_LEN
        # server info
        self.base_time = time.time()
        
    def register_client(self):
        if (self.max_idx >= self.max_client_num):
            return -1
        new_idx = self.max_idx
        client = client_info(new_idx)
        self.client_list[new_idx] = client
        self.max_idx += 1 
        self.client_num += 1
        
        # return a suggested gop for download        
        return new_idx, self.jump_suggestion()
        
    def client_exit(self, idx):
        self.client_list.pop(idx)
        self.client_num -= 1
        
    def jump_suggestion(self):
        lower, upper = self.check_range()
        return max(0, upper - self.suggestion_diff)
    
    def process_request(self, request_gop, request_rate):
        start = time.time()
        while(request_gop > self.check_range()[1]):
            pass
        end = time.time()
        prepare = end - start
        lower, upper = self.check_range()
        
        if lower > request_gop:
            suggestion = self.jump_suggestion()
            video_idx = suggestion % Config.SEG_NUM
            video_filename = f"{video_idx}_{request_rate:.1f}" + Config.VIDEO_FORMAT
            suggestion = suggestion + 1
        else:
            video_idx = request_gop % Config.SEG_NUM
            suggestion = request_gop + 1
            video_filename = f"{video_idx}_{request_rate:.1f}" + Config.VIDEO_FORMAT
            
        return suggestion, video_filename, prepare
            
    def check_range(self):
        second = time.time() - self.base_time
        valid = int(second - Config.PUSEDO_ENCODE_TIME)
        lower = max(0, valid - Config.SERVER_MAX_BUFFER_LEN)
        upper = valid - 1
        return (lower, upper)

class client_info:
    
    def __init__(self, idx) -> None:
        self.client_idx = idx
        
        self.last_action = None
        self.last_latency = -1