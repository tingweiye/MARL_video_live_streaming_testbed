from algorithms.marl_server import marl_server
import time
import os
import sys
import threading
sys.path.append("..")
from utils.utils import Logger, delete_files_in_folder
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
        delete_files_in_folder(os.path.join(os.getcwd(), 'data'))
        self.encoder = LiveEncoder()
        self.encoder.start()
        
    def register_client(self):
        if (self.max_idx >= self.max_client_num):
            return -1
        new_idx = self.max_idx
        client = client_info(new_idx)
        self.client_list[new_idx] = client
        self.max_idx += 1 
        self.client_num += 1
        
        # wait until there is something to download
        while(0 > self.encoder.check_range()[1]):
            pass
        
        # return a suggested gop for download        
        return new_idx, self.jump_suggestion()
        
    def client_exit(self, idx):
        self.client_list.pop(idx)
        self.client_num -= 1
        
    def get_server_time(self):
        return self.encoder.get_server_time()
        
    def jump_suggestion(self):
        return max(0, self.encoder.check_range()[1] - self.suggestion_diff)
    
    def process_request(self, request_gop, request_rate):
        start = time.time()
        while(request_gop > self.encoder.check_range()[1]):
            pass
        end = time.time()
        prepare = end - start
        lower, upper = self.encoder.check_range()
        
        if lower > request_gop:
            suggestion = self.jump_suggestion()
            video_idx = suggestion
            video_filename = f"{video_idx}_{request_rate:.1f}" + Config.VIDEO_FORMAT
            suggestion = suggestion + 1
        else:
            video_idx = request_gop
            suggestion = request_gop + 1
            video_filename = f"{video_idx}_{request_rate:.1f}" + Config.VIDEO_FORMAT
            
        return suggestion, video_filename, prepare
            

# A pesudo encoder
class LiveEncoder(threading.Thread):
    
    def __init__(self):
        super(LiveEncoder, self).__init__()
        self.path = os.path.join(os.getcwd(), "data")
        self.correction = 0
        self.decay = 0.995
        self.running = True
        self.latest_completed_play_seg = -1
        self.high = -1
        self.low = 0
        self.base_time = -1
        
        
    
    def generate_files(self, idx):
        for rate in Config.BITRATE:
            numBytes = rate * 1e6 / 8
            file_name = str(idx) + '_' + str(rate) + '.mp4'
            with open(os.path.join(self.path, file_name), 'w') as f:
                f.write('a' * int(numBytes))
                
    def delete_files(self, idx):
        # Iterate over the files and delete each one
        for rate in Config.BITRATE:
            file_name = str(idx) + '_' + str(rate) + '.mp4'
            file_path = os.path.join(self.path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    def pesudo_encode(self, idx):
        # time.sleep(self.encode_time) # simulate encoding time
        self.generate_files(idx)
            
    def get_server_time(self):
        return time.time() - self.base_time
    
    def check_range(self):
        return self.low, self.high
                    
    def run(self):
        self.base_time = time.time()
        Logger.log("Server encoder started")
        while self.running:
            # dynamic control
            self.pesudo_encode(self.high + 1)
            if self.high - self.low + 2 > Config.SERVER_MAX_BUFFER_LEN:
                self.low += 1
                self.delete_files(self.low)
            while int(self.get_server_time()) == self.high + 1:
                pass
            self.high += 1

            print(self.low, self.high, self.get_server_time())
            

class client_info:
    
    def __init__(self, idx):
        self.client_idx = idx
        
        self.last_action = None
        self.last_latency = -1
        