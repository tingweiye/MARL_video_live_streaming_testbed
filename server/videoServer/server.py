from algorithms.marl_server import marl_server
from algorithms.pesudo_server import pesudo_server
import time
import os
import sys
import threading
sys.path.append("..")
from utils.utils import Logger, delete_files_in_folder
from utils.config import Config

class Server:
    
    def __init__(self, algo="PESUDO"):
        
        # client info
        self.first_time = True
        self.client_num = 0
        self.max_client_num = 100     
        if algo == "MARL":
            self.algo = marl_server()
        else:
            self.algo = pesudo_server()
            
        print(f"Using algorithm {algo}")
        
        self.next_idx = 0
        self.suggestion_diff = 0
        assert self.suggestion_diff >= 0 and self.suggestion_diff < Config.SERVER_MAX_BUFFER_LEN
        # server info
        delete_files_in_folder(os.path.join(os.getcwd(), 'data'))
        self.encoder = LiveEncoder()
        self.encoder.start()
        
    def register_client(self):
        if (self.next_idx >= self.max_client_num):
            return -1
        new_idx = self.next_idx

        if self.first_time:
            time.sleep(1)
            self.first_time = False
            
        self.algo.add_client(new_idx)
        
        self.next_idx += 1 
        self.client_num += 1
        
        # wait until there is something to download
        while(0 > self.encoder.check_range()[1]):
            pass
        
        # return a suggested gop for download        
        return new_idx, self.jump_suggestion()
        
    def client_exit(self, idx):
        self.algo.remove_client(idx)
        self.client_num -= 1
        
    def get_server_time(self):
        return self.encoder.get_server_time()
        
    def jump_suggestion(self):
        return max(0, self.encoder.check_range()[1] - self.suggestion_diff)
    
    def process_request(self, client_idx, request_gop, request_rate):
        # wait until the gop is generated
        start = time.time()
        while(request_gop > self.encoder.check_range()[1]):
            time.sleep(0)
        end = time.time()
        prepare = end - start
        lower, upper = self.encoder.check_range()
        # t1 = time.time()
        # Examine if the requested gop is lagging too much
        if lower > request_gop:
            suggestion = self.jump_suggestion()
            video_idx = suggestion
            video_filename = f"{video_idx}_{request_rate:.1f}" + Config.VIDEO_FORMAT
            suggestion = suggestion + 1
        else:
            video_idx = request_gop
            suggestion = request_gop + 1
            video_filename = f"{video_idx}_{request_rate:.1f}" + Config.VIDEO_FORMAT
        # t2 = time.time()
        # print(f"In processing: wait: {prepare}, check: {t1 - end}, suggest: {t2 - t1}")
        return suggestion, video_filename, prepare
    
    def update_client(self, idx, rate, bw, buffer, latency):
        self.algo.update_info(idx, rate, bw, buffer, latency)
        
    def coordinate_agent(self, idx):
        return self.algo.orchestrate(idx)

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
        
        
    # Generate dummy rate files on the fly
    def generate_files(self, idx):
        for rate in Config.BITRATE:
            numBytes = rate * 1e6 / 8
            file_name = str(idx) + '_' + str(rate) + '.mp4'
            with open(os.path.join(self.path, file_name), 'w') as f:
                f.write('a' * int(numBytes))
                
    # Iterate over the files and delete each one
    def delete_files(self, idx):
        for rate in Config.BITRATE:
            file_name = str(idx) + '_' + str(rate) + '.mp4'
            file_path = os.path.join(self.path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except FileNotFoundError as e:
                print(f"No such file {file_path}: {e}")
    
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
            while int(self.get_server_time()) == self.high + 1:
                time.sleep(0)
            if self.high - self.low + 2 > Config.SERVER_MAX_BUFFER_LEN:
                self.low += 1
                # delay one deletion to avoid data problem
                self.delete_files(self.low - 2)
            self.pesudo_encode(self.high + 1)
            self.high += 1

            # print(self.low, self.high, self.get_server_time())
            


        