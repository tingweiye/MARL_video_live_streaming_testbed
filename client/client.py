import http.client
import requests
from socket import socket
import time
import queue
from algorithms.stallion import stallion_solver
import threading
import sys
import argparse
sys.path.append("..")
import os
from utils.config import Config
from utils.utils import *

class download_seg_info:
    
    def __init__(self, idx, rate):
        self.idx = idx
        self.rate = rate
        
class req_info:
    
    def __init__(self, idx, rate):
        self.last_idx = idx
        self.last_rate = rate

class Client:
    
    def __init__(self, host, port, algo):
        self.server_host = host
        self.server_port = port
        self.base_url = f'http://{host}:{port}'
        self.base_register_url = '/videoServer/register'
        self.base_exit_url = '/videoServer/exit'
        self.base_get_url = '/videoServer/download'
        self.client_idx = -1
        self.get_next_lock = threading.Lock()
        self.buffer_not_empty = threading.Event()
        self.buffer_not_full = threading.Event()
        self.freeze_avialable = threading.Event()
        self.freeze_avialable.set() # initialize to True
        
        # client info
        self.accumulative_latency = 0.0
        self.base_time = -1.
        self.first_gop = 0
        self.next_gop = 0
        self.last_gop = 0
        self.frame_time = 1 / Config.FPS
        self.play_speed = 1
        # self.current_gop = 0
        
        self.buffer = queue.Queue(Config.CLIENT_MAX_BUFFER_LEN)
        self.rtt = 0.01
        self.idle = 0
        self.freeze = 0
        self.latency = 3.0
        self.download_time = 0
        self.bw = 0 # in Mb/s
        self.jump_seconds = 0
        self.seg_left = 0
        
        self.buffer_his = []
        self.rtt_his = []
        self.idle_his = []
        self.freeze_his = []
        self.latency_his = []
        self.download_time_his = []
        self.rate_his = []
        self.bw_his = []
        self.jump_his = []
        self.server_time_his = []
        
        self.test = time.time()
        year, month, day, hours, minutes, seconds, milliseconds = convert_timestamp(time.time())
        print(f"Client start time: {year}/{month}/{day}:{hours}:{minutes}:{seconds}:{milliseconds}")
        
        if algo == 'stallion':
            self.algo = algo
            self.solver = stallion_solver(Config.INITIAL_LATENCY)
            
        # self.connection = http.client.HTTPConnection(self.server_host, self.server_port)
        self.connection = requests.Session()
    
    """
    Define client registry and exit methods
    """
    
    # register to the server when first connected to it
    def register(self):
        # connection = http.client.HTTPConnection(self.server_host, self.server_port)
        
        # self.connection.request('POST', self.base_register_url)
        
        # response = self.connection.getresponse()
        try:
            response = self.connection.post(self.base_url + self.base_register_url)
            response.raise_for_status()
            
            self.client_idx = int(response.headers.get('idx'))
            self.next_gop = int(response.headers.get('next'))
            self.first_gop = self.next_gop
            # self.base_time = time.time() # TODO
            print(f"Client {self.client_idx} successfully connected to the server {self.server_host}:{self.server_port}")
        except:
            print(f"Client failed to connected to the server {self.server_host}:{self.server_port}")
            raise
            
            
    def exit(self):
        self.playing = False
        headers = {'idx': str(self.client_idx)}
        
        try:
            response = self.connection.post(self.base_url + self.base_exit_url, headers=headers)
            response.raise_for_status()
            print(f"Client {self.client_idx} successfully exited from the server {self.server_host}:{self.server_port}")
        except:
            print(f"Client failed to exited from the server {self.server_host}:{self.server_port}")
            raise
        self.connection.close()

    
        
    """
    Define client initialization methods
    """
        
    def start(self):
        timer = threading.Timer(Config.INITIAL_LATENCY, self.__start_play)
        timer.start()
        while (self.base_time < 0):
            download_start = time.time()
            self.__request_video_seg(Config.INITIAL_RATE)
            download_end = time.time()
            self.download_time = download_end - download_start
            # don't record except for bw
            self.bw = Config.INITIAL_RATE / self.download_time
                
            # wait until buffer is not full
            if self.buffer.full():
                self.buffer_not_full.clear()
                self.buffer_not_full.wait()
            # push to buffer
            self.buffer.put(download_seg_info(self.last_gop, Config.INITIAL_RATE))
            print(f"Buffer: {self.get_buffer_size():.3f}, Latency: {self.latency:.3f}, idle: {self.idle:.3f}, Freeze: {self.freeze:.3f}, Download time: {self.download_time:.3f}, BW: {self.bw:.3f}")
            self.buffer_not_empty.set()
            
    """
    Define client player methods
    """
    
    def get_buffer_size(self):
        return self.buffer.qsize() + self.seg_left
    
    def current_play_seconds(self):
        return self.current_playing + Config.SEG_DURATION - self.seg_left
            
    def __start_play(self):
        # start a thread executing a local video player simulator
        self.player = threading.Thread(target=self.play)
        self.player.start()
        self.base_time = time.time()
        
            
    # Customer method getting segs out of the buffer
    def play(self):
        Logger.log(f"Client {self.client_idx} start playing")
        self.playing = True
        self.current_playing = -1
        ratio = (0.995, 1.005)
        while(self.playing):
            # Wait until the buffer is not empty, and calculate freeze time
            ###################### Handling video freezes ######################
            if self.buffer.empty():
                self.freeze_avialable.clear()
                self.seg_left = 0
                self.buffer_not_empty.clear()
                freeze_start = time.time()
                self.buffer_not_empty.wait()
                freeze_end = time.time()
                self.freeze = freeze_end - freeze_start
                self.freeze_avialable.set()
            ###################### Handling video freezes ######################
            
            start = time.time()
            seg = self.buffer.get()
            self.seg_left = 1
                
            self.current_playing = seg.idx
            # release block for the downloader to put new segments in the buffer
            self.buffer_not_full.set()
            Logger.log(f"Client {self.client_idx} playing segment {seg.idx} at rate {seg.rate}")

            
            for _ in range(int(Config.SEG_DURATION * Config.FPS)):
                time.sleep(self.frame_time / self.play_speed)
                self.seg_left -= self.frame_time#Config.FRAME_DURATION
                # Speed != 1 affects latency. Use accumulative_latency to avoid data integrety issue
                if self.play_speed != 1:
                    self.accumulative_latency -= (self.play_speed - 1) * (self.frame_time)
            end = time.time()
            t1 = end - start
            # print(t1, self.frame_time)
            if t1 > Config.SEG_DURATION / self.play_speed:
                self.frame_time *= ratio[0]
            else:
                self.frame_time *= ratio[1]
        
    
    """
    Define client downloader methods
    """
    
    # http request to get the next gop segment
    # download the requested segment with designated rate and idx in self.next_gop
    def __request_video_seg(self, rate):
        # Define the download URL and filename
        filename = f'{self.next_gop}_{rate:.1f}.mp4'
        download_url = os.path.join(self.base_get_url, filename)  # Replace with the actual URL path
        download_filename = 'd_' + filename  # Replace with the desired local filename
        
        headers = {'idx': str(self.client_idx),
                   'gop': str(self.next_gop),
                   'rate': str(rate)}
        # Create an HTTP connection to the server
        # connection = http.client.HTTPConnection(self.server_host, self.server_port)
        # Send an HTTP GET request to the download URL
        t1 = time.time()
        # self.connection.request('GET', download_url, headers=headers)
        
        # Check if the response status code indicates success (e.g., 200 for OK)
        try:
            response = self.connection.get(self.base_url + download_url, headers=headers)
            response.raise_for_status()
            t2 = time.time()
            
            # Read and save the downloaded content to a local file
            # Get server time and calculate
            server_time = float(response.headers.get('Server-Time'))
            suggestion = int(response.headers.get('suggestion'))
            prepare = float(response.headers.get('Prepare-Time'))

            with open('data/' + download_filename, 'wb') as local_file:
                local_file.write(response.content)
            t3 = time.time()
            
            print(f"Request and response: {t2 - t1}, write: {t3 - t2}")
            # self.connection.close()

            self.last_gop = self.next_gop
            passive_jump = suggestion - self.last_gop - 1
            self.next_gop = suggestion
            return suggestion, prepare, passive_jump, server_time
        except:
            self.connection.close()
            # print(f"Failed to download. Status code: {response.status}")
            raise Exception(f"Failed to download.")

        
    # merchant method putting segs into the buffer
    def download(self, rate, speed=1):
        print("   ")
        
        #############################################################################
        ###################### Adaptive flow control Algorithm ######################
        #############################################################################
        # Use designed algorithm to control the video flow
        # TODO 
        # if self.algo == 'stallion':
        #     self.solver.update_bw_latency(self.bw, self.latency)
        #     rate, _ = self.solver.solve(self.get_buffer_size(), self.latency)
        #############################################################################
        ###################### Adaptive flow control Algorithm ######################
        #############################################################################
        
        # get the next gop and calculate the download time
        download_start = time.time()
        # time.sleep(6) # simulate congestion
        suggestion, prepare, passive_jump, server_time = self.__request_video_seg(rate)
        
        download_end = time.time()
        self.download_time = download_end - download_start - prepare
        
        ######### get freeze time #########
        # release block for the player to play downloaded segments
        self.buffer_not_empty.set()
        # if the video freezes, wait until it finishes calculating the freeze time
        self.freeze_avialable.wait()
        
        ######### get idle time #########
        self.idle = prepare
        # wait until buffer is not full
        full_start = time.time()
        if self.buffer.full():
            self.buffer_not_full.clear()
            self.buffer_not_full.wait()
        full_end = time.time()
        self.idle += full_end - full_start
        
        ######### get latency #########
        if self.latency == Config.INITIAL_DUMMY_LATENCY:
            self.latency = server_time - self.current_play_seconds() - self.rtt
        else:
            self.latency += 0 if self.freeze < 0.00001 else self.freeze # add freeze time
            self.latency += self.accumulative_latency                   # speed correction
            self.latency -= passive_jump                                # latency too high, server forces jump
            self.accumulative_latency = 0.0                             # reset speed correction
        # print(f"Server time: {server_time}, current: {self.current_play_seconds()}")
        # self.latency = server_time - self.current_play_seconds() - self.rtt
            
        ######### get bandwidth #########
        self.bw = rate / self.download_time
        
        ######### get buffer length #########
        # push to buffer
        self.buffer.put(download_seg_info(self.last_gop, rate))
        # buffer_len = self.get_buffer_size()
        Logger.log(f"Client {self.client_idx} downloaded segment {self.last_gop} at rate {rate}")
        # update data
        self.rate_his.append(rate)
        self.server_time_his.append(server_time)
        self.update_data()
        
        return self.latency_his[-1], \
                self.idle_his[-1], \
                self.buffer_his[-1], \
                self.freeze_his[-1], \
                self.download_time_his[-1], \
                self.bw_his[-1], \
                passive_jump, \
                self.server_time_his[-1]
        
    def update_data(self):
        
        print(f"Buffer: {self.get_buffer_size():.3f}, Latency: {self.latency:.3f}, idle: {self.idle:.3f}, Freeze: {self.freeze:.3f}, Download time: {self.download_time:.3f}, BW: {self.bw:.3f}")
        self.buffer_his.append(self.get_buffer_size())
        # self.rtt_his = [] #TODO
        self.idle_his.append(self.idle)
        self.latency_his.append(self.latency)
        self.download_time_his.append(self.download_time)
        self.bw_his.append(self.bw)
        self.freeze_his.append(self.freeze)
        
        if len(self.buffer_his) > Config.MAX_HISTORY:
            self.buffer_his.pop(0)
            self.idle_his.pop(0)
            self.latency_his.pop(0)
            self.download_time_his.pop(0)
            self.bw_his.pop(0)
            self.freeze_his.pop(0)
            self.server_time_his.pop(0)

        self.freeze = 0
        # self.rtt = 0.0
        self.idle = 0
        # self.latency = 0
        # self.download_time = 0
        # self.bw = 0
        
    """
    Client data record methods
    """
        
    def run(self):
        self.register()
        self.start()
        for i in range(610):
            if i % 300 == 0 and i != 0:
                Logger.log("Experinment data saved to results.")
                save_as_csv(self.server_time_his, self.latency_his, f"latency_t_{i}")
                save_as_csv(self.server_time_his, self.rate_his, f"rate_t_{i}")
                save_as_csv(self.server_time_his, self.bw_his, f"bw_t_{i}")
                # save_as_csv(self.server_time_his, self.idle_his, f"idle_t_{i}")
                save_as_csv(self.server_time_his, self.freeze_his, f"freeze_t_{i}")
                save_as_csv(self.server_time_his, self.buffer_his, f"buffer_t_{i}")
                
            self.download()
        self.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streaming client')
    parser.add_argument('--ip', default='127.0.0.1', type=str, help='ip')
    parser.add_argument('--port', default='8080', type=str, help='port')
    parser.add_argument('--algo', default='stallion', type=str, help='ABR algorithm')
    parser.add_argument('--sleep', default=0, type=float, help='Wait time')
    args = parser.parse_args()
    time.sleep(args.sleep)
    delete_files_in_folder('data/')
    client = Client(args.ip, args.port, args.algo)
    client.run()
