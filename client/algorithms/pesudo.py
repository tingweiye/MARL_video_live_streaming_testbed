import logging
import torch
import sys
import numpy as np
from torch.autograd import Variable
from algorithms.components import ac, replay_memory
sys.path.append("..")
from utils.config import Config


SUMMARY_DIR = './results'
HMARL_LOG_FILE = './results/hmarl/log'


TOTALEPOCH=100000


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class pesudo_solver:
    
    def __init__(self, client, weight):
        self.client = client
        self.weight = weight
                        
    def solve(self, train=True):
        logging.basicConfig(filename=HMARL_LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)
    
        with open(HMARL_LOG_FILE + '_record', 'w') as log_file, open(HMARL_LOG_FILE + '_reward', 'w') as reward_file:

            rate = 0
            last_rate = rate
            for epoch in range(TOTALEPOCH):
                
                total_bw = 0
                info = self.client.download(0)
                latency = info["latency"]
                idle = info["idle"]
                buffer_size = info["buffer_size"]
                freeze = info["freeze"]
                download_time = info["download_time"]
                bw = info["bw"]
                jump = info["jump"]
                server_time = info["server_time"]
                instruction = info["instruction"]
                download_rate = info["download_rate"]
                goal = info["goal"]
                intrinsic_reward = info["intrinsic_reward"]
                extrinsic_reward = info["extrinsic_reward"]
                true_bandwidth = info["true_bandwidth"]
                propotional_fairness = info["propotional_fairness"]
                maxmin_fairness = info["maxmin_fairness"]
                client_qoe = info["client_qoe"]
                
                time_stamp = server_time
                last_rate = rate
                rate = download_rate
                last_instruction = instruction
                total_bw += bw

            # log time_stamp, rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                            str(rate) + '\t' +
                            str(bw) + '\t' +
                            str(buffer_size) + '\t' +
                            str(freeze) + '\t' +
                            str(idle) + '\t' +
                            str(latency) + '\t' +
                            str(jump) + '\t' +
                            str(goal) + '\t' +
                            str(intrinsic_reward) + '\t' + 
                            str(extrinsic_reward) + '\t' +
                            str(true_bandwidth) + '\t' +
                            str(propotional_fairness) + '\t' +
                            str(maxmin_fairness) + '\t' +
                            str(client_qoe) + '\n')
                log_file.flush()
                
                if true_bandwidth < 0:
                    return

