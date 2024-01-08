import argparse
import time
from client import Client
import sys
from algorithms import (stallion)
import numpy as np
import torch
from torch.autograd import Variable
import logging

from algorithms.components import ac, replay_memory
from algorithms import pensieve

sys.path.append("..")
import os
from utils.config import Config
from utils.utils import *

parser = argparse.ArgumentParser(description='Streaming client')
parser.add_argument('--ip', default='127.0.0.1', type=str, help='ip')
parser.add_argument('--port', default='8080', type=str, help='port')
parser.add_argument('--algo', default='stallion', type=str, help='ABR algorithm')
parser.add_argument('--sleep', default=0, type=float, help='Wait time')
args = parser.parse_args()

STALLION_LOG_FILE = './results/stallion/log'

class Simulator:
    
    def __init__(self, algo):
        self.client = Client(args.ip, args.port, args.algo)
        self.algo = algo
        
    def start(self):
        self.client.register()
        self.client.start()
        
    def stallionRun(self):
        self.solver = stallion.stallion_solver(Config.INITIAL_LATENCY)
        
        with open(STALLION_LOG_FILE + '_record', 'w') as log_file:
            for i in range(610):
                self.solver.update_bw_latency(self.client.bw, self.client.latency)
                rate, _ = self.solver.solve(self.client.get_buffer_size(), self.client.latency)
                latency, idle, buffer_size, freeze, download_time, bw, jump, server_time = self.client.download(rate)
            log_file.write(str(server_time) + '\t' +
                        str(rate) + '\t' +
                        str(bw) + '\t' +
                        str(buffer_size) + '\t' +
                        str(freeze) + '\t' +
                        str(idle) + '\t' +
                        str(latency) + '\t' +
                        str(jump) + '\t')
            log_file.flush()
            
            
    def pensieveRun(self):
        self.solver = pensieve.pensieve_solver(self.client)
        self.solver.solve()


if __name__ == '__main__':
    # time.sleep(args.sleep)
    delete_files_in_folder('data/')
    sim = Simulator(args.algo)

    sim.start()
    sim.pensieveRun()
    # sim.stallionRun()