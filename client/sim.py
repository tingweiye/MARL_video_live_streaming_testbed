import argparse
import time
from client import Client
import sys
from algorithms import (stallion,
                        pensieve,
                        pensieve_dqn,
                        marl,
                        pesudo)
import numpy as np
import torch
from torch.autograd import Variable
import logging

sys.path.append("..")
import os
from utils.config import Config
from utils.utils import *

parser = argparse.ArgumentParser(description='Streaming client')
parser.add_argument('--ip', default='127.0.0.1', type=str, help='ip')
parser.add_argument('--port', default='8080', type=str, help='port')
parser.add_argument('--algo', default='MARL', type=str, help='ABR algorithm')
parser.add_argument('--sleep', default=0, type=float, help='Wait time')
parser.add_argument('--weight', default=1, type=float, help='Weight of a client')
parser.add_argument('--trail', default=0, type=str, help='# of experinment')
args = parser.parse_args()

STALLION_LOG_FILE = './results/stallion/log'

class Simulator:
    
    def __init__(self, algo):
        self.client = Client(args.ip, args.port, args.algo, args.weight)
        self.algo = algo
        
    def start(self):
        self.client.register()
        self.client.start()
        
    def stop(self):
        self.client.exit()
        
    def stallionRun(self):
        self.solver = stallion.stallion_solver(Config.INITIAL_LATENCY)
        
        with open(STALLION_LOG_FILE + '_record', 'w') as log_file:
            for i in range(610):
                self.solver.update_bw_latency(self.client.bw, self.client.latency)
                rate, _ = self.solver.solve(self.client.get_buffer_size(), self.client.latency)
                latency, idle, buffer_size, freeze, download_time, bw, jump, server_time, _, _ = self.client.download(rate)
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
        self.solver = pensieve_dqn.pensieve_solver(self.client, args.trail)
        self.solver.solve()
    
    def marlRun(self):
        self.solver = marl.marl_solver(self.client, args.weight)
        self.solver.solve()
        
    def hmarlRun(self):
        self.solver = pesudo.pesudo_solver(self.client, args.weight, args.trail)
        self.solver.solve()
        
    def run(self):
        if self.algo == "STALLION":
            self.stallionRun()
        elif self.algo == "PENSIEVE":
            self.pensieveRun()
        elif self.algo == "MARL":
            self.marlRun()
        elif self.algo == "HMARL":
            self.hmarlRun()
        else:
            print("No such algorithm, please select a valid algorithm.")
            exit()


if __name__ == '__main__':
    # time.sleep(args.sleep)
    delete_files_in_folder('data/')
    sim = Simulator(args.algo)

    sim.start()
    sim.run()
    sim.stop()
    # sim.stallionRun()