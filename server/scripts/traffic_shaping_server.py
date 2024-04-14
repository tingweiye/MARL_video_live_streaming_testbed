import subprocess
import platform
import time
import random
from utils.utils import Logger


def regulator(queue, train):
    shaper = traffic_shaper(queue)
    if platform.system() == "Linux":
        if train:
            Logger.log("Training traffic shaper initiated")
            shaper.train_shaping()
        else:
            Logger.log("Testing traffic shaper initiated")
            shaper.test_shaping()
    else:
        Logger.log("System not Linux, traffic shaper not applicable")

class traffic_shaper:
    
    def __init__(self, queue):
        self.train_trace = []
        self.test_trace = []
        
        self.queue = queue
        self.interface = 'eth1'
        self.duration = 0.9999
        self.test_start = 0
        self.episode = 400
        
        self.get_control_number()
        self.read_train_trace('data/traces/50ms_loss0.5_train_all.txt')
        self.read_test_trace('data/traces/50ms_loss0.5_test_0.txt')
        
    
    def set_bandwidth(self, interface, rate, rtt):
        subprocess.call(['sudo', 'tc', 'qdisc', 'replace', 'dev', interface, 'root', 'netem', 'rate', rate, 'delay', f'{rtt}ms'])
        # subprocess.call(['sudo', 'tc', 'qdisc', 'replace', 'dev', interface, 'root', 'tbf', 'rate', rate, 'burst', '32kbit', 'latency', '30ms'])
        
    def get_random_duration(self):
        sample = random.random()
        if sample < 0.1:
            return 8
        elif sample < 0.2:
            return 12
        else:
            return 10
        
    def read_train_trace(self, file_path):
        with open(file_path, "r") as file:
            numbers_text = file.readlines()
            self.train_trace = [float(number.strip()) for number in numbers_text]
            
    def read_test_trace(self, file_path):
        with open(file_path, "r") as file:
            numbers_text = file.readlines()
            self.test_trace = [float(number.strip()) for number in numbers_text]
            
    def train_shaping(self):
        for _ in range(10):
            for r in self.train_trace:
                rate = str(r) + 'Mbit'
                rtt = random.randint(10, 20)
                self.set_bandwidth(self.interface, rate, rtt)
                Logger.log(f"Bandwitdh set to {rate}, RTT set to {rtt}")
                self.queue.put(r)
                time.sleep(self.get_random_duration())
        subprocess.call(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'])
        
    def test_shaping(self):
        Logger.log(f"Test sub trace from {self.test_start*self.episode} to {self.test_start*self.episode+self.episode}")
        for r in self.test_trace[self.test_start*self.episode:self.test_start*self.episode+self.episode+1]:
            rate = str(r) + 'Mbit'
            rtt = random.randint(10, 20)
            self.set_bandwidth(self.interface, rate, rtt)
            Logger.log(f"Bandwitdh set to {rate}, RTT set to {rtt}")
            self.queue.put(r)
            time.sleep(self.duration)
        self.queue.put(-1)
        subprocess.call(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'])
        
    def get_control_number(self):
        with open('.control', 'r') as file:
            content = file.read().strip()
        self.test_start = int(content)

    def test(self):
        rate_start = '10Mbit'
        rate_end = '2Mbit'
        duration = 60
        steps = 8

        rate_start_num = int(rate_start[:-4])
        rate_end_num = int(rate_end[:-4])
        rate_step = (rate_start_num - rate_end_num) // steps

        for i in range(steps):
            rate = str(rate_start_num - i) + 'Mbit'
            print("Setting rate to", rate)
            self.set_bandwidth(self.interface, rate)
            time.sleep(duration / steps)

        subprocess.call(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'])

if __name__ == "__main__":
    shaper = traffic_shaper()
    shaper.test()