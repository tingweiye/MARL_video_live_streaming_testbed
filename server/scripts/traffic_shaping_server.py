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
        self.duration = 0.999
        
        self.read_train_trace('data/traces/50ms_loss0.5_train_all.txt')
        self.read_test_trace('data/traces/50ms_loss0.5_test_0.txt')
    
    def set_bandwidth(self, interface, rate):
        subprocess.call(['sudo', 'tc', 'qdisc', 'replace', 'dev', interface, 'root', 'tbf', 'rate', rate, 'burst', '128kbit', 'latency', '30ms'])
        
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
                self.set_bandwidth(self.interface, rate)
                Logger.log(f"Bandwitdh set to {rate}")
                self.queue.put(r)
                time.sleep(self.get_random_duration())
        subprocess.call(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'])
        
    def test_shaping(self):
        for r in self.test_trace[:401]:
            rate = str(r) + 'Mbit'
            self.set_bandwidth(self.interface, rate)
            self.queue.put(r)
            time.sleep(self.duration)
        self.queue.put(-1)
        subprocess.call(['sudo', 'tc', 'qdisc', 'del', 'dev', self.interface, 'root'])
        

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