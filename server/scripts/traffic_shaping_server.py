import subprocess
import time

def regulator(queue):
    shaper = traffic_shaper()

class traffic_shaper:
    
    def __init__(self):
        self.train_trace = []
        self.test_trace = []
        
        self.interface = 'eth1'
        self.duration = 10
        
        self.read_train_trace('data/traces/50ms_loss0.5_train_all.txt')
        self.read_test_trace('data/traces/50ms_loss0.5_test_0.txt')
    
    def set_bandwidth(self, interface, rate):
        subprocess.call(['sudo', 'tc', 'qdisc', 'replace', 'dev', interface, 'root', 'tbf', 'rate', rate, 'burst', '32kbit', 'latency', '30ms'])
        
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
                time.sleep(self.duration)
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