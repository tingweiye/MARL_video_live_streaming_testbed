from algorithms.client_info import client_info

class pesudo_server:
    
    def __init__(self):
        self.client_list = {}
        
    def add_client(self, idx, rate=0, buffer=0, latency=0):
        info = client_info(idx, rate, buffer, latency)
        self.client_list[idx] = info
    
    def remove_client(self, idx):
        self.client_list.pop(idx)
        
    def update_info(self, idx, rate, buffer, latency):
        self.client_list[idx].update(rate, buffer, latency)