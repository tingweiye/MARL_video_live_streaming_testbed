from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils.config import Config
from collections import deque
import time
import os



def convert_timestamp(timestamp):
    # Convert the timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Extract different time components
    year = dt_object.year
    month = dt_object.month
    day = dt_object.day
    hours = dt_object.hour
    minutes = dt_object.minute
    seconds = dt_object.second
    milliseconds = int(dt_object.microsecond / 1000)

    return year, month, day, hours, minutes, seconds, milliseconds

def diff(timestamp1, timestamp2):
    return timestamp2 - timestamp1

def save_as_csv(server_time_his, data, filename, name='data'):
    df = pd.DataFrame({'time': server_time_his, name: data})
    # Save DataFrame to a CSV file
    df.to_csv('results/' + filename, index=False)

class Logger:
    
    def log(message):
        year, month, day, hours, minutes, seconds, milliseconds = convert_timestamp(time.time())
        print(f"[{year}/{month}/{day} {hours}:{minutes}:{seconds}:{milliseconds}]: ", message)
    
    def logTime(time, message):
        year, month, day, hours, minutes, seconds, milliseconds = convert_timestamp(time)
        print(f"[{year}/{month}/{day} {hours}:{minutes}:{seconds}:{milliseconds}]: ", message)
        
def delete_files_in_folder(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files and delete each one
    for file_name in files:
        if file_name == '.gitkeep':
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            
def z(x, threshold):
    if abs(x) < threshold:
        return 0
    elif x >= threshold:
        return -(x - threshold)
    else:
        return -(x + threshold)
            
def zfun(x, threshold0, threshold1):
    assert threshold1 > 0 and threshold0 >= 0
    return min(threshold1, max(-threshold1, z(x, threshold0)))

def zclip(x, threshold0, threshold1):
    assert threshold1 > 0 and threshold0 >= 0
    return -min(threshold1, max(-threshold1, z(x, threshold0)))

def get_allocation(bottlenecks, weights, totalBw, client_idx):
    buffer = {key:2.5 for key, _ in weights.items()}
    result = {key:2.5 for key, _ in weights.items()}
    clients = [key for key, _ in weights.items()]
    maxScore = 0

    def backtrace(bottlenecks, weights, totalBw, client, score):
        if client == len(weights):
            nonlocal maxScore
            nonlocal result
            if score > maxScore:
                maxScore = score
                result = buffer.copy()
            return
        client_idx = clients[client]
        for r in Config.REVERSED_BITRATE:
            if r <= bottlenecks[client_idx] and r <= totalBw:
                buffer[client_idx] = r
                fair_contribution = weights[client_idx] * np.log(r)
                backtrace(bottlenecks, weights, totalBw-r, client+1, score + fair_contribution)
            else:
                continue
    backtrace(bottlenecks, weights, totalBw, 0, 0)
    return result[client_idx]

def get_allocation2(bottlenecks, weights, last_rates, totalBw, client_idx):
    buffer = {key:2.5 for key, _ in weights.items()}
    result = {key:2.5 for key, _ in weights.items()}
    clients = [key for key, _ in weights.items()]
    maxScore = 0
    maxLogRate = 0

    def backtrace(bottlenecks, weights, last_rates, totalBw, client, score, sumLogRate):
        if client == len(weights):
            nonlocal maxScore
            nonlocal maxLogRate
            nonlocal result
            if score > maxScore:
                maxScore = score
                maxLogRate = sumLogRate
                result = buffer.copy()
            if score == maxScore and sumLogRate > maxLogRate:
                maxLogRate = sumLogRate
                result = buffer.copy()
            return
        client_idx = clients[client]
        for r in Config.REVERSED_BITRATE:
            log_last_rate = np.log(last_rates[client_idx])
            if r < bottlenecks[client_idx] and r <= totalBw:
                log_rate = np.log(r)
                buffer[client_idx] = r
                fair_contribution = weights[client_idx] * (log_rate - np.abs(log_rate - log_last_rate))
                # print(r, fair_contribution, log_rate, np.abs(log_rate - log_last_rate))
                backtrace(bottlenecks, weights, last_rates, totalBw-r, client+1, score + fair_contribution, sumLogRate + weights[client_idx] * log_rate)
            else:
                continue
    backtrace(bottlenecks, weights, last_rates, totalBw, 0, 0, 0)
    return result[client_idx]

def get_allocation3(bottlenecks, weights, last_rates, totalBw, client_idx):
    buffer = {key:2.5 for key, _ in weights.items()}
    result = {key:2.5 for key, _ in weights.items()}
    clients = [key for key, _ in weights.items()]
    maxScore = -10000
    maxLogRate = 0

    def backtrace(bottlenecks, weights, last_rates, totalBw, client, score, sumLogRate):
        if client == len(weights):
            nonlocal maxScore
            nonlocal maxLogRate
            nonlocal result
            if score > maxScore:
                print(score, buffer)
                maxScore = score
                maxLogRate = sumLogRate
                result = buffer.copy()
            if score == maxScore and sumLogRate > maxLogRate:
                maxLogRate = sumLogRate
                result = buffer.copy()
            return
        client_idx = clients[client]
        for r in Config.REVERSED_BITRATE:
            log_last_rate = np.log(last_rates[client_idx])
            if r < bottlenecks[client_idx] and r <= totalBw:
                log_rate = np.log(r)
                buffer[client_idx] = r
                QoE_pre = log_rate - np.abs(log_rate - log_last_rate)
                QoE_pre = QoE_pre if QoE_pre > 0 else 0.01
                fair_contribution = weights[client_idx] * np.log(QoE_pre)
                # print(r, fair_contribution, log_rate, np.abs(log_rate - log_last_rate))
                backtrace(bottlenecks, weights, last_rates, totalBw-r, client+1, score + fair_contribution, sumLogRate + weights[client_idx] * np.log(log_rate))
            else:
                continue
    backtrace(bottlenecks, weights, last_rates, totalBw, 0, 0, 0)
    return result[client_idx]

def get_allocation0(bottlenecks, weights, totalBw, client_idx):
    buffer = {key:2.5 for key, _ in weights.items()}
    result = {key:2.5 for key, _ in weights.items()}
    clients = [key for key, _ in weights.items()]
    maxScore = -10

    def backtrace(bottlenecks, weights, totalBw, client, score):
        if client == len(weights):
            nonlocal maxScore
            nonlocal result
            if score > maxScore:
                maxScore = score
                print(score, buffer)
                result = buffer.copy()
            return
        client_idx = clients[client]
        for r in Config.REVERSED_BITRATE:
            if r <= bottlenecks[client_idx] and r <= totalBw:
                buffer[client_idx] = r
                fair_contribution = weights[client_idx] * np.log(np.log(r))
                backtrace(bottlenecks, weights, totalBw-r, client+1, score + fair_contribution)
            else:
                continue
    backtrace(bottlenecks, weights, totalBw, 0, 0)
    return result[client_idx]
    

class MovingQueue:
    def __init__(self, N):
        self.capacity = N
        self.queue = deque(maxlen=N)
        self.add(0)
        
    def __len__(self):
        return len(self.queue)
        
    def add(self, item):
        self.queue.append(item)
        
    def get(self, idx):
        return self.queue[idx]
    
    def sum(self):
        return sum(self.queue, skipna=True)
    
    def avg(self):
        return np.nanmean(self.queue)
    
    def max(self):
        return np.nanmax(self.queue)
    
    def std(self):
        return np.nanstd(self.queue)
    
    def get_last_mean(self, idx):
        return np.mean(list(self.queue)[-idx:])
    
    
