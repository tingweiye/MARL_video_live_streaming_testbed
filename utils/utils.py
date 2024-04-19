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

def get_allocation(bottlenecks, weights, last_rates, totalBw, client_idx):
    buffer = []
    result = 2.5
    maxScore = 0

    def backtrace(bottlenecks, weights, last_rates, totalBw, client, score):
        if client == len(weights):
            nonlocal maxScore
            nonlocal result
            if score > maxScore:
                maxScore = score
                result = buffer[client_idx]
            return
        for r in Config.REVERSED_BITRATE:
            log_last_rate = np.log(last_rates[client])
            if r < bottlenecks[client] and r <= totalBw:
                log_rate = np.log(r)
                buffer.append(r)
                fair_contribution = weights[client] * log_rate#(log_rate - np.abs(log_rate - log_last_rate))
                # print(r, fair_contribution, log_rate, np.abs(log_rate - log_last_rate))
                backtrace(bottlenecks, weights, last_rates, totalBw-r, client+1, score + fair_contribution)
                buffer.pop()
            else:
                continue
    backtrace(bottlenecks, weights, last_rates, totalBw, 0, 0)
    return result

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
    
    
