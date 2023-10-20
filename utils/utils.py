from datetime import datetime, timedelta
import pandas as pd
from utils.config import Config
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
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                # t1 = time.time()
                # with open(file_path, 'rb') as source_file:
                # # Read the content of the source file
                #     file_content = source_file.read()
                # t2 = time.time()
                # os.remove(file_path)
                # t3 = time.time()
                # # Open the destination file in binary mode for writing
                # with open(file_path, 'wb') as destination_file:
                #     # Write the content to the destination file
                #     destination_file.write(file_content)
                # t4 = time.time()
                # print(f"{t2 - t1}, {t4 - t3}")
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


