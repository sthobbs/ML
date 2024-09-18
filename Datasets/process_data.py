# load in data, convert the Time column to a datetime-string format

import pandas as pd
import glob
from datetime import datetime, timedelta
from pathlib import Path

str_format = '%Y-%m-%d %H:%M:%S.%f'
start_time = '2024-01-01 00:00:00.000000'
start_time = datetime.strptime(start_time, str_format)

input_dir = './Datasets/creditcard_raw'
output_dir = './Datasets/creditcard'

# create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

file_paths = glob.glob(f'{input_dir}/*.csv')

for f in file_paths:

    # read in csv
    df = pd.read_csv(f)

    # convert Time column to a string-datetime field named timestamp
    df.insert(0, "timestamp", None)
    for i in range(len(df)):
        df.loc[i, "timestamp"] = datetime.strftime(start_time + timedelta(minutes=df.loc[i, "Time"]), str_format)
    df.drop(columns=["Time"], inplace=True)

    # save output to csv
    file_name = Path(f).name
    output_file_path = f'{output_dir}/{file_name}'
    df.to_csv(output_file_path, index=False) 
