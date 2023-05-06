import os
import csv
import numpy as np
import pandas as pd


def slice(root_dir, file, target_dir, time=10):
    """slice a csv file into smaller files"""
    path = os.path.join(root_dir, file)

    target_dir = os.path.join(target_dir, file[:-4])

    # Load the data into a Pandas dataframe
    df = pd.read_csv(path)

    total_time = df['Time'].max()

    total_rows = len(df)

    num_of_file = int(total_time / time)

    rows_per_file = int(time / (df['Time'][1] - df['Time'][0]))

    # trim the data on both sides
    reminder = total_rows % rows_per_file
    df = df[reminder // 2:total_rows - reminder // 2]

    # Create a new directory to store the sliced files
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # clear the content of the directory
    for f in os.listdir(target_dir):
        os.remove(os.path.join(target_dir, f))

    row = 0

    for i in range(num_of_file):
        df_slice = df[row:row + rows_per_file]
        row += rows_per_file
        target_path = os.path.join(target_dir, 'slice_' + str(i) + '.csv')
        df_slice.to_csv(target_path, index=False)


def re_format(df, dt=0.2):
    # deduct the time column by the first value
    df['Time'] = df['Time'] - df['Time'][0]

    df.set_index('Time', inplace=True)

    start_time = df.index.min()
    end_time = df.index.max()
    total_time = (len(df) - 1) * dt

    # Data Validation check
    if abs(total_time - end_time) > 0.1:
        raise (Exception("Data is not valid, Check param"))
    new_index = np.arange(0, total_time + dt, dt)

    df.index = pd.to_timedelta(new_index, unit='s')

    return df


def smooth(root_dir, file, target_dir, resampling_factor=2):
    path = os.path.join(root_dir, file)

    # Load the data into a Pandas dataframe
    df = pd.read_csv(path)

    # reformat the time offset of that it starts from zero and has constant time interval
    df = re_format(df, dt=0.1)

    dt = (df.index[1] - df.index[0]).total_seconds()

    freq = str(dt / resampling_factor) + 'S'

    df_resampled = df.resample(freq).interpolate(method='linear')

    # Convert the index to a column
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': 'Time'}, inplace=True)

    # Convert the time column to seconds
    df_resampled['Time'] = df_resampled['Time'].dt.total_seconds()

    # Save the smoothed data to a CSV file
    target_path = os.path.join(target_dir, file[:-4] + '_smoothed.csv')
    df_resampled.to_csv(target_path, index=False)


def prepare_data(root_dir, file_name, target_dir, resampling_factor=None):
    """prepare the data for training"""
    smooth(root_dir, file_name, target_dir, resampling_factor=resampling_factor)
    # slice(target_dir, file_name[:-4] + '_smoothed' + '.csv', 'train_set\sliced', time=time)


# name = 'F-80_Fight_160.csv'
# smooth('train_set', name, 'train_set\smooth')
# slice('train_set\smooth', name[:-4] + '_smoothed' + '.csv', 'train_set\sliced', time=10)


prepare_data('train_set', 'F-80_Fight_160.csv', "train_set\smooth", resampling_factor=0.5)
prepare_data('train_set', 'F-80_Flight_1_140.csv', 'train_set\smooth', resampling_factor=0.5)
prepare_data('train_set', 'F-80_Chase_DogFight_146.csv', 'train_set\smooth', resampling_factor=0.5)
prepare_data('train_set', 'F4U_4B_Flight_64.csv', 'train_set\smooth', resampling_factor=0.5)
prepare_data('train_set', 'F4U_4B_Flight_142.csv', 'train_set\smooth', resampling_factor=0.5)
