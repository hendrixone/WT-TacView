import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the data
file_path = "your_data.csv"
df = pd.read_csv(file_path)

# Drop the Time_offset column as it is not needed for training
df = df.drop('Time_offset', axis=1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df)

# Convert the normalized data to a DataFrame
df_normalized = pd.DataFrame(data_normalized, columns=df.columns)


def create_sequences(data, input_seq_len, target_seq_len):
    input_seq = []
    target_seq = []

    for i in range(len(data) - input_seq_len - target_seq_len + 1):
        input_seq.append(data[i : i + input_seq_len])
        target_seq.append(data[i + input_seq_len : i + input_seq_len + target_seq_len])

    return np.array(input_seq), np.array(target_seq)


input_seq_len = 20
target_seq_len = 20

input_data, target_data = create_sequences(df_normalized.values, input_seq_len, target_seq_len)

train_X, val_X, train_Y, val_Y = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_Y = torch.tensor(train_Y, dtype=torch.float32)
val_X = torch.tensor(val_X, dtype=torch.float32)
val_Y = torch.tensor(val_Y, dtype=torch.float32)