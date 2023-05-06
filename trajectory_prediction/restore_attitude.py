import csv
import os

import numpy as np
import pandas
import pandas as pd
import torch.nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, num_layers=1, device='cpu'):
        super(RNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.layer = torch.nn.Linear(hidden_size * 2, hidden_size * 4, dtype=torch.float64)
        self.out_layer = torch.nn.Linear(hidden_size * 4, output_size, dtype=torch.float64)

        self.gru_left = torch.nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True,
                                     dtype=torch.float64)
        self.gru_left.to(device)
        self.gru_right = torch.nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True,
                                      dtype=torch.float64)
        self.gru_right.to(device)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, 2, seq_len, input_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        x_left = x[:, 0, :, :]
        x_right = x[:, 1, :, :]
        l_output, l_hidden = self.gru_left(x_left)
        r_output, r_hidden = self.gru_right(x_right)

        l_output = l_output[:, -1, :]
        r_output = r_output[:, -1, :]

        output = self.layer(torch.cat((l_output, r_output), dim=1))

        output = torch.nn.functional.leaky_relu(output)

        return self.out_layer(output)


class FdrDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, seq_len=50, step=1):
        self.root_dir = root_dir
        file_list = os.listdir(root_dir)
        data = [pd.read_csv(os.path.join(root_dir, file)) for file in file_list]
        self.x = [x[["Longitude", "Latitude", "Altitude", "Yaw", "AOA", "AOS", "TAS"]].values for x in data]
        self.y = [y[["Roll", "Pitch"]].values for y in data]

        self.x_min = np.array([-128.63564803540788, 0.0, 0.0, -180, -10, -10, 0])
        self.x_max = np.array([-127.48874388438719, 1.1524208273218974, 6000.0, 180, 10, 10, 1000])

        range_val = self.x_max - self.x_min

        # Perform min-max normalization
        self.x = [(x - self.x_min) / range_val for x in self.x]

        self.seq_len = seq_len
        self.step = step

        self.seq_len_list = [(len(seq) - self.seq_len) // step for seq in self.x]  # length of each sequence

    def __len__(self):
        return sum(self.seq_len_list)

    def __getitem__(self, idx):
        # Always return a sequence of odd length
        seq_index = 0
        while idx > self.seq_len_list[seq_index]:
            idx -= self.seq_len_list[seq_index]
            seq_index += 1
        start = idx * self.step
        end = start + self.seq_len
        if self.seq_len % 2 == 0:
            end -= 1

        mid = (end + start) // 2

        x = torch.tensor(np.array((self.x[seq_index][start:mid], self.x[seq_index][mid + 1:end])))

        return x, self.y[seq_index][mid]


def restore(path):
    model = RNN(7, 256, 2)

    model.load_state_dict(torch.load('model'))

    fdr = pandas.read_csv(path)

    x = fdr[["Longitude", "Latitude", "Altitude", "Yaw", "AOA", "AOS", "TAS"]].values

    x_min = np.array([-128.63564803540788, 0.0, 0.0, -180, -10, -10, 0])
    x_max = np.array([-127.48874388438719, 1.1524208273218974, 6000.0, 180, 10, 10, 1000])

    range_val = x_max - x_min

    x = [(x - x_min) / range_val for x in x]

    seq_length = 49

    d_length = len(fdr)
    y = []

    for i in range(d_length - seq_length * 2):
        start = i
        end = i + seq_length
        mid = (end + start) // 2
        left = x[start:mid]
        right = x[mid + 1:end]
        _x = torch.unsqueeze(torch.tensor((left, right)), dim=0)
        y.append(model(_x)[0].detach().numpy())

    y = np.array(y)
    fdr = fdr[seq_length:-seq_length]

    fdr.loc[:, "Roll"] = y[:, 0]
    fdr.loc[:, "Pitch"] = y[:, 1]

    fdr.to_csv(path[:-4] + "_pred.csv", index=False)


def train(dataset_path, validation_dataset_path):
    sequence_length = 200

    dataset = FdrDataset(dataset_path, seq_len=sequence_length)

    validation_dataset = FdrDataset(validation_dataset_path, seq_len=sequence_length, step=sequence_length)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=500, shuffle=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    model = RNN(7, 256, 2)

    model.train()

    # if model does not exist, create a new one
    if os.path.exists('model'):
        model.load_state_dict(torch.load('model'))

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, metric)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    print(len(dataset))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    print(device)

    epoch_num = 3000

    # plot the loss against epoch
    fig, ax = plt.subplots()
    ax.set(xlabel='Epoch', ylabel='Loss', title='Loss against Epoch')

    loss_list = []
    epoch_list = []

    # Train the model using dataset
    for epoch in range(epoch_num):
        batch_loss = 0
        batch_count = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x)

            # Compute loss
            loss = criterion(y_pred, y)

            # backward pass, update weights
            loss.backward()

            batch_loss += loss.item()
            batch_count += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            scheduler.step()

        # Print loss, epoch and lr
        print('Epoch: %d, Loss: %.5f, LR: %.5f' % (epoch, batch_loss / batch_count, scheduler.get_last_lr()[0]))
        if epoch % 10 == 0:
            # evaluate the model using validation dataset
            with torch.no_grad():
                batch_loss = 0
                batch_count = 0
                for x, y in validation_loader:
                    x = x.to(device)
                    y = y.to(device)
                    # Forward pass
                    y_pred = model(x)

                    # Compute loss
                    loss = criterion(y_pred, y)

                    batch_loss += loss.item()
                    batch_count += 1

                print('Validation Loss: %.5f' % (batch_loss / batch_count))

            torch.save(model.state_dict(), "model")

        loss_list.append(batch_loss / batch_count)
        epoch_list.append(epoch)

    ax.plot(epoch_list, loss_list)
    plt.show()


path = "E:\WT-TacView\data\\train_set\F4U_4B_Flight_64.csv"
dataset_path = "E:\WT-TacView\data\\train_set\smooth"
validation_path = "E:\WT-TacView\data\\train_set\\validation"
restore(path)

# train(dataset_path, validation_path)
