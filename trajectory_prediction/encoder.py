import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        output, hidden = self.gru(x, hidden)
        return output, hidden