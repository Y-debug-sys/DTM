import torch
from torch import nn


class PRE(nn.Module):
    def __init__(self, input_size, output_size, ratio=0.25, rm=None):
        super().__init__()
        self.rm = rm
        self.proj = nn.Sequential(
            nn.Linear(input_size, int(input_size / 4)),
            nn.ReLU(),
            nn.Dropout(ratio),
            nn.Linear(int(input_size / 4), 32),
            nn.ReLU(),
            BiRNN(32, 16),
            nn.ReLU(),
            nn.Linear(32, int(input_size / 4)),
            nn.ReLU(),
            nn.Dropout(ratio),
            nn.Linear(int(input_size / 4), output_size),
            nn.Sigmoid()
        )
        self.log_scale = nn.Parameter(torch.zeros(1, 1, output_size))
        self.act = nn.Sigmoid()

    @property
    def scale(self):
        return self.act(self.log_scale)

    def forward(self, x):
        return self.proj(x) * self.scale
    
    @torch.no_grad()
    def preprocessing(self, x1, x2, mask):
        x = self.proj(x2) * self.scale
        index = (mask==0).reshape(*mask.shape)
        x[~index] = x1[~index]
        return x


class BiRNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.rnn =  nn.LSTM(in_dim, out_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x