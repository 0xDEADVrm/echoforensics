import torch
from torch import nn

class CNNBiLSTMAttention(nn.Module):
    """Small CNN -> BiLSTM -> learned temporal attention classifier."""
    def __init__(self, n_classes=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(30, 48, kernel_size=7, padding=3),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=48, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.attention = nn.Sequential(nn.Linear(96, 48), nn.Tanh(), nn.Linear(48, 1))
        self.head = nn.Sequential(
            nn.Linear(96, 48), nn.ReLU(), nn.Dropout(0.25), nn.Linear(48, n_classes)
        )

    def forward(self, x):
        # x: [batch, subcarriers, time]
        z = self.cnn(x).transpose(1, 2)  # [batch, time', channels]
        z, _ = self.lstm(z)
        weights = torch.softmax(self.attention(z).squeeze(-1), dim=1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)
