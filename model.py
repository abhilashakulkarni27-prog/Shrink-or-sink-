import torch as pt
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # Block 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)

        # Block 2
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(32)

        # Block 3
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(64)

        # MLP
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = pt.relu(self.bn1(self.conv1(x)))
        x = pt.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = pt.relu(self.bn3(self.conv3(x)))
        x = pt.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = pt.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)

        x = x.mean(dim=(2, 3))

        x = pt.relu(self.fc1(x))
        x = self.fc2(x)
        return x