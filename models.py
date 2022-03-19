import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, indim, classes):
        super().__init__()
        self.l1 = nn.Linear(indim, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        out = torch.softmax(self.l5(x), 1)

        return out

class CNN(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, (3,3))
        self.conv2 = nn.Conv2d(32, 64, (3,3))
        self.mp1 = nn.MaxPool2d((2,2))
        self.mp2 = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(2304, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 128)
        self.l4 = nn.Linear(128, classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.mp1(self.conv1(x)))
        x = self.relu(self.mp1(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        out = torch.softmax(self.l4(x), 1)

        return out

class ConvNeXt(nn.Module):
    def __init__(self):
        pass