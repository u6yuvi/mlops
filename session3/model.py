import torch
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, ReLU, LogSoftmax, Flatten

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = Sequential(
            Conv2d(1, 32, 3, 1),
            ReLU(),
            Conv2d(32, 64, 3, 1),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(9216, 128),
            ReLU(),
            Linear(128, 10),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)