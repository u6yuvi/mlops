import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Define your model architecture here

    def forward(self, x):
        # TODO: Define the forward pass
        pass

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # TODO: Implement the training loop here
    pass

def test_epoch(model, device, data_loader):
    # TODO: Implement the testing loop here
    pass

def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    # TODO: Define your command line arguments here
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # TODO: Load the MNIST dataset for training and testing
    
    model = Net().to(device)
    # TODO: Add a way to load the model checkpoint if 'resume' argument is True

    # TODO: Choose and define the optimizer here
    
    # TODO: Implement the training and testing cycles
    # Hint: Save the model after each epoch

if __name__ == "__main__":
    main()
