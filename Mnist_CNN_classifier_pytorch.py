########################################
# python 3.7.5 pytorch 1.4.0
# 2020 08 12
########################################

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings

batch_size = 128
epochs = 5
lr = 0.1
momentum = 0.9
no_cuda = False
seed = 1

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Define Network, we implement LeNet here
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 6, kernel_size=(5,5),stride=1, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(6, 16, kernel_size=(5,5),stride=1, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)
                    )
        self.fc1 = nn.Sequential(
                        nn.Linear(16 * 4 * 4, 120),
                        nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                        nn.Linear(120, 84),
                        nn.ReLU()
                    )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Learning rate scheduling
def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
       lr = 0.01
    elif epoch < 15:
       lr = 0.001
    else: 
       lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training function
def train(epoch):
    model.train()
    adjust_learning_rate(optimizer, epoch)

    print ('\nStart training ...')
    for batch_idx, (data, label) in enumerate(train_loader):
        if cuda:
            data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = Loss(output, label)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        if cuda:
            data, label = data.to(device), label.to(device)
        output = model(data)
        test_loss += Loss(output, label).item()
        _, pred = torch.max(output, 1) # get the index of the max log-probability
        correct += (pred == label).sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./MNIST_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True,num_workers = 2)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./MNIST_data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True,num_workers = 2)

    # create model
    model = Net()
    print ("Simple model architecture:")
    print (model)
    if cuda:
        device = torch.device('cuda')
        model.to(device)

    # Define optimizer/ loss function
    Loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # Run and save model
    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
        savefilename = 'LeNet_'+str(epoch)+'.mdl'
        torch.save(model.state_dict(), savefilename)
