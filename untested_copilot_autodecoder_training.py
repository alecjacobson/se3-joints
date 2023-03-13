#PyTorch method to train an autodecoder which maps from a 2D latent space to a 6D vector space

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from IPython.display import HTML
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import time
import os
import sys

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Set up formatting for the movie files
rc('animation', html='html5')

#Define the autodecoder class
class AutoDecoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
#Define the dataset class
class AutoDecoderDataset(Dataset):
        
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            sample = self.data[idx]
            if self.transform:
                sample = self.transform(sample)
            return sample
        
#Define the training function
def train(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            data = Variable(data)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                       %(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                
#Define the testing function
def test(model, test_loader, criterion):
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = Variable(data)
            output = model(data)
            loss = criterion(output, data)
            print ('Test Loss: %.4f' %(loss.item()))

#Define the plotting function
def plot(model, test_loader):
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = Variable(data)
            output = model(data)
            output = output.data.numpy()
            data = data.data.numpy()
            plt.figure()
            plt.scatter(data[:,0], data[:,1], c='r', label='Original data')
            plt.scatter(output[:,0], output[:,1], c='b', label='Reconstructed data')
            plt.legend(loc='best')
            plt.show()
            plt.pause(0.001)
            if i == 4:
                break
                
#Define the plotting function
def plot2(model, test_loader):
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = Variable(data)
            output = model(data)
            output = output.data.numpy()
            data = data.data.numpy()
            plt.figure()
            plt.scatter(data[:,0], data[:,1], c='r', label='Original data')
            plt.scatter(output[:,0], output[:,1], c='b', label='Reconstructed data')
            plt.legend(loc='best')
            plt.show()
            plt.pause(0.001)
            if i == 9:
                break
                
#Define the plotting function
def plot3(model, test_loader):
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = Variable(data)
            output = model(data)
            output = output.data.numpy()
            data = data.data.numpy()
            plt.figure()
            plt.scatter(data[:,0], data[:,1], c='r', label='Original data')
            plt.scatter(output[:,0], output[:,1], c='b', label='Reconstructed data')
            plt.legend(loc='best')
            plt.show()
            plt.pause(0.001)
            if i == 14:
                break
                



