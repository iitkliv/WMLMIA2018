# Load packages
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time

# Load data
# Transformation to be applied on the input images
input_transform = transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor()])
# Creating pytorch dataset 
trainDataset = ImageFolder('data/day4/train/', transform=input_transform)
valDataset = ImageFolder('data/day4/val/', transform=input_transform)
# Creating dataloader
BatchSize = 32
trainLoader = DataLoader(trainDataset, batch_size=BatchSize, shuffle=True,num_workers=4, pin_memory=True)
valLoader = DataLoader(valDataset, batch_size=BatchSize, shuffle=True,num_workers=4, pin_memory=True)

print(len(trainDataset))
print(len(valDataset))

# Check availability of GPU
use_gpu = torch.cuda.is_available()
pinMem = False
if use_gpu:
    print('GPU is available!')
    pinMem = True 


# Transformation to be applied on the input images
input_transform = transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor()])
# Creating pytorch dataset 
trainDataset = ImageFolder('data/day4/train/', transform=input_transform)
valDataset = ImageFolder('data/day4/val/', transform=input_transform)
# Creating dataloader
BatchSize = 32
trainLoader = DataLoader(trainDataset, batch_size=BatchSize, shuffle=True,num_workers=4, pin_memory=pinMem)
valLoader = DataLoader(valDataset, batch_size=BatchSize, shuffle=True,num_workers=4, pin_memory=pinMem)

# Define network architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out),dim=1)
        return out

# Initialize network
net = LeNet()
print(net)

if use_gpu:
    net = net.cuda()

# Define loss function and optimizer
criterion = nn.NLLLoss() # Negative Log-Likelihood
# optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.9) # Stochastic gradient descent
optimizer = optim.Adam(net.parameters(), lr=1e-3) # Adam

# Train the network
iterations = 10
trainLoss = []
testAcc = []
start = time.time()
for epoch in range(iterations):
    epochStart = time.time()
    runningLoss = 0    
    net.train(True) # For training
    for data in trainLoader:
        inputs,labels = data
        # Wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.float().cuda()), \
                Variable(labels.long().cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labelslong())         
        # Initialize gradients to zero
        optimizer.zero_grad()
        # Feed-forward input data through the network
        outputs = net(inputs)
        # Compute loss/error
        loss = criterion(outputs, labels)
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()
        # Accumulate loss per batch
        runningLoss += loss.item()  
    avgTrainLoss = runningLoss/300
    trainLoss.append(avgTrainLoss)
    
    # Evaluating performance on test set for each epoch
    net.train(False) # For testing
    correct = 0
    for data in valLoader:
        inputs,labels = data
    
        if use_gpu:
            inputs = Variable(inputs.cuda())
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
        else:
            inputs = Variable(inputs)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)               
        correct += (predicted == labels).sum()
    avgTestAcc = correct.numpy()/100.0
    testAcc.append(avgTestAcc)        
    
    epochEnd = time.time()-epochStart
    print('At Iteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f} ; Testing Acc: {:.3f} ; Time consumed: {:.0f}m {:.0f}s '\
          .format(epoch + 1,iterations,avgTrainLoss,avgTestAcc*100,epochEnd//60,epochEnd%60))
end = time.time()-start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))

# Plotting training loss vs Epochs
fig1 = plt.figure(1)        
plt.plot(range(epoch+1),trainLoss,'r-',label='train')        
if epoch==0:
    plt.legend(loc='upper left')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')   
# Plotting testing accuracy vs Epochs
fig2 = plt.figure(2)        
plt.plot(range(epoch+1),testAcc,'g-',label='test')        
if epoch==0:
    plt.legend(loc='upper left')
    plt.xlabel('Epochs')
    plt.ylabel('Testing accuracy')   

