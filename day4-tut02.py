# Load packages
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from random import shuffle
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

# Check availability of GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU is available!')

# Define autoencoder class
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64*3, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 64*64*3),
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the network
net = autoencoder()
print(net)

if use_gpu:
    net = net.cuda()

# Define optimization technique
criterion = nn.MSELoss() # Mean Squared Error 
optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.9) # Stochastic Gradient Descent
# optimizer = optim.Adam(net.parameters(), lr=1e-3) # Adam

# Training the autoencoder for representation learning
iterations = 50
for epoch in range(iterations):   
    net.train(True) # For training
    runningLoss = 0
    for data in trainLoader:
        inputs,_  = data # Labels are not required     
        inputs = inputs.view(-1,64*64*3)
        if use_gpu:
            inputs = Variable(inputs).cuda()
        else:
            inputs = Variable(inputs)
        # Initialize the gradients to zero
        optimizer.zero_grad()
        # Feed forward the input data through the network
        outputs = net(inputs)       
        # Compute the error/loss
        loss = criterion(outputs, inputs)
        # Backpropagate the loss to compute gradients
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Accumulate loss per batch
        runningLoss += loss.data[0]
    # Printing average loss per epoch
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iterations,runningLoss/
                                                                (len(trainDataset)/BatchSize)))

# Modifying the autoencoder for classification
# Removing the decoder module from the autoencoder
new_classifier = nn.Sequential(*list(net.children())[:-1])
net = new_classifier
# Adding linear layer for 2-class classification problem
net.add_module('classifier', nn.Sequential(nn.Linear(100, 2)))
print(net)
if use_gpu:
    net = net.cuda()
# Copying initial weights  for visualization
cll_weights = copy.deepcopy(net[0][0].weight.data)
init_classifier_weights = copy.deepcopy(net.classifier[0].weight.data)

# Define loss function and optimizer
criterion = nn.NLLLoss() # Negative Log-Likelihood
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9) # Stochastic gradient descent
# optimizer = optim.Adam(net.parameters(), lr=1e-4) # Adam

# Train the network
iterations = 10
trainLoss = []
testAcc = []
start = time.time()
for epoch in range(iterations):
    epochStart = time.time()
    runningLoss = 0
    for data in trainLoader:
        inputs,labels = data
        inputs = inputs.view(-1,64*64*3)
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # Initialize gradients to zero
        optimizer.zero_grad()
        # Feed-forward input data through the network
        outputs = net(inputs)
        # Compute loss/error
        loss = criterion(F.log_softmax(outputs,dim=1), labels)
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
        inputs = inputs.view(-1,64*64*3)    
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

                                                                    