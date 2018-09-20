############################### Transfer Learnign using AlexNet  ####################################

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
from torchvision import models
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
input_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
# Creating pytorch dataset 
trainDataset = ImageFolder('data/day4/train/', transform=input_transform)
valDataset = ImageFolder('data/day4/val/', transform=input_transform)
# Creating dataloader
BatchSize = 32
trainLoader = DataLoader(trainDataset, batch_size=BatchSize, shuffle=True,num_workers=4, pin_memory=pinMem)
valLoader = DataLoader(valDataset, batch_size=BatchSize, shuffle=True,num_workers=4, pin_memory=pinMem)

# Initialize the network
# AlexNet
net = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
new_classifier.add_module('fc',nn.Linear(4096,2))
net.classifier = new_classifier
print(net)
if use_gpu:
    net = net.cuda()

# Visualization of the weights
# functions to display an image
def imshow(img, strlabel):
    npimg = img.numpy()
    npimg = np.abs(npimg)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.figure()
    plt.title(strlabel)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Copying initial weights for visualization
init_weightConv1 = copy.deepcopy(net.features[0].weight.data.cpu()) # 1st conv layer
init_weightConv2 = copy.deepcopy(net.features[3].weight.data.cpu()) # 2nd conv layer

imshow(torchvision.utils.make_grid(init_weightConv1,nrow=8,normalize=True),'Initial weights: conv1')
plt.show()

# Define loss function and optimizer
criterion = nn.NLLLoss() # Negative Log-Likelihood
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9) # Stochastic gradient descent

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
        loss = criterion(F.log_softmax(outputs,dim=1), labels)        
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()
        # Accumulate loss per batch
        runningLoss += loss.item()    
    avgTrainLoss = runningLoss/200
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
        
    # Plotting Loss vs Epochs
    fig1 = plt.figure(1)        
    plt.plot(range(epoch+1),trainLoss,'r--',label='train')        
    if epoch==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')    
    # Plotting testing accuracy vs Epochs
    fig2 = plt.figure(2)        
    plt.plot(range(epoch+1),testAcc,'g-',label='test')        
    if epoch==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Testing accuracy')    
    epochEnd = time.time()-epochStart
    print('At Iteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f} ; Testing Acc: {:.3f} ; Time consumed: {:.0f}m {:.0f}s '\
          .format(epoch + 1,iterations,avgTrainLoss,avgTestAcc*100,epochEnd//60,epochEnd%60))
end = time.time()-start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))

# Copying trained weights for visualization
trained_weightConv1 = copy.deepcopy(net.features[0].weight.data.cpu())
trained_weightConv2 = copy.deepcopy(net.features[3].weight.data.cpu())

imshow(torchvision.utils.make_grid(init_weightConv1,nrow=8,normalize=True),'Initial weights: conv1')
imshow(torchvision.utils.make_grid(trained_weightConv1,nrow=8,normalize=True),'Trained weights: conv1')
imshow(torchvision.utils.make_grid(init_weightConv1-trained_weightConv1,nrow=8,normalize=True),'Difference of weights: conv1')

imshow(torchvision.utils.make_grid(init_weightConv2[0].unsqueeze(1),nrow=8,normalize=True),'Initial weights: conv2')
imshow(torchvision.utils.make_grid(trained_weightConv2[0].unsqueeze(1),nrow=8,normalize=True),'Trained weights: conv2')
imshow(torchvision.utils.make_grid(init_weightConv2[0].unsqueeze(1)-trained_weightConv2[0].unsqueeze(1),nrow=8,normalize=True),'Difference of weights: conv2')

plt.show()
