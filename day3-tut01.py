# Load packages

import pickle
from skimage import feature
import struct
from sklearn import preprocessing
import os
from PIL import Image
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RF

# Path to the dataset
dataPath = 'data/day4/'
classes = os.listdir(dataPath+'train')

# Data preparation
train_data = np.zeros((300,50,50),dtype=np.uint8)
count = 0
for c in classes:
    list1 =os.listdir(dataPath+'train/'+c)
    for f in list1:
        train_data[count,:] = np.array(Image.open(dataPath+'train/'+c+'/'+f).resize((50,50)))
        count += 1

    
test_data = np.zeros((100,50,50),dtype=np.uint8)
count = 0
for c in classes:
    list1 =os.listdir(dataPath+'val/'+c)
    for f in list1:
        test_data[count,:] = np.array(Image.open(dataPath+'val/'+c+'/'+f).resize((50,50)))
        count += 1

# Train and test labels
train_label = np.zeros(len(train_data))
train_label[150:] = 1
test_label = np.zeros(len(test_data))
test_label[50:] = 1

# Feature extarction
# Train data
train_greycoHomFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='homogeneity') for x in list(train_data)]
train_greycoConFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='contrast') for x in list(train_data)]
train_greycoEnFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='energy') for x in list(train_data)]
train_greycoCorrFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='correlation') for x in list(train_data)]
train_hogFeat = [feature.hog(x, orientations=2, pixels_per_cell=(5,5)) for x in list(train_data)]
# train_lbpFeat = [feature.local_binary_pattern(x, 5, 3) for x in list(train_data)]

# Test data
test_greycoHomFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='homogeneity') for x in list(test_data)]
test_greycoConFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='contrast') for x in list(test_data)]
test_greycoEnFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='energy') for x in list(test_data)]
test_greycoCorrFeat = [feature.greycoprops(feature.greycomatrix(x, [1], [np.pi/4],normed=True),prop='correlation') for x in list(test_data)]
test_hogFeat = [feature.hog(x, orientations=2, pixels_per_cell=(5,5)) for x in list(test_data)]
# test_lbpFeat = [feature.local_binary_pattern(x, 5, 3) for x in list(test_data)]

# Length of features
print('Length of individual features:')
print(train_greycoHomFeat[0][0].shape)
print(train_greycoConFeat[0][0].shape)
print(train_greycoEnFeat[0][0].shape)
print(train_greycoCorrFeat[0][0].shape)
print(train_hogFeat[0].shape)
# print(train_lbpFeat[0].shape)

# Concatenating features
# Train data
trainFeat = np.zeros((300,1156),dtype=float)
for num in range(300):    
    trainFeat[num][:] = np.concatenate((train_greycoHomFeat[num].reshape(1,),train_greycoConFeat[num].reshape(1,),
                            train_greycoEnFeat[num].reshape(1,),train_greycoCorrFeat[num].reshape(1,),
                                        train_hogFeat[num].reshape(1152,)),axis=0)#,train_data[num].reshape(50*50)),axis=0)#,train_data[num].reshape(50*50)),axis=0)
# Test data
testFeat = np.zeros((100,1156),dtype=float)
for num in range(100):    
    testFeat[num][:] = np.concatenate((test_greycoHomFeat[num].reshape(1,),test_greycoConFeat[num].reshape(1,),
                            test_greycoEnFeat[num].reshape(1,),test_greycoCorrFeat[num].reshape(1,),
                                        test_hogFeat[num].reshape(1152,)),axis=0)#,test_lbpFeat[num].reshape(50*50),axis=0)#,test_data[num].reshape(50*50)),axis=0)

 # Data pre-processing
 # Scaling
trainFeat_scaled = preprocessing.scale(trainFeat)
testFeat_scaled = preprocessing.scale(testFeat)

# Shuffling training data
trainFeat_scaled, train_label = shuffle(trainFeat_scaled,train_label, random_state=0)

# MLP
print('Multi-layer Perceptron')
nn = MLPClassifier(hidden_layer_sizes=(100,50,), max_iter=200, alpha=1e-4,
                    solver='sgd', verbose=True, tol=1e-4, random_state=1,
                    learning_rate_init=1e-3) #alpha - L regularization parameteer
nn.fit(trainFeat_scaled, train_label)       
prediction = nn.predict(testFeat_scaled)

plt.plot(nn.loss_curve_)
plt.xlabel('Epochs')
plt.ylabel('Loss')

print("Training set score: %f" % nn.score(trainFeat_scaled, train_label)) # mean accuracy
print("Test set score: %f" % nn.score(testFeat_scaled, test_label))


# SVM
print('Support Vector Machine')

clf = svm.SVC(kernel='sigmoid') # SVM classifier
clf.fit(trainFeat_scaled,train_label)
svm_prediction = clf.predict(testFeat_scaled)

print("Training set score: %f" % clf.score(trainFeat_scaled, train_label)) # mean accuracy
print("Test set score: %f" % clf.score(testFeat_scaled, test_label))

# Random Forest
print('Random Forest')

rf = RF(n_estimators=50)
rf.fit(trainFeat_scaled,train_label)
print("Training set score: %f" % rf.score(trainFeat_scaled, train_label)) # mean accuracy
print("Test set score: %f" % rf.score(testFeat_scaled, test_label))