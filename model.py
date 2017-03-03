#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, normalization,Dropout
from keras.models import model_from_json
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Visualizations will be shown in the notebook.
%matplotlib inline
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import os

# flips the image, saves new image, inverts the steering angle
def flipImage(img,st):
    openName = "."+img
    #print(openName)
    if float(st) == 0:
        st = 0
    else:
        st = float(st)*-1
    flippedImage = cv2.imread(openName)
    flippedImage = cv2.flip(flippedImage,1)
    folder = img.split('/')
    #print(folder)
    imName = folder[2].split('.')
    #print(imName)
    #.split('
    #print(folder)
    #print(imNa
    #flippedImage = cv2.fromArray(flippedImage)
    newName = "./invImg/"+imName[0]+"_inv."+imName[1]
    cv2.imwrite(newName,flippedImage)
    return (imName[0]+"_inv."+imName[1],st)
    
# prepares files for train, test, and validate by splitting driving log   
def prepareData(filename,train_size=.8,test_size=.2,val_size=.3):
    data = []
    f1 = open(filename,'r')
    head = f1.readline()
    for l in f1.readlines():
        data.append(l.strip())
    #print(head)
    print(len(data))
    data = np.array(data)
    trainx,testx,trainy,testy = train_test_split(data,data,test_size=test_size,train_size=train_size)
    testx,valx,testy,valy = train_test_split(testx,testy,test_size=val_size,train_size=val_size)
    f1.close()
    print("Train size = "+str(len(trainx)))
    print("Test size = "+str(len(testx)))
    print("Val size = "+str(len(valx)))
    
    f1 = open("train_2.csv",'w')
    for l in trainx:
        f1.write(l+"\n")
    f1.close()
    
    f1 = open("test_2.csv","w")
    for l in testx:
        f1.write(l+"\n")
    f1.close()
    
    f1 = open("val_2.csv","w")
    for l in valx:
        f1.write(l+"\n")
    f1.close()
    
    return len(trainx),len(testx),len(valx)

# data generator to resolve memory issues
def data_generator(path):
    while 1:
        f = open(path,'r')
        for line in f:
            x, y = process_line(line)
            img = np.array(process_image(x))
            img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
            yield (img,np.array([y]))
        f.close()
        
def normalizeGrey(img):
    a = 0 # lower norm value
    b = 1 # upper norm value
    
    minGrey = 0
    maxGrey = 255
    return a + ( ( (img - minGrey)*(b - a) )/( maxGrey - minGrey ) )

def val_generator(path):
    while 1:
        f2 = open(path,'r')
        for line in f2:
            x, y = process_line(line)
            img = process_image(x)
            img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
            yield (img,np.array([y]))
        f2.close()

# image preprocessing function used by generator to resize image   
def process_image(img):
    neg = Image.open("."+img)
    neg =  np.asarray(neg.resize((neg.size[0]//2,neg.size[1]//2)))
    return neg

# processes file line to find image file and steering angle
def process_line(line):
    temp = line.strip().split(",")
    x = temp[0]
    y = float(temp[3].strip())
    return x,y
    
def normalize(f):
    lmin = float(f.min())
    lmax = float(f.max())
    return np.floor((f-lmin)/(lmax-lmin)*255.)
    
    
# create a directory to store the inverted data    
os.mkdir("./invImg")

# Read the driving log to get image names
f1 = open("driving_log.csv",'r')
files = []
for l in f1.readlines():
    temp = l.split(",")
    files.append((temp[0].strip(),temp[3].strip()))
f1.close()

# read in each file and generated a flipped and inverted example for each
invertedImg = []
for c in files:
    # flip the image
    flipped = flipImage(c[0],c[1])
    # save the new image name
    newImgName = "/IMG/"+flipped[0]
    invertedImg.append((newImgName,flipped[1]))
 
# add the new images to the driving log  
f1 = open("driving_log.csv",'a')
for i in invertedImg:
    f1.write(i[0]+", , ,"+str(i[1])+", , ,\n")
f1.close()


# prepare the traing, test and validation data
trainSize,testSize,valSize = prepareData("combined_log_inv.csv")

# dropout prob for dropout layers
dropout_prob = .05

# begin model definition

# paramters for layer sizes
fullDimensions = {
    'full1':100,
    'full2':50,
    'full3':10,
    'full4':1,
}


layerDepth = {
    'convo1':24,
    'convo2':36,
    'convo3':48,
    'convo4':64,
    'convo5':64
}

kernalRow = {
    'convo1':5,
    'convo2':5,
    'convo3':5,
    'convo4':3,
    'convo5':3
}

kernalColumn = {
    'convo1':5,
    'convo2':5,
    'convo3':5,
    'convo4':3,
    'convo5':3
}
stride ={
    'convo1':(2,2),
    'convo2':(2,2),
    'convo3':(2,2),
    'convo4':(1,1),
    'convo5':(1,1)
}

# sequential model layers
model = Sequential()
# normalization layer
model.add(normalization.BatchNormalization(mode=2,input_shape=(80,160,3)))
# convolution layer with pooling_1 : 5x5
model.add(Convolution2D(layerDepth['convo1'], kernalRow['convo1'],kernalColumn['convo1'],
        subsample=stride['convo1'], border_mode='valid', dim_ordering='tf',input_shape=(80,160,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(kernalRow['convo1'],kernalColumn['convo1']),
        strides=(1,1),border_mode='same',dim_ordering='tf'))

# convolution layer with pooling_2 : 5x5
model.add(Convolution2D(layerDepth['convo2'], kernalRow['convo2'],kernalColumn['convo2'],
        subsample=stride['convo2'], border_mode='valid', dim_ordering='tf'))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(kernalRow['convo2'],kernalColumn['convo2']),
        strides=(1,1),border_mode='same',dim_ordering='tf'))

# convolution layer with pooling_3 : 5x5
model.add(Convolution2D(layerDepth['convo3'], kernalRow['convo3'],kernalColumn['convo3'],
        subsample=stride['convo3'], border_mode='valid', dim_ordering='tf'))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(kernalRow['convo3'],kernalColumn['convo3']),
        strides=(1,1),border_mode='same',dim_ordering='tf'))
model.add(Dropout(dropout_prob))

# convolution layer with pooling_3 : 3x3
model.add(Convolution2D(layerDepth['convo4'], kernalRow['convo4'],kernalColumn['convo4'],
        subsample=stride['convo4'], border_mode='valid', dim_ordering='tf'))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(kernalRow['convo4'],kernalColumn['convo4']),
        strides=(1,1),border_mode='same',dim_ordering='tf'))

# convolution layer with pooling_3 : 3x3
model.add(Convolution2D(layerDepth['convo5'], kernalRow['convo5'],kernalColumn['convo5'],
        subsample=stride['convo5'], border_mode='valid', dim_ordering='tf'))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(kernalRow['convo5'],kernalColumn['convo5']),
        strides=(1,1),border_mode='same',dim_ordering='tf'))

# flatten outputs
model.add(Flatten())
#fully connected layer_1: 100
model.add(Dense(fullDimensions['full1'],activation='relu'))
# dropout_1
# fully connected layer_2:50
model.add(Dense(fullDimensions['full2'],activation='relu'))
# dropout
model.add(Dropout(dropout_prob))
# fully connected layer_3:10
model.add(Dense(fullDimensions['full3'],activation='relu'))
# fully connected layer_4:1 with linear activation
model.add(Dense(fullDimensions['full4'],activation='linear'))

print("Training the model ...")
# train model using mean squar error and adam optimizer
model.compile(loss='mse',
              optimizer=Adam(lr=0.00001),
              metrics=['accuracy','mean_squared_error'])
history = model.fit_generator(data_generator('train_2.csv'),
        samples_per_epoch=trainSize, nb_epoch=6,verbose =1,
        validation_data = val_generator('test_2.csv'),nb_val_samples=testSize)

# print the loss to standard outputs
print("Training Loss: ")
for h in history.history['val_mean_squared_error']:
	print(h)
	
print("Validating the model ....")
# validate the model
eval_history = model.evaluate_generator(data_generator('val.csv'), val_samples=valSize, max_q_size=10, nb_worker=1, pickle_safe=False)


f1 = open('val.csv','r')
actual = []
for l in f1.readlines():
    temp = l.split(",")
    val = float(temp[3].strip())
    actual.append(val)
f1.close()

samples = len(actual)

count = 0
totalLoss = 0
for e in eval_history:
    totalLoss += pow(e-actual[count],2)
    count+=1

avgLoss = float(totalLoss)/float(samples)
print("Validation error: "+str(avgLoss))


print("Saving the model ...")
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk.....")
