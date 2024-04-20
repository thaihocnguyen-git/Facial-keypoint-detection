## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 7)
        xavier_uniform(self.conv1.weight)        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        xavier_uniform(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        xavier_uniform(self.conv3.weight)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        xavier_uniform(self.conv4.weight)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.1)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 1)
        xavier_uniform(self.conv5.weight)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(0.1)
        self.batch_norm5 = nn.BatchNorm2d(512)
        
        self.dense1 = nn.Linear(12800, 1024)
        xavier_uniform(self.dense1.weight)
        self.dense2 = nn.Linear(1024, 136)          
        xavier_uniform(self.dense2.weight)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x) 
  
        x = x.view(x.shape[0],-1)
              
        x = F.relu(self.dense1(F.relu(x)))
        x = self.dense2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
