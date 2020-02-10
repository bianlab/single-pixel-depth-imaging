from __future__ import division

from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import scipy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
 

EPOCH = 100
BATCH_SIZE = 1
LR = 0.001         # learning rate

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(        # input shape (1, 128, 128)

            nn.Conv2d(

                in_channels=1,              # input height

                out_channels=50,            # n_filters

                kernel_size=3,              # filter size

                stride=1,                   # filter movement/step

                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1

            ),                              # output shape (50, 128, 128)

            nn.ReLU(),                      # activation
            
            #pooling 
            nn.AdaptiveMaxPool2d((64,64)),    # choose max value in 2x2 area, output shape (50, 128, 128)
        )
            
  
        self.block1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )   
            
        self.block3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )   
        
        self.block4 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )   

    ### END ###   (50,128,128)
        
        self.conv2 = nn.Sequential(         # input shape (50, 128, 128)

            nn.Conv2d(50, 1, 3, 1, 1),     # output shape (50, 128, 128)

            nn.ReLU(),                      # activation

        )
        
             
       
       


        
    def forward(self, x):
        
        
        x = self.conv1(x)

        residual1 = x    #Save input as residual
        x = self.block1(x)
    
        x += residual1 #add input to output of block1
        residual2 = x  #save output of block1 as residual
        
        x = self.block2(x)
        
        x += residual2 #add output of block1 to output of block2
        residual3 = x
        
        x = self.block3(x)
        
        x += residual3 #add output of block2 to output of block3
        residual4 = x
        
        x = self.block4(x)
        x += residual4 #add output of block3 to output of block4
        
        output = self.conv2(x)
  
        #output = self.out(x)   #(1,128,128)

        return output    # return x for visualization



autoencoder = CNN()
autoencoder=autoencoder.cuda()
autoencoder.train()
autoencoder=torch.load("/home/divinezeng/unn/netD.pt")
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()









dataloader1 = DataLoader(ImageFolder('/home/divinezeng/unn/origin/', img_size=64),
                        batch_size=1, shuffle=False)

dataloader2 = DataLoader(ImageFolder('/home/divinezeng/unn/origin/', img_size=64),
                        batch_size=1, shuffle=False)



for epoch in range(EPOCH):
    torch.save(autoencoder,"/home/divinezeng/unn/netD.pt")
    for a,b in zip(enumerate(dataloader1),enumerate(dataloader2)):

        in_img=a[1][1]
        f=open("/home/divinezeng/unn/3d/"+str(a[0]+1)+".txt","r")
        e=np.eye(64)
        for number in range(64):
           r=f.readline()
           e[number][0],e[number][1],e[number][2],e[number][3],e[number][4],e[number][5],e[number][6],e[number][7],e[number][8],e[number][9],e[number][10],e[number][11],e[number][12],e[number][13],e[number][14],e[number][15],e[number][16],e[number][17],e[number][18],e[number][19],e[number][20],e[number][21],e[number][22],e[number][23],e[number][24],e[number][25],e[number][26],e[number][27],e[number][28],e[number][29],e[number][30],e[number][31],e[number][32],e[number][33],e[number][34],e[number][35],e[number][36],e[number][37],e[number][38],e[number][39],e[number][40],e[number][41],e[number][42],e[number][43],e[number][44],e[number][45],e[number][46],e[number][47],e[number][48],e[number][49],e[number][50],e[number][51],e[number][52],e[number][53],e[number][54],e[number][55],e[number][56],e[number][57],e[number][58],e[number][59],e[number][60],e[number][61],e[number][62],e[number][63]=r.split(",")
           '''for x in range(64):
              e[number][x]=e[number][x]*10'''

        e=torch.Tensor(e)
        target=e
        in_img=in_img.unsqueeze(0)
        in_img=in_img.type(torch.cuda.FloatTensor)
        target=target.type(torch.cuda.FloatTensor)
        decoded = autoencoder(in_img)
        loss = loss_func(decoded[0], target)# mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if a[0] % 1 == 0:
             loss=loss.type(torch.FloatTensor)
             decoded=decoded.type(torch.FloatTensor)
             print('Epoch: ', epoch ,a[0], '| train loss: %.4f' % loss.data.numpy())
             #scipy.misc.imsave("/home/nvidia/Desktop/unn/f/re"+str(a[0])+".jpg",decoded.data.numpy()[0][0])
             loss=loss.type(torch.cuda.FloatTensor)
             decoded=decoded.type(torch.cuda.FloatTensor)

        if a[0] % 100 == 0:
             decoded=decoded.type(torch.FloatTensor)
             target=target.type(torch.FloatTensor)
             #test out####################################################33
             fig = plt.figure()
             ax = fig.gca(projection='3d')
             buff=decoded[0].data.numpy()[0]
             X = np.arange(-16, 16, 0.5)
             Y = np.arange(-16, 16, 0.5)
             X, Y = np.meshgrid(X, Y)
             Z = X+Y
             for x in range(64):
                for y in range(64):
                  Z[x][y]=buff[x][y]
             surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
             ax.set_zlim(0, 40)
             ax.zaxis.set_major_locator(LinearLocator(10))
             ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
             fig.colorbar(surf, shrink=0.5, aspect=5)
             plt.show()

             target=target.type(torch.cuda.FloatTensor)
             decoded=decoded.type(torch.cuda.FloatTensor)
             ####################################################################
             #yuan tu ##########################################################
             '''fig = plt.figure()
             ax = fig.gca(projection='3d')
             X = np.arange(-16, 16, 0.5)
             Y = np.arange(-16, 16, 0.5)
             X, Y = np.meshgrid(X, Y)
             Z = X+Y
             for x in range(64):
                for y in range(64):
                  Z[x][y]=e[x][y]
             surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
             ax.set_zlim(0, 40)
             ax.zaxis.set_major_locator(LinearLocator(10))
             ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
             fig.colorbar(surf, shrink=0.5, aspect=5)
             plt.show()'''
             #####################################################################
             '''cm=plt.cm.get_cmap('bwr')
             xy=range(0)
             z=xy
             sc=plt.scatter(xy,xy,c=z,vmin=-1.5,vmax=1.5,s=35,cmap=cm)
             plt.colorbar(sc)
             test=(target-decoded).data.numpy()[0][0]
             plt.imshow(test,cmap=cm)
             plt.show()'''
             target=target.type(torch.cuda.FloatTensor)
             decoded=decoded.type(torch.cuda.FloatTensor)

