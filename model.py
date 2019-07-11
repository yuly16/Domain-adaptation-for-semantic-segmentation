import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FCN_nd(torch.nn.Module):
    def __init__(self):
        super(FCN_nd,self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv3d(1,32,3,padding=1),nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv3d(32,32,3,padding=1),nn.ReLU())
        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv1 = nn.Sequential(self.conv1_1,self.conv1_2,self.pool1)

        self.conv2_1 = nn.Sequential(nn.Conv3d(32,64,3,padding=1),nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv3d(64,64,3,padding=1),nn.ReLU())
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Sequential(self.conv2_1,self.conv2_2,self.pool2)

        self.conv3_1 = nn.Sequential(nn.Conv3d(64,128,3,padding=1),nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv3d(128,128,3,padding=1),nn.ReLU())
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Sequential(self.conv3_1,self.conv3_2,self.pool3)

        self.score1_1 = nn.Sequential(nn.Conv3d(128,128,1),nn.ReLU())
        self.upscore1 = nn.Upsample(scale_factor=(2,2,2))
        self.score_pool1 = nn.Conv3d(64,128,1)

        self.score2_1 = nn.Sequential(nn.Conv3d(128,64,1),nn.ReLU())
        self.upscore2 = nn.Upsample(scale_factor=(2,2,2))
        self.score_pool2 = nn.Conv3d(32,64,1)

        self.score3 = nn.Sequential(nn.Conv3d(64,32,1),nn.ReLU())
        self.upscore3 = nn.Upsample(scale_factor=(2,2,2))

        self.upscore4 = nn.Conv3d(32,2,1)
        self.sigmoid = torch.nn.Sigmoid()
    # score1_1=Conv3D(128, 1, padding='same',activation='relu')(pool3)  
    # upscore1=UpSampling3D((2,2,2))(score1_1)

    # score_pool1=Conv3D(128, 1, padding='same')(pool2) 
    # score1=Add()([upscore1,score_pool1])

    # score2_1=Conv3D(64, 1, padding='same',activation='relu')(score1)    
    # upscore2=UpSampling3D((2,2,2))(score2_1)

    # score_pool2=Conv3D(64, 1, padding='same')(pool1) 
    # score2=Add()([upscore2,score_pool2])

    # score3_1=Conv3D(32, 1, padding='same',activation='relu')(score2)  
    # upscore3=UpSampling3D((2,2,2))(score3_1)
    # #upscore4=checker(upscore3,input_img)
    # upscore4=Conv3D(2, 1, padding='same')(upscore3)  
    # output1=Reshape((np.prod(img_shape), 2))(upscore4)
    # output=Activation('softmax')(output1)

    def forward(self,input):
        hidden1 = self.conv1(input)
        hidden2 = self.conv2(hidden1)
        hidden3 = self.conv3(hidden2)
        hidden_3 = self.upscore1(self.score1_1(hidden3))
        hidden_2 = hidden_3 + self.score_pool1(hidden2)
        hidden_1 = self.upscore2(self.score2_1(hidden_2)) + self.score_pool2(hidden1)
        hidden = self.score3(hidden_1)
        hidden = self.upscore3(hidden)
        hidden = self.upscore4(hidden)
        output = torch.reshape(hidden, (input.shape[0],2,np.prod(input.shape[1:])))
        output = torch.transpose(output,1,2)
        output=self.sigmoid(output)
        return output
        