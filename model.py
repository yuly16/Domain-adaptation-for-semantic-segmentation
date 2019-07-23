import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FCN_nd(torch.nn.Module):
    def __init__(self, aux_output = False):
        super(FCN_nd,self).__init__()
        self.aux_output = aux_output
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
        if self.aux_output == False:
            return output
        else:
            return output, hidden3

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv3d(128,128,2),nn.ReLU())
        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv1 = nn.Sequential(self.conv1_1,self.pool1)

        self.conv2_1 = nn.Sequential(nn.Conv3d(128,128,3,padding=1),nn.ReLU())
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Sequential(self.conv2_1,self.pool2)

        # self.conv3_1 = nn.Sequential(nn.Conv3d(64,128,3,padding=1),nn.ReLU())
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        # self.conv3 = nn.Sequential(self.conv3_1,self.pool3)

        # self.conv4_1 = nn.Sequential(nn.Conv3d(128,256,3,padding=1),nn.ReLU())
        # self.pool4 = nn.MaxPool3d((5, 5, 5), stride=(2, 2, 2))
        # self.conv4 = nn.Sequential(self.conv4_1,self.pool4)

        self.linear1 = nn.Sequential(nn.Linear(128,64),nn.LeakyReLU())
        self.linear2 = nn.Sequential(nn.Linear(64,2))
    def forward(self,input):
        hidden = self.conv1(input)
        hidden = self.conv2(hidden)
        # hidden = self.conv3(hidden)
        # hidden = self.conv4(hidden)
        hidden = torch.reshape(hidden,hidden.shape[0:2])
        hidden = self.linear1(hidden)
        output = self.linear2(hidden)
        return output



class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 1):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = 1
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size()[0]),int(total.size()[0]),int(total.size()[1]),int(total.size()[2]),int(total.size()[3]),int(total.size()[4]))
        total1 = total.unsqueeze(1).expand(int(total.size()[0]),int(total.size()[0]),int(total.size()[1]),int(total.size()[2]),int(total.size()[3]),int(total.size()[4]))
        L2_distance = ((total0-total1)**2).sum(5).sum(4).sum(3).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)/n_samples**2 for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss