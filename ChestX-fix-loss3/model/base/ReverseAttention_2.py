import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# class RA(nn.Module):
#     def __init__(self):
#         super(RA, self).__init__()
#         self.conv1 = nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=False)
#         self.conv2 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def getAttention(self,x):
#         attention = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
#         return attention
#
#     def forward(self,x,y):
#         y_norm2 = F.normalize(-y, p=2, dim=1, eps=1e-12)
#         avg = torch.mean(y_norm2,dim=1,keepdim=True)
#         max,_ = torch.max(y_norm2,dim=1,keepdim=True)
#         cated = torch.cat([avg,max],dim=1)
#         atten = self.getAttention(cated)
#         print("atten.shape:",atten.shape)
#         print("x.shape:", x.shape)
#         x = atten*x + x
#         return x


class RA(nn.Module):
    def __init__(self):
        super(RA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048,2048//32,False),
            nn.ReLU(),
            nn.Linear(2048//32, 2048, False),
            nn.Sigmoid()
        )
        # self.conv1 = nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=False)
        # self.conv2 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self,x,y):
        b,c,h,w = y.size()
        avg = self.avg_pool(y).view([b,c])
        fc = self.fc(avg).view([b,c,1,1])
        x = fc * x
        return x
