import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BatchChannelPreturbeAttention(nn.Module):
    def __init__(self):
        super(BatchChannelPreturbeAttention, self).__init__()
        self.bs = 512
        self.hidden_bs = 32
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(self.bs,self.hidden_bs,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_bs,self.bs,1,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        B,C,H,W = x.shape
        G = C//self.bs
        result = []
        for i in range(G):
            x_crop = x[:, (0 + i) * self.bs:(1 + i) * self.bs, :, :]
            avg = self.avg_pool(x_crop).view([B,self.bs,1,1])
            fc = self.fc(avg).view([B,self.bs])
            result.append(fc)
        result = 1 - torch.cat(result,dim=1)
        result = result.view(B, C, 1, 1)
        # print(result)
        return x*result

# x1 = torch.ones(4, 512,50,50)
# x2 = torch.ones(4, 1024,25,25)
# x3 = torch.ones(4, 2048, 13, 13)
# BCPA = BatchChannelPreturbeAttention()
# x = BCPA(x2)
# print(x)
# print(x.shape)