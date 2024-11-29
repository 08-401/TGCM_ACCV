import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class RA(nn.Module):
    def __init__(self,in_channel,hidden_channel=64):
        super(RA, self).__init__()
        # for logit_mask
        self.conv1x1_0_a_1 = nn.Conv2d(1, out_channels=8, kernel_size=5,stride=3,padding=0)
        self.conv1x1_0_a_2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=3, padding=0)
        self.conv1x1_0_a_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=3, padding=0)
        # for support_mask
        self.conv1x1_0_b = nn.Conv2d(in_channel, hidden_channel, 1)  # for inchannel
        # self.conv1x1_1_a = nn.Conv2d(in_channel, hidden_channel, 1)
        self.conv1x1_2 = nn.Conv2d(hidden_channel, in_channel, 1)
        self.conv1 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel

    def forward(self, x, y, beta=0.75,ismask=False):
        x_old = x.clone()
        y_old = y.clone()
        # ismask=True and y is 4*2*400*400
        if ismask:
            # 4, 2, 400, 400 -> 4, 1, 400, 400 = logit_mask
            if y.shape[1] == 2:
                support_mask_old = y.clone()
                support_mask_old = support_mask_old.argmax(dim=1, keepdim=True)
                pre_mask = support_mask_old.bool()
                y = y.softmax(dim=1)# (pre_mask+1)防止 0 的出现
                mask = torch.masked_fill(input=y[:, 1, :, :].unsqueeze(1).float(), mask=pre_mask, value=0) + 1
            else:
                # 4, 400, 400 ->4, 1, 400, 400
                y = y.reshape(y.shape[0], 1, y.shape[-2], y.shape[-1])
                mask = y[:, 0, :, :].unsqueeze(1).float()
            a = torch.sigmoid(-mask)
            a = self.conv1x1_0_a_3(self.conv1x1_0_a_2(self.conv1x1_0_a_1(a)))
        else:
            a = torch.sigmoid(-y_old)
            a = self.conv1x1_0_b(a)
        x = self.conv1x1_0_b(x)
        x = a.mul(x)  # x = sigmoid(-y)*x
        x = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
        # 升维
        x = self.conv1x1_2(x)
        x = x + x_old * beta

        return x