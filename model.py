import torch.nn as nn
from torch import Tensor
from typing import Optional
import MPNCOV
    
class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=3, stride=1, padding=1):
        super(Residual_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu= nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        if in_channels != out_channel:
            self.downsample = True
            
            self.conv_downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(),
                )
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
    
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity

        out = self.relu(out)
        out = self.maxpool(out)
        return out
    
class Model(nn.Module):
    def __init__(self, args, channel_num=20):
        super().__init__()
        self.num_blocks = args.num_blocks
        out_channel = 512
        self.block0 = nn.Sequential(
            nn.Conv1d(channel_num, out_channel, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            )
        
        self.blocks = nn.Sequential(*[Residual_block(out_channel, out_channel, kernel_size=3, 
                                                     stride=1, padding=1) for _ in range(self.num_blocks)])

        self.convcls = nn.Conv1d(out_channel, 2, kernel_size=1, stride=1)

        self.fc = nn.Linear(int(out_channel*(out_channel+1)/2), 2)

        

    def forward(self, x, x2=None):
        x1 = self.block0(x)
        for i in range(self.num_blocks):
            x1 = self.blocks[i](x1)

        conv_out = self.convcls(x1)
        
        x = x1.unsqueeze(2)
        cov = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(cov, 5)
        x = MPNCOV.TriuvecLayer(x).squeeze()
        output = self.fc(x)

        return x, conv_out, output