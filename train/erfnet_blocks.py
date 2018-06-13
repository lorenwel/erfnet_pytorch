# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np



class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        # self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        # output = self.bn(output)
        return F.relu(output)
    


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        # self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        # output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        # output = self.bn2(output)

        # if (self.dropout.p != 0):
            # output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)



class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        # self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        # output = self.bn(output)
        return F.relu(output)



class SoftMaxConv (nn.Module):
    def __init__(self, in_channels, softmax_classes, late_dropout_prob):
        super().__init__()

        print("Added intermediate softmax layer with ", softmax_classes, " classes")
        self.convolution = nn.ConvTranspose2d( in_channels, softmax_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        print("Set late dropout prob to ", late_dropout_prob)
        self.dropout = torch.nn.Dropout2d(p=late_dropout_prob)

    def forward(self, input):
        output = self.convolution(input)
        output = self.dropout(output)
        output = torch.nn.functional.softmax(output, dim=1)

        return output



class DecoderBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(in_channels, out_channels))
        self.layers.append(non_bottleneck_1d(out_channels, 0, 1))
        self.layers.append(non_bottleneck_1d(out_channels, 0, 1))

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        return output



class LadderBlock(nn.Module):
    def __init__(self, noutput):
        super().__init__()
        self.layer = non_bottleneck_1d(2*noutput, 0.03, 1)
        self.conv = nn.Conv2d(2*noutput, noutput, 3, stride=1, padding=1, bias=True)
        # self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input, input_ladder):
        output = torch.cat((input, input_ladder), dim=1)
        output = self.layer(output)
        output = self.conv(output)
        # output = self.bn(output)
        return F.relu(output)
