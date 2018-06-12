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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using self-supervised encoder.")
        self.initial_block = DownsamplerBlock(3,16)

        self.block1 = nn.ModuleList()
        self.block2 = nn.ModuleList()

        self.block1.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.block1.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.block2.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.block2.append(non_bottleneck_1d(128, 0.3, 2))
            self.block2.append(non_bottleneck_1d(128, 0.3, 4))
            self.block2.append(non_bottleneck_1d(128, 0.3, 8))
            self.block2.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output1 = self.initial_block(input)

        output2 = output1
        for layer in self.block1:
            output2 = layer(output2)

        output3 = output2    
        for layer in self.block2:
            output3 = layer(output3)

        return output1, output2, output3


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        # self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        # output = self.bn(output)
        return F.relu(output)

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

class SoftMaxConv (nn.Module):
    def __init__(self, softmax_classes, late_dropout_prob):
        super().__init__()

        print("Added intermediate softmax layer with ", softmax_classes, " classes")
        self.convolution = nn.ConvTranspose2d( 32, softmax_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        # print("Set late dropout prob to ", late_dropout_prob)
        # self.dropout = torch.nn.Dropout2d(p=late_dropout_prob)

    def forward(self, input):
        output = self.convolution(input)
        # output = self.dropout(output)
        output = torch.nn.functional.softmax(output, dim=1)

        return output

class DecoderBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(in_channels, out_channels))
        self.layers.append(non_bottleneck_1d(out_channels, 0, 1))
        self.layers.append(non_bottleneck_1d(out_channels, 0, 1))

        self.ladder_block = LadderBlock(out_channels)

    def forward(self, input, input_ladder):
        output = input

        for layer in self.layers:
            output = layer(output)

        return self.ladder_block(output, input_ladder)

class Decoder (nn.Module):
    def __init__(self, softmax_classes, late_dropout_prob):
        super().__init__()

        self.scalar_decoder_1 = DecoderBlock(128, 64)
        self.scalar_decoder_2 = DecoderBlock(64, 16)

        self.scalar_output_conv = nn.ConvTranspose2d( 16, softmax_classes, 2, stride=2, padding=0, output_padding=0, bias=True)


    def forward(self, enc1, enc2, enc3):
        output_scalar = self.scalar_decoder_1(enc3, enc2)

        output_scalar = self.scalar_decoder_2(output_scalar, enc1)

        output_scalar = self.scalar_output_conv(output_scalar)

        return output_scalar

#ERFNet
class Net(nn.Module):
    def __init__(self, encoder=None, 
                       softmax_classes=0, 
                       spread_class_power=False, 
                       fix_class_power=False, 
                       late_dropout_prob=0.3):  #use encoder to pass pretrained encoder
        super().__init__()

        self.softmax_classes = softmax_classes

        # Initialize class power consumption only when we have discretized into classes. 
        if softmax_classes > 0:
            if spread_class_power:
                print("Spreading initial class power estimates to:")
                init_vals = np.ndarray((1,softmax_classes,1,1), dtype="float32")
                init_vals[0,:,0,0] = np.linspace(0.7, 2.0, softmax_classes)
                print(init_vals[0,:,0,0])
                init_tensor = torch.from_numpy(init_vals)
            else:
                init_tensor = torch.ones(1,softmax_classes,1,1)

            if fix_class_power:
                self.class_power = torch.autograd.Variable(init_tensor).cuda()
                print("Fixing class power")
            else:
                self.class_power = torch.nn.Parameter(init_tensor)


        if (encoder == None):
            self.encoder = Encoder()
        else:
            self.encoder = encoder
        self.decoder = Decoder(softmax_classes, late_dropout_prob)

    def forward(self, input):
        enc1, enc2, enc3 = self.encoder(input)
        output_scalar = self.decoder.forward(enc1, enc2, enc3)
        if self.softmax_classes > 0:
            return output_scalar, self.class_power
        else:
            return output_scalar
