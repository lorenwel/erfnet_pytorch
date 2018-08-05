# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

from erfnet_blocks import *


class Encoder(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        print("Using self-supervised encoder.")
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1, use_dropout=use_dropout)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2, use_dropout=use_dropout))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4, use_dropout=use_dropout))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8, use_dropout=use_dropout))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16, use_dropout=use_dropout))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output

class Decoder (nn.Module):
    def __init__(self, softmax_classes, late_dropout_prob, use_dropout=False):
        super().__init__()

        self.scalar_decoder_1 = DecoderBlock(128, 64)
        self.scalar_decoder_2 = DecoderBlock(64, 16)

        if softmax_classes:
            self.scalar_output_conv = SoftMaxConv(16, softmax_classes, late_dropout_prob, use_dropout=use_dropout)
        else:
            self.scalar_output_conv = nn.ConvTranspose2d( 16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)


    def forward(self, input):
        output_scalar = self.scalar_decoder_1(input)

        output_scalar = self.scalar_decoder_2(output_scalar)

        output_scalar = self.scalar_output_conv(output_scalar)

        return output_scalar

#ERFNet
class Net(nn.Module):
    def __init__(self, encoder=None, 
                       softmax_classes=0, 
                       spread_class_power=False, 
                       fix_class_power=False, 
                       late_dropout_prob=0.1):  #use encoder to pass pretrained encoder
        super().__init__()

        self.softmax_classes = softmax_classes

        # Initialize class power consumption only when we have discretized into classes. 
        use_dropout=False
        if softmax_classes > 0:
            use_dropout=True
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
            self.encoder = Encoder(use_dropout=use_dropout)
        else:
            print("ERFnet set encoder from external")
            self.encoder = encoder
        self.decoder = Decoder(softmax_classes, late_dropout_prob, use_dropout=use_dropout)

    def forward(self, input):
        output_scalar = self.encoder(input)
        output_scalar = self.decoder.forward(output_scalar)
        if self.softmax_classes > 0:
            return output_scalar, self.class_power
        else:
            return output_scalar
