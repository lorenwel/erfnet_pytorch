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

    def forward(self, input, predict=False):
        output1 = self.initial_block(input)

        output2 = output1
        for layer in self.block1:
            output2 = layer(output2)

        output3 = output2    
        for layer in self.block2:
            output3 = layer(output3)

        if predict:
            return self.output_conv(output3)

        return output1, output2, output3


class Decoder (nn.Module):
    def __init__(self, softmax_classes, late_dropout_prob):
        super().__init__()

        self.scalar_decoder_1 = DecoderBlock(128, 64)
        self.ladder_block_scalar_1 = LadderBlock(64)
        self.scalar_decoder_2 = DecoderBlock(128, 16)
        self.ladder_block_scalar_2 = LadderBlock(16)

        self.trav_decoder_1 = DecoderBlock(128, 64)
        self.ladder_block_trav_1 = LadderBlock(64)
        self.trav_decoder_2 = DecoderBlock(64, 16)
        self.ladder_block_trav_2 = LadderBlock(16)

        self.trav_output_conv = nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)

        self.scalar_output_conv = nn.ConvTranspose2d( 32, softmax_classes, 2, stride=2, padding=0, output_padding=0, bias=True)


    def forward(self, enc1, enc2, enc3):
        output_scalar = self.scalar_decoder_1(enc3)
        output_scalar = self.ladder_block_scalar_1(output_scalar, enc2)
        output_trav = self.trav_decoder_1(enc3)
        output_trav = self.ladder_block_trav_1(output_trav, enc2)

        output_scalar = self.scalar_decoder_2(output_scalar)
        output_scalar = self.ladder_block_scalar_2(output_scalar, enc1)
        output_trav = self.trav_decoder_2(output_trav)
        output_trav = self.ladder_block_trav_2(output_trav, enc1)

        output_scalar = self.scalar_output_conv(torch.cat((output_scalar, output_trav), dim=1))
        output_trav = self.trav_output_conv(output_trav)
        output_trav = torch.nn.functional.sigmoid(output_trav)

        return output_scalar, output_trav

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

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            enc1, enc2, enc3 = self.encoder(input)    #predict=False by default
            output_scalar, output_trav = self.decoder.forward(enc1, enc2, enc3)
            if self.softmax_classes > 0:
                return output_scalar, output_trav, self.class_power
            else:
                return output_scalar, output_trav
