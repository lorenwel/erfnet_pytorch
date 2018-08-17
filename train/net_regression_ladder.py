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

        self.block1 = nn.ModuleList()
        self.block2 = nn.ModuleList()

        self.block1.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.block1.append(non_bottleneck_1d(64, 0.03, 1, use_dropout=use_dropout)) 

        self.block2.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.block2.append(non_bottleneck_1d(128, 0.3, 2, use_dropout=use_dropout))
            self.block2.append(non_bottleneck_1d(128, 0.3, 4, use_dropout=use_dropout))
            self.block2.append(non_bottleneck_1d(128, 0.3, 8, use_dropout=use_dropout))
            self.block2.append(non_bottleneck_1d(128, 0.3, 16, use_dropout=use_dropout))

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

class Decoder (nn.Module):
    def __init__(self, softmax_classes, late_dropout_prob, likelihood_loss, use_dropout=False):
        super().__init__()

        self.scalar_decoder_1 = DecoderBlock(128, 64, use_dropout=use_dropout)
        self.ladder_block_1 = LadderBlock(64, use_dropout=use_dropout)
        self.scalar_decoder_2 = DecoderBlock(64, 16, use_dropout=use_dropout)
        self.ladder_block_2 = LadderBlock(16, use_dropout=use_dropout)

        if softmax_classes:
            self.scalar_output_conv = SoftMaxConv(16, softmax_classes, late_dropout_prob, use_dropout=use_dropout)
        else:
            if likelihood_loss:
                # We want two output channels for mean and std, respectively. 
                self.scalar_output_conv = nn.ConvTranspose2d( 16, 2, 2, stride=2, padding=0, output_padding=0, bias=True)
            else:
                self.scalar_output_conv = nn.ConvTranspose2d( 16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)


    def forward(self, enc1, enc2, enc3):
        output_scalar = self.scalar_decoder_1(enc3)
        output_scalar = self.ladder_block_1(output_scalar, enc2)
        output_scalar = self.scalar_decoder_2(output_scalar)
        output_scalar = self.ladder_block_2(output_scalar, enc1)
        output_scalar = self.scalar_output_conv(output_scalar)

        return output_scalar

#ERFNet
class Net(nn.Module):
    def __init__(self, encoder=None, 
                       softmax_classes=0, 
                       likelihood_loss=False,
                       spread_class_power=False, 
                       late_dropout_prob=0.0):  #use encoder to pass pretrained encoder
        super().__init__()

        self.softmax_classes = softmax_classes
        self.likelihood_loss = likelihood_loss

        # Initialize class power consumption only when we have discretized into classes. 
        use_dropout = False
        if softmax_classes > 0:
            use_dropout = True
            if spread_class_power:
                print("Spreading initial class power estimates to:")
                init_vals = np.ndarray((1,softmax_classes,1,1), dtype="float32")
                init_vals[0,:,0,0] = np.linspace(0.0, 2.0, softmax_classes)
                print(init_vals[0,:,0,0])
                init_tensor = torch.from_numpy(init_vals)
            else:
                init_tensor = torch.ones(1,softmax_classes,1,1)

            self.class_power = torch.nn.Parameter(init_tensor)
            self.class_power_var = torch.nn.Parameter(torch.ones(1,softmax_classes,1,1))


        if (encoder == None):
            self.encoder = Encoder(use_dropout=use_dropout)
        else:
            print("ERFnet set encoder from external")
            self.encoder = encoder
        self.decoder = Decoder(softmax_classes, late_dropout_prob, likelihood_loss, use_dropout=use_dropout)

    def forward(self, input):
        enc1, enc2, enc3 = self.encoder(input)
        output_scalar = self.decoder.forward(enc1, enc2, enc3)
        if self.softmax_classes > 0:
            if self.likelihood_loss:
                return output_scalar, self.class_power, self.class_power_var
            else:
                return output_scalar, self.class_power
        else:
            return output_scalar
