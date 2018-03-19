import numpy as np
import torch

from PIL import Image

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class FloatToLongLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)*1000000).long().unsqueeze(0)


class ToFloatLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).unsqueeze(0)



def getColorImageFromMinMax(gray_image, min_val, max_val, factor_val = None, extend=False, white_val = 1.0):
    if factor_val is None:
        factor_val = white_val / (max_val - min_val)

    size = gray_image.size()
    color_image = torch.FloatTensor(3, size[1], size[2]).fill_(0.0)
    # Pixels in interval.
    mask = torch.lt(gray_image, max_val) & torch.gt(gray_image, min_val)
    # Color pixels greater than max_val green.
    mask_ge_max = torch.ge(gray_image, max_val)
    color_image[0][mask_ge_max] = white_val
    # Color pixels less than min_val green.
    mask_le_min = torch.le(gray_image, min_val)
    color_image[1][mask_le_min] = white_val
    # Compute pixel values in interval. 
    # TODO: This might be slow. 
    color_image[0][mask] = ((gray_image[mask] - min_val) * factor_val)
    color_image[1][mask] = ((max_val - gray_image[mask]) * factor_val)

    # Extend past min max values to avoid saturation. 
    if extend:
        # Extend max value
        new_max = 2*max_val - min_val
        mask_ext_max = mask_ge_max & torch.lt(gray_image, new_max)
        color_image[0][mask_ext_max] = ((max_val - gray_image[mask_ext_max]) * factor_val)
        # Clip above new max.
        color_image[0][torch.ge(gray_image, new_max)] = 0
        # Extend min value
        new_min = 2*min_val - max_val
        mask_ext_min = mask_le_min & torch.gt(gray_image, new_min)
        color_image[0][mask_le_min] = ((min_val - gray_image[mask_le_min]) * factor_val)
        color_image[2][mask_le_min] = ((min_val - gray_image[mask_le_min]) * factor_val)
        # Clip below new min
        color_image[0][torch.le(gray_image, new_min)] = white_val
        color_image[2][torch.le(gray_image, new_min)] = white_val

    return color_image




class ColorizeMinMax:

    def __call__(self, gray_image):
        min_val = gray_image.min()
        max_val = gray_image.max()

        return getColorImageFromMinMax(gray_image, min_val, max_val)




class Colorize:

    def __init__(self, min_val = 0.0, max_val = 1.0, remove_negative = False, extend=False, white_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.white_val = white_val
        self.factor = white_val / (max_val - min_val)
        self.remove_negative = remove_negative
        self.extend = extend

    def __call__(self, gray_image):
        color_image = getColorImageFromMinMax(gray_image, self.min_val, self.max_val, self.factor, self.extend, self.white_val)

        # Remove negative color.
        if self.remove_negative:
            mask = torch.lt(gray_image, 0.0)
            color_image[0][mask] = 0
            color_image[1][mask] = 0
            color_image[2][mask] = 0

        return color_image




# tensor is assumed to be shape [1, C, 1, 1]
# prob should be shape [..., N x C x H x W]
# returns tensor with max probability [..., N x H x W] and value of class with max probability [..., H x W]
def getMaxProbValue(prob, tensor):
    channel_dim = prob.dim() - 3
    max_prob, max_ind = prob.max(dim=channel_dim, keepdim=True)
    return max_prob, (tensor.expand(prob.size())).gather(channel_dim, max_ind)




class ColorizeWithProb:

    def __init__(self, min_val = 0.0, max_val = 1.0, remove_negative = False, extend=False, white_val=1.0):
        self.func = Colorize(min_val, max_val, remove_negative, extend, white_val)

    def __call__(self, prob, power):
        _, val = getMaxProbValue(prob, power)
        return self.func(val)




class ColorizeClasses:

    def __init__(self, n=22):
        self.cmap = colormap(256)
        # self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, prob):
        dump, gray_image = prob.max(dim=0, keepdim=True)
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
