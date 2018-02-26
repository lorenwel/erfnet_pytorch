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


class Colorize:

    def __init__(self, min_val = 0.0, max_val = 1.0, remove_negative = False):
        self.min_val = min_val
        self.max_val = max_val
        self.factor = 255.0 / (max_val - min_val)
        self.remove_negative = remove_negative

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        # Pixels in interval.
        mask = torch.lt(gray_image, self.max_val) & torch.gt(gray_image, self.min_val)
        # Color pixels greater than max_val green.
        color_image[0][torch.ge(gray_image, self.max_val)] = 255
        # Color pixels less than min_val green.
        color_image[1][torch.le(gray_image, self.min_val)] = 255

        # TODO: This might be slow. 
        color_image[0][mask] = ((gray_image[mask] - self.min_val) * self.factor).byte()
        color_image[1][mask] = ((self.max_val - gray_image[mask]) * self.factor).byte()

        # Remove negative color.
        if self.remove_negative:
            mask = torch.lt(gray_image, 0.0)
            color_image[0][mask] = 0
            color_image[1][mask] = 0
            color_image[2][mask] = 0

        return color_image


# class Colorize:

#     def __init__(self, min_val = 0, max_val = 1, remove_negative = False):
#         self.min_val = min_val
#         self.max_val = max_val
#         self.factor = 255.0 / (max_val - min_val)
#         self.remove_negative = remove_negative

#     def __call__(self, gray_image):
#         size = gray_image.size()
#         color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

#         # Pixels in interval.
#         mask = torch.lt(gray_image, self.max_val) & torch.gt(gray_image, self.min_val)
#         # Color pixels greater than max_val green.
#         color_image[0][torch.gt(gray_image, self.max_val)] = 255
#         # Color pixels less than min_val green.
#         color_image[1][torch.lt(gray_image, self.min_val)] = 255

#         # TODO: This might be slow. 
#         color_image[0][mask] = ((gray_image[mask].float() - self.min_val) * self.factor).byte()
#         color_image[1][mask] = ((self.max_val - gray_image[mask].float()) * self.factor).byte()

#         # Remove negative color.
#         if self.remove_negative:
#             mask = torch.lt(gray_image, 0)
#             color_image[0][mask] = 255
#             color_image[1][mask] = 255
#             color_image[2][mask] = 255

#         return color_image
