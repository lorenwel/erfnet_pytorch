import numpy as np
import os

import torch

from PIL import Image

from numpy import genfromtxt, count_nonzero

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def is_self_supervised_image(filename):
    return filename.endswith("_img.bmp")

def is_self_supervised_label(filename, ext="npy"):
    return filename.endswith("_label." + ext)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)




class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)



class self_supervised_power(Dataset):

    def __init__(self, root, co_transform, subset='train', file_format="npy"):
        self.images_root = os.path.join(root, subset)
        self.labels_root = os.path.join(root, subset)
        self.file_format = file_format
        
        print ("Image root is: " + self.images_root)
        print ("Label root is: " + self.labels_root)
        print ("Load files with extension: " + self.file_format)

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root), followlinks=True) for f in fn if is_self_supervised_image(f)]
        self.filenames.sort()
        print ("Found " + str(len(self.filenames)) + " images.")

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root), followlinks=True) for f in fn if is_self_supervised_label(f, self.file_format)]
        self.filenamesGt.sort()
        print ("Found " + str(len(self.filenamesGt)) + " labels.")

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        label_array = None
        if self.file_format == "npy":
            label_array = np.load(image_path_city(self.labels_root, filenameGt))
        elif self.file_format == "csv":
            label_array = genfromtxt(image_path_city(self.labels_root, filenameGt), delimiter=',', dtype="float32")
        else:
            print("Unsupported file format " + self.file_format)

        label_img = Image.fromarray(label_array, 'F')

        # print ("Float " + filenameGt)
        # print (count_nonzero(label_array == -1.0))

        # print (list(label_img.getdata()).count(-1.0))

        # label_tensor = torch.from_numpy(np.array(label_img))
        # print ("Label tensor is type " + label_tensor.type())
        # print ("Number of zero elements: " + str(torch.eq(label_tensor, 0).sum()))
        # print ("Number of negative elements: " + str(torch.lt(label_tensor, 0).sum()))
        # print ("Number of positive elements: " + str(torch.gt(label_tensor, 0).sum()))

        # Image transformation which is also expected to return a tensor. 
        image, label = self.co_transform(image, label_img)


        # print ("Label is type " + label.type())
        # print ("Number of zero elements: " + str(torch.eq(label, 0).sum()))
        # print ("Number of negative elements: " + str(torch.lt(label, 0).sum()))
        # print ("Number of positive elements: " + str(torch.gt(label, 0).sum()))
        # print (label[torch.lt(label, 0)])

        # Sanitize labels. 
        if self.file_format == "csv":
            label[label != label] = -1.0

        n_nan = np.count_nonzero(np.isnan(label.numpy()))
        if n_nan > 0:
            print("File " + filenameGt + " produces nan " + str(n_nan))

        return image, label

    def __len__(self):
        return len(self.filenames)


