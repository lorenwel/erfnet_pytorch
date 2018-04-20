import numpy as np
import os

import torch

from PIL import Image

from numpy import genfromtxt, count_nonzero

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from transform import ToFloatLabel

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
    return filename.endswith("_label_0." + ext)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def split_first_subname(filename, delim='_'):
    return filename.split(delim)[0]



class self_supervised_power(Dataset):

    def __init__(self, root, co_transform, subset='train', file_format="npy", subsample=1):
        self.images_root = os.path.join(root, subset)
        self.labels_root = os.path.join(root, subset)
        self.file_format = file_format
        
        print ("Image root is: " + self.images_root)
        print ("Label root is: " + self.labels_root)
        print ("Load files with extension: " + self.file_format)
        if subsample > 1:
            print("Using every ", subsample, "th image")

        filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root), followlinks=True) for f in fn if is_self_supervised_image(f)]
        filenames.sort()
        self.filenames = [val for ind, val in enumerate(filenames) if ind % subsample == 0]
        base_filenames = [split_first_subname(val) for val in self.filenames]
        print ("Found " + str(len(self.filenames)) + " images.")

        filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root), followlinks=True) for f in fn if is_self_supervised_label(f, self.file_format)]
        self.filenamesGt = [val for val in filenamesGt if split_first_subname(val) in base_filenames]
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

        label = Image.fromarray(label_array, 'F')

        # print ("Float " + filenameGt)
        # print (count_nonzero(label_array == -1.0))

        # print (list(label_img.getdata()).count(-1.0))

        # label_tensor = torch.from_numpy(np.array(label_img))
        # print ("Label tensor is type " + label_tensor.type())
        # print ("Number of zero elements: " + str(torch.eq(label_tensor, 0).sum()))
        # print ("Number of negative elements: " + str(torch.lt(label_tensor, 0).sum()))
        # print ("Number of positive elements: " + str(torch.gt(label_tensor, 0).sum()))

        # Image transformation which is also expected to return a tensor. 
        if self.co_transform is not None:
            image1, image2, label = self.co_transform(image, label)
        else:
            image1 = image
            image2 = image
        # Convert to tensor
        image1 = ToTensor()(image1)
        image2 = ToTensor()(image2)
        label = ToFloatLabel()(label)
        # Remove 0.0 image regions from transform padding
        label[label == 0.0] = -1.0


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

        return image1, image2, label

    def __len__(self):
        return len(self.filenames)


