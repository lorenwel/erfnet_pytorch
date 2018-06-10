import numpy as np
import os

import torch

from PIL import Image

from numpy import genfromtxt, count_nonzero

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from transform import ToFloatLabel, ToLabel

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def is_self_supervised_image(filename):
    return filename.endswith("_img.bmp")

def is_self_supervised_label(filename, ext="npy", label_name="0"):
    return filename.endswith("_label_" + label_name + "." + ext)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def split_first_subname(filename, delim='_'):
    return filename.split(delim)[0]



class self_supervised_power(Dataset):

    def __init__(self, root, co_transform, subset='train', file_format="npy", label_name="0", subsample=1):
        self.images_root = os.path.join(root, subset)
        self.labels_root = os.path.join(root, subset)
        self.file_format = file_format
        self.label_name = label_name
        
        print ("Image root is: " + self.images_root)
        print ("Label root is: " + self.labels_root)
        print ("Load files with extension: " + self.file_format)
        print ("Load labels with name: " + self.label_name)
        if subsample > 1:
            print("Using every ", subsample, "th image")


        filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root), followlinks=True) for f in fn if is_self_supervised_label(f, ext=self.file_format, label_name=self.label_name)]
        filenamesGt.sort()

        filenamesGt = [val for ind, val in enumerate(filenamesGt) if ind % subsample == 0] # Subsample.
        # print("Filtering empty labels.")
        # if self.file_format == "npy":
        #     filenamesGt = [val for val in filenamesGt if (np.load(image_path_city(self.labels_root, val)) > 0.0).any()] # Remove label images without label.
        # elif self.file_format == "csv":
        #     filenamesGt = [val for val in filenamesGt if (genfromtxt(image_path_city(self.labels_root, val), delimiter=',', dtype="float32") > 0.0).any()] # Remove label images without label.
        # else:
        #     print("Unsupported file format " + self.file_format)

        self.filenamesGt = filenamesGt
        base_filenames = [split_first_subname(image_basename(val)) for val in self.filenamesGt]
        print ("Found " + str(len(self.filenamesGt)) + " labels.")

        filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root), followlinks=True) for f in fn if is_self_supervised_image(f)]
        self.filenames = [val for val in filenames if split_first_subname(image_basename(val)) in base_filenames]
        self.filenames.sort()
        print ("Found " + str(len(self.filenames)) + " images.")


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

        # label_array[label_array == 0] = -2
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
        label = ToLabel()(label)
        # Remove 0.0 image regions from transform padding
        # label[label == 0] = -1
        # label[label == -2] = 0


        # print ("Label is type " + label.type())
        # print ("Number of zero elements: " + str(torch.eq(label, 0).sum()))
        # print ("Number of negative elements: " + str(torch.lt(label, 0).sum()))
        # print ("Number of positive elements: " + str(torch.gt(label, 0).sum()))
        # print (label[torch.lt(label, 0)])

        # Sanitize labels. 
        if self.file_format == "csv":
            label[label != label] = -1

        n_nan = np.count_nonzero(np.isnan(label.numpy()))
        if n_nan > 0:
            print("File " + filenameGt + " produces nan " + str(n_nan))

        image1 = image1 - 0.5
        image2 = image2 - 0.5

        return image1, image2, label

    def __len__(self):
        return len(self.filenames)


