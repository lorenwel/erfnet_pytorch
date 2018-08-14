# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, ColorJitter
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import functional

from scipy import ndimage
from numpy.lib.stride_tricks import as_strided

from tensorboardX import SummaryWriter

from dataset import self_supervised_power
from transform import Relabel, ToLabel, Colorize, ColorizeMinMax, ColorizeWithProb, ColorizeClasses, ColorizeClassesProb, FloatToLongLabel, ToFloatLabel, getMaxProbValue
from visualize import Dashboard

from evaluation_functions import *

import importlib

from shutil import copyfile

color_transform_target = Colorize(1.0, 2.0, remove_negative=True, extend=True, white_val=1.0)  # min_val, max_val, remove negative
color_transform_output = Colorize(1.0, 2.0, remove_negative=False, extend=True, white_val=1.0)  # Automatic color based on tensor min/max val
# color_transform_output = ColorizeMinMax()  # Automatic color based on tensor min/max val
image_transform = ToPILImage()

class StepTracker():

    def setNZeros(self, n_zeros):
        self.prepend_str = ''
        for i in range(0,n_zeros):
            self.prepend_str += '0'

    def initPrependStr(self, n_images):
        self.prepend_str = ''
        # Find number of zeros.
        n_zeros = int(math.log10(n_images))
        self.setNZeros(n_zeros)
        self.max_n_zeros = n_zeros
        

    def __init__(self, n_images):
        self.initPrependStr(n_images)


    def getStepString(self, step):
        if step == 0:
            n_zeros = self.max_n_zeros
        else:
            n_zeros = int(math.log10(step))
        if self.max_n_zeros - n_zeros != len(self.prepend_str):
            self.setNZeros(self.max_n_zeros - n_zeros)

        return self.prepend_str + str(step)



def record_video(args, model_student, model_teacher, enc=False):
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    if args.classification:
        tensor_type = 'long'
    else:   # regression
        tensor_type = 'float'
    dataset_train = self_supervised_power(args.datadir, None, 'train', file_format="csv", label_name=args.label_name, tensor_type=tensor_type)
    dataset_val = self_supervised_power(args.datadir, None, 'val', file_format="csv", label_name=args.label_name, tensor_type=tensor_type)

    if args.force_n_classes > 0:
        if args.classification:
            apply_softmax = True
        else:   # regression
            apply_softmax = False
        color_transform_classes_prob = ColorizeClassesProb(args.force_n_classes)  # Automatic color based on max class probability
        color_transform_classes = ColorizeClasses(args.force_n_classes)  # Automatic color based on max class probability

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=1, shuffle=False)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=1, shuffle=False)

    step_tracker_train = StepTracker(len(dataset_train))
    step_tracker_val = StepTracker(len(dataset_val))

    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    # Check if all output dirs exist.
    output_dir_train = os.path.join(args.savedir, 'train/')
    output_dir_val = os.path.join(args.savedir, 'val/')
    if not os.path.exists(output_dir_train):
        os.makedirs(output_dir_train)
    if not os.path.exists(output_dir_val):
        os.makedirs(output_dir_val)

    print("----- TRAINING PROCESSING -----")
    model_student.eval()
    model_teacher.eval()

    tot_inf_dur = 0.0

    for step, (images1, images2, labels) in enumerate(loader):

        if args.cuda:
            images1 = images1.cuda()
            labels = labels.cuda()

        inputs1 = Variable(images1)
        targets = Variable(labels)

        start_time = time.time()
        if (args.force_n_classes) > 0:  # Captures both classification and regression with forced classes. 
            # Forced into discrete classes. 
            output_student_prob, output_student_power = model_student(inputs1)
        else:
            output_student = model_student(inputs1)
        end_time = time.time()
        tot_inf_dur += end_time - start_time

        # Figure out and compute tensor to visualize. 
        if args.force_n_classes > 0:
            max_prob, vis_output = getMaxProbValue(output_student_prob[0].cpu().data, output_student_power[0].cpu().data, apply_softmax)
            class_img = image_transform(color_transform_classes_prob(output_student_prob[0].cpu().data))
            prob_img = image_transform(max_prob[0].unsqueeze(0))
        else:
            vis_output = output_student[0].cpu().data

        regression_output = image_transform(color_transform_output(vis_output))

        if args.classification:
            target_output = image_transform(color_transform_classes(targets[0].cpu().data))
        else:   # regression
            target_output = image_transform(color_transform_target(targets[0].cpu().data))

        # Save images.
        prepend_str = step_tracker_train.getStepString(step)
        image_transform(images1.cpu().squeeze()).save(os.path.join(output_dir_train, 'input_' + prepend_str + '.jpg'))
        target_output.save(os.path.join(output_dir_train, 'target_' + prepend_str + '.jpg'))
        if args.regression:
            regression_output.save(os.path.join(output_dir_train, 'regression_' + prepend_str + '.jpg'))
        if args.force_n_classes:
            class_img.save(os.path.join(output_dir_train, 'class_' + prepend_str + '.jpg'))
            prob_img.save(os.path.join(output_dir_train, 'prob_' + prepend_str + '.jpg'))
            
        
    #Validate on 500 val images after each epoch of training
    print("----- VALIDATING PROCESSING -----")

    for step, (images1, images2, labels) in enumerate(loader_val):

        if args.cuda:
            images1 = images1.cuda()
            labels = labels.cuda()

        inputs1 = Variable(images1)
        targets = Variable(labels)

        start_time = time.time()
        if (args.force_n_classes) > 0:  # Captures both classification and regression with forced classes. 
            # Forced into discrete classes. 
            output_student_prob, output_student_power = model_student(inputs1)
        else:
            output_student = model_student(inputs1)
        end_time = time.time()
        tot_inf_dur += end_time - start_time

        # Figure out and compute tensor to visualize. 
        if args.force_n_classes > 0:
            max_prob, vis_output = getMaxProbValue(output_student_prob[0].cpu().data, output_student_power[0].cpu().data, apply_softmax)
            class_img = image_transform(color_transform_classes_prob(output_student_prob[0].cpu().data))
            prob_img = image_transform(max_prob[0].unsqueeze(0))
        else:
            vis_output = output_student[0].cpu().data

        regression_output = image_transform(color_transform_output(vis_output))

        if args.classification:
            target_output = image_transform(color_transform_classes(targets[0].cpu().data))
        else:   # regression
            target_output = image_transform(color_transform_target(targets[0].cpu().data))

        # Save images.
        prepend_str = step_tracker_val.getStepString(step)
        image_transform(images1.cpu().squeeze()).save(os.path.join(output_dir_val, 'input_' + prepend_str + '.jpg'))
        target_output.save(os.path.join(output_dir_val, 'target_' + prepend_str + '.jpg'))
        if args.regression:
            regression_output.save(os.path.join(output_dir_val, 'regression_' + prepend_str + '.jpg'))
        if args.force_n_classes:
            class_img.save(os.path.join(output_dir_val, 'class_' + prepend_str + '.jpg'))
            prob_img.save(os.path.join(output_dir_val, 'prob_' + prepend_str + '.jpg'))

        avg_time = tot_inf_dur / (len(dataset_train) + len(dataset_val))

    print ("Total inference time", tot_inf_dur, "s. Average per image", avg_time, "s.")
                   
       
    return


def main(args):
    # Check that we don't do classification and regression
    assert bool(args.classification) or bool(args.regression), "Did not specify --regression or --classification"
    assert bool(args.classification) != bool(args.regression), "Can only do regression or classification, not both. "
    if args.classification:
        assert args.force_n_classes > 0, "Did not specify number of classes for classification."

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
   
    model_student = model_file.Net( softmax_classes=args.force_n_classes)  #Add decoder to encoder
    model_teacher = model_file.Net( softmax_classes=args.force_n_classes)  #Add decoder to encoder

    if args.cuda:
        def make_cuda(model):
            return torch.nn.DataParallel(model).cuda()

        model_student = make_cuda(model_student)
        model_teacher = make_cuda(model_teacher)

    assert args.network, "Did not specify network to usel."
    pretrained = torch.load(args.network)['state_dict']
    model_student.load_state_dict(pretrained)
    model_teacher.load_state_dict(pretrained)
    print("Loaded pretrained model")
        
    #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model_student, model_teacher = record_video(args, model_student, model_teacher, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet_classification")

    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--network') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--regression', action='store_true', default=False) # Plot power consumption or whatever scalar value
    parser.add_argument('--classification', action='store_true', default=False) # Plot power consumption or whatever scalar value
    parser.add_argument('--label-name', type=str, default="class")
    parser.add_argument('--force-n-classes', type=int, default=0)

    main(parser.parse_args())
