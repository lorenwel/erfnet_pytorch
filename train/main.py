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
from transform import *
from visualize import Dashboard

from evaluation_functions import *

import importlib

from shutil import copyfile

NUM_CHANNELS = 3
# NUM_CLASSES = 20 #pascal=22, cityscapes=20
NUM_HISTOGRAMS = 5
NUM_IMG_PER_EPOCH = 10
# Optimizer params.
LEARNING_RATE=5e-5
BETAS=(0.9, 0.999)
OPT_EPS=1e-04
WEIGHT_DECAY=1e-6

DISCOUNT_RATE_START=0.01
DISCOUNT_RATE=0.001
MAX_CONSISTENCY_EPOCH=10
DISCOUNT_RATE_START_EPOCH=5

color_transform_target = Colorize(0.0, 2.0, remove_negative=True, extend=True, white_val=1.0)  # min_val, max_val, remove negative
color_transform_output = Colorize(0.0, 2.0, remove_negative=False, extend=True, white_val=1.0)  # Automatic color based on tensor min/max val
# color_transform_output = ColorizeMinMax()  # Automatic color based on tensor min/max val
image_transform = ToPILImage()



#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height

        self.rotation_angle = 5.0
        self.affine_angle = 5.0
        self.shear_angle = 5.0
        self.crop_ratio = 0.7

        self.color_augmentation = ColorJitter(brightness=0.4,
                                              contrast=0.4,
                                              saturation=0.4,
                                              hue=0.06)
        pass

    def transform_augmentation(self, image, flip, rotation, affine_angle, affine_shear):
        # Horizontal flip
        if flip:
            image = functional.hflip(image)
        # Rotate image. 
        image = functional.rotate(image, rotation)
        # Affine transformation
        # image = functional.affine(image, affine_angle, (0,0), affine_shear)   # Affine not available in this pytorch version

        return image


    def __call__(self, input, target):
        # Crop needs to happen here to avoid cropping out all footsteps
        while True:
            # Generate parameters for image transforms
            rotation_angle = random.uniform(-self.rotation_angle, self.rotation_angle)  
            tan_ang = abs(math.tan(math.radians(rotation_angle)))
            y_bound_pix = tan_ang*320
            x_bound_pix = tan_ang*240
            crop_val = random.uniform(self.crop_ratio, 1.0-(y_bound_pix/240))
            affine_angle = random.uniform(-self.affine_angle, self.affine_angle)
            shear_angle = random.uniform(-self.shear_angle, self.shear_angle)
            flip = random.random() < 0.5
            img_size = np.array([640, 480]) * crop_val
            hor_pos = int(random.uniform(tan_ang, 1-tan_ang) * (640 - img_size[0]))
            # Do other transform.
            input_crop = self.transform_augmentation(input, flip, rotation_angle, affine_angle, shear_angle)
            target_crop = self.transform_augmentation(target, flip, rotation_angle, affine_angle, shear_angle)
            # Do crop
            crop_tuple = (hor_pos, 480 - img_size[1]-y_bound_pix, hor_pos + img_size[0], 480-y_bound_pix)
            input_crop = input_crop.crop(crop_tuple)
            target_crop = target_crop.crop(crop_tuple)
            target_test = np.array(target_crop, dtype="float32")
            # Make this condition proper for regression where we want > 0.0. Or fix border issues?!
            if np.any(target_test != -1):
                input = input_crop.resize((640,480))
                target = target_crop.resize((640,480))
                break
            elif img_size[0] == 640:
                print ("Encountered image with labels crictically close to border")

        # Color transformation
        input1 = self.color_augmentation(input)
        input2 = self.color_augmentation(input)

        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        # print (target)
        # target = Relabel(255, 19)(target)

        return input1, input2, target



def copyWeightsToModelNoGrad(model_source, model_target):
    for source, target in zip(model_source.parameters(), model_target.parameters()):
        target = source.clone() # Using detach here doesn't work.
        # Setting requires_grad False here doesn't work.
    for target in model_target.parameters():
        target.requires_grad=False

def copyWeightsToModelWithDiscount(model_source, model_target, discount_factor):
    for source, target in zip(model_source.parameters(), model_target.parameters()):
        target.data = target.data * (1-discount_factor) + discount_factor * source.data


best_acc = -1000000

def train(args, model_student, model_teacher, enc=False):
    global best_acc

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    # Set data loading variables
    co_transform = MyCoTransform(enc, augment=True, height=480)
    co_transform_val = MyCoTransform(enc, augment=False, height=480)
    if args.classification:
        tensor_type = 'long'
    else:   # regression
        tensor_type = 'float'
    dataset_train = self_supervised_power(args.datadir, co_transform, 'train', file_format="csv", label_name=args.label_name, subsample=args.subsample, tensor_type=tensor_type)
    dataset_val = self_supervised_power(args.datadir, None, 'val', file_format="csv", label_name=args.label_name, subsample=args.subsample,tensor_type=tensor_type)

    if args.force_n_classes > 0:
        if args.classification:
            apply_softmax = True
        else:   # regression
            apply_softmax = False
        color_transform_classes_prob = ColorizeClassesProb(args.force_n_classes)  # Automatic color based on max class probability
        color_transform_classes = ColorizeClasses(args.force_n_classes)  # Automatic color based on max class probability

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # Set Loss functions
    if args.regression:
        if args.force_n_classes > 0:
            if args.likelihood_loss:
                criterion = LogLikelihoodLossClassProbMasked()
            else:
                criterion = L1LossClassProbMasked() # L1 loss weighted with class prob with averaging over mini-batch
        else:
            if args.likelihood_loss:
                criterion = LogLikelihoodLossMasked(opt_eps=OPT_EPS)
            else:
                criterion = L1LossMasked()     
    elif args.classification:
        criterion = CrossEntropyLoss2d()
        criterion_acc = ClassificationAccuracy()
        criterion_mean_acc = MeanAccuracy(args.force_n_classes)

    criterion_val = criterion
    criterion_consistency = MSELossWeighted(args.consistency_weight)

    savedir = f'../save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model_student))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893
    # Ignore the class_power parameter if we fix class power.
    opt_params = [param for key, param in model_student.named_parameters() if not args.fix_class_power or 'class_power' not in key]
    optimizer = Adam(opt_params, LEARNING_RATE, BETAS,  eps=OPT_EPS, weight_decay=WEIGHT_DECAY)

    start_epoch = 1

    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model_student.load_state_dict(checkpoint['state_dict'])
        model_teacher.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    if args.mean_teacher:
        # Initialize teacher with same weights as student. 
        copyWeightsToModelNoGrad(model_student, model_teacher)

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2
    
    if args.visualize:
        writer = SummaryWriter()
        log_base_dir = writer.file_writer.get_logdir() + "/"
        print("Saving tensorboard log to: " + log_base_dir)
        total_steps_train = 0
        total_steps_val = 0
        # Figure out histogram plot indices.
        steps_hist = int(len(loader_val)/NUM_HISTOGRAMS)
        steps_img_train = int(len(loader)/(NUM_IMG_PER_EPOCH-1))
        if steps_img_train == 0:
            steps_img_train = 1
        steps_img_val = int(len(loader_val)/(NUM_IMG_PER_EPOCH-1))
        if steps_img_val == 0:
            steps_img_val = 1
        hist_bins = np.arange(-0.5, args.force_n_classes+0.5, 1.0)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        if args.mean_teacher:
            if epoch < MAX_CONSISTENCY_EPOCH:
                cur_consistency_weight = epoch / MAX_CONSISTENCY_EPOCH
            else:
                cur_consistency_weight = 1.0

        scheduler.step(epoch)

        average_loss_student_val = 0
        average_loss_teacher_val = 0

        epoch_loss_student = []
        epoch_loss_teacher = []
        if args.classification:
            epoch_acc_student = []
            epoch_mean_acc_student = []
            epoch_class_acc_student = np.empty((len(dataset_train), args.force_n_classes))
            if args.mean_teacher:
                epoch_acc_teacher = []
                epoch_mean_acc_teacher = []
                epoch_class_acc_teacher = np.empty((len(dataset_train), args.force_n_classes))
        if args.mean_teacher:
            epoch_loss_consistency = []
        time_train = []
        time_load = []
        time_iter = [0.0]

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model_student.train()
        model_teacher.train()

        start_time = time.time()

        for step, (images1, images2, labels) in enumerate(loader):

            time_load.append(time.time() - start_time)
            start_time = time.time()

            if args.cuda:
                images1 = images1.cuda()
                if args.mean_teacher:
                    images2 = images2.cuda()
                labels = labels.cuda()

            inputs1 = Variable(images1)
            if args.mean_teacher:
                inputs2 = Variable(images2)
            targets = Variable(labels)

            if (args.force_n_classes) > 0:  # Captures both classification and regression with forced classes. 
                # Forced into discrete classes. 
                if args.likelihood_loss:
                    output_student_prob, output_student_power, output_student_power_var = model_student(inputs1)
                    if args.mean_teacher:
                        output_teacher_prob, output_teacher_power, output_teacher_power_var = model_teacher(inputs2)
                else:
                    output_student_prob, output_student_power = model_student(inputs1)
                    if args.mean_teacher:
                        output_teacher_prob, output_teacher_power = model_teacher(inputs2)

                optimizer.zero_grad()

                if args.classification:
                    loss_student_pred = criterion(output_student_prob, targets)
                    if args.mean_teacher:
                        loss_teacher_pred = criterion(output_teacher_prob, targets)
                else:   # regression
                    if args.likelihood_loss:
                        loss_student_pred = criterion(output_student_prob, output_student_power, output_student_power_var, targets)
                        if args.mean_teacher:
                            loss_teacher_pred = criterion(output_teacher_prob, output_teacher_power, output_teacher_power_var, targets)
                    else:
                        loss_student_pred = criterion(output_student_prob, output_student_power, targets)
                        if args.mean_teacher:
                            loss_teacher_pred = criterion(output_teacher_prob, output_teacher_power, targets)

                if args.mean_teacher:
                    loss_consistency = criterion_consistency(output_student_prob, output_teacher_prob, cur_consistency_weight)
                if args.classification:
                    acc_student = criterion_acc(output_student_prob, targets)
                    mean_acc_student, epoch_class_acc_student[step,:] = criterion_mean_acc(output_student_prob, targets)
                    if args.mean_teacher:
                        acc_teacher = criterion_acc(output_teacher_prob, targets)
                        mean_acc_teacher, epoch_class_acc_teacher[step,:] = criterion_mean_acc(output_teacher_prob, targets)
            else:
                output_student = model_student(inputs1)
                if args.mean_teacher:
                    output_teacher = model_teacher(inputs2)
                    # Compute consistency loss before we split output in likelihood loss case.
                    loss_consistency = criterion_consistency(output_student, output_teacher, cur_consistency_weight)
                if args.likelihood_loss:
                    # Split output into mean and variance.
                    output_var_student = output_student[:,1,:,:].pow(2)
                    output_student = output_student[:,0,:,:]
                    if args.mean_teacher:
                        output_var_teacher = output_teacher[:,1,:,:].pow(2)
                        output_teacher = output_teacher[:,0,:,:]

                optimizer.zero_grad()

                if args.likelihood_loss:
                    loss_student_pred = criterion(output_student, output_var_student, targets)
                    if args.mean_teacher:
                        loss_teacher_pred = criterion(output_teacher, output_var_teacher, targets)
                else:
                    loss_student_pred = criterion(output_student, targets)
                    if args.mean_teacher:
                        loss_teacher_pred = criterion(output_teacher, targets)
                        


            # Do backward pass.
            if epoch>start_epoch and args.mean_teacher:
                loss_consistency.backward(retain_graph=True)

            loss_student_pred.backward()

            optimizer.step()

            # Update weights of teacher model
            if args.mean_teacher:
                # Copy more from student for the first DISCOUNT_RATE_START_EPOCH epochs.
                if epoch < DISCOUNT_RATE_START_EPOCH:
                    cur_discount_rate = DISCOUNT_RATE_START
                else:
                    cur_discount_rate = DISCOUNT_RATE
                copyWeightsToModelWithDiscount(model_student, model_teacher, cur_discount_rate)

            epoch_loss_student.append(loss_student_pred.data.item())
            if args.mean_teacher:
                epoch_loss_teacher.append(loss_teacher_pred.data.item())
                epoch_loss_consistency.append(loss_consistency.data.item())
            if (args.force_n_classes) > 0 and args.classification:
                epoch_acc_student.append(acc_student.data.item())
                epoch_mean_acc_student.append(mean_acc_student.data.item())
                if args.mean_teacher:
                    epoch_acc_teacher.append(acc_teacher.data.item())
                    epoch_mean_acc_teacher.append(mean_acc_teacher.data.item())
            time_train.append(time.time() - start_time)

            if args.visualize and step % steps_img_train == 0:
                step_vis_no = total_steps_train + len(epoch_loss_student)

                # Figure out and compute tensor to visualize. 
                start_time_plot = time.time()
                if args.force_n_classes > 0:
                    
                    max_prob, vis_output = getMaxProbValue(output_student_prob[0].cpu().data, output_student_power[0].cpu().data, apply_softmax)
                    writer.add_image("train/2_classes_student", color_transform_classes_prob(output_student_prob[0].cpu().data), step_vis_no)
                    writer.add_image("train/3_max_class_probability_student", max_prob[0], step_vis_no)
                    if args.mean_teacher:
                        max_prob_teacher, vis_output_teacher = getMaxProbValue(output_teacher_prob[0].cpu().data, output_teacher_power[0].cpu().data, apply_softmax)
                        writer.add_image("train/2_classes_teacher", color_transform_classes_prob(output_teacher_prob[0].cpu().data), step_vis_no)
                        writer.add_image("train/3_max_class_probability_teacher", max_prob_teacher[0], step_vis_no)
                    if args.regression:
                        # Compute weighted power consumption
                        sum_dim = output_student_prob.dim()-3
                        weighted_sum_output = (output_student_prob * output_student_power).sum(dim=sum_dim, keepdim=True)
                        writer.add_image("train/4_weighted_output_student", color_transform_output(weighted_sum_output[0].cpu().data), step_vis_no)
                else:
                    vis_output = output_student[0].cpu().data
                    if args.mean_teacher:
                        vis_output_teacher = output_teacher[0].cpu().data
                    if args.likelihood_loss:
                        writer.add_image("train/std_dev_student", output_var_student[0].sqrt().cpu().data, step_vis_no)
                        if args.mean_teacher:
                            writer.add_image("train/std_dev_teacher", output_var_teacher[0].sqrt().cpu().data, step_vis_no)

                image1 = inputs1[0].cpu().data
                writer.add_image("train/1_input_student", image1, step_vis_no)
                if args.mean_teacher:
                    image2 = inputs2[0].cpu().data
                    writer.add_image("train/1_input_teacher", image2, step_vis_no)
                if args.regression:
                    writer.add_image("train/5_output_student", color_transform_output(vis_output), step_vis_no)
                    if args.mean_teacher:
                        writer.add_image("train/5_output_teacher", color_transform_output(vis_output_teacher), step_vis_no)
                if args.classification:
                    writer.add_image("train/6_target", color_transform_classes(targets[0].cpu().data), step_vis_no)
                else:   # regression
                    writer.add_image("train/6_target", color_transform_target(targets[0].cpu().data), step_vis_no)


                # Visualize graph.
                # writer.add_graph(model_student, inputs1)    # This is broken when using multi-GPU

                print ("Time for visualization: ", time.time() - start_time_plot)
                
        # Add scalar tensorboard output
        len_epoch_loss = len(epoch_loss_student)
        if args.mean_teacher:
            for ind, (s_val, t_val) in enumerate(zip(epoch_loss_student, epoch_loss_teacher)):
                loss_dict = {'student': s_val, 'teacher': t_val}
                writer.add_scalars("train/instant_loss", loss_dict, total_steps_train + ind)
            for ind, val in enumerate(epoch_loss_consistency):
                writer.add_scalar("train/instant_loss_consistency", val, total_steps_train + ind)
            if args.classification:
                for ind, (s_val, t_val) in enumerate(zip(epoch_acc_student, epoch_acc_teacher)):
                    acc_dict = {'student': s_val, 'teacher': t_val}
                    writer.add_scalars("train/instant_pixel_acc", acc_dict, total_steps_train + ind)
                for ind, (s_val, t_val) in enumerate(zip(epoch_mean_acc_student, epoch_mean_acc_teacher)):
                    acc_dict = {'student': s_val, 'teacher': t_val}
                    writer.add_scalars("train/instant_mean_acc", acc_dict, total_steps_train + ind)
                for ind in range(0, len(dataset_train)):
                    acc_dict = {}
                    for class_ind in range(0,args.force_n_classes):
                        s_val = epoch_class_acc_student[ind, class_ind]
                        t_val = epoch_class_acc_teacher[ind, class_ind]
                        if (~np.isnan(s_val)):
                            acc_dict['student/'+str(class_ind)] = s_val
                        if (~np.isnan(t_val)):
                            acc_dict['teacher/'+str(class_ind)] = t_val
                    writer.add_scalars("train/instant_class_acc", acc_dict, total_steps_train + ind)

        else:
            for ind, val in enumerate(epoch_loss_student):
                writer.add_scalar("train/instant_loss", val, total_steps_train + ind)
            if args.classification:
                for ind, val in enumerate(epoch_acc_student):
                    writer.add_scalar("train/instant_pixel_acc", val, total_steps_train + ind)
                for ind, val in enumerate(epoch_mean_acc_student):
                    writer.add_scalar("train/instant_mean_acc", val, total_steps_train + ind)
                for ind in range(0, len(dataset_train)):
                    acc_dict = {}
                    for class_ind in range(0, args.force_n_classes):
                        val = epoch_class_acc_student[ind, class_ind]
                        if ~np.isnan(val):
                            acc_dict[str(class_ind)] = val
                    writer.add_scalars("train/instant_class_acc/", acc_dict, total_steps_train + ind)

        total_steps_train += len_epoch_loss

        avg_epoch_loss_student = sum(epoch_loss_student)/len(epoch_loss_student)
        if args.mean_teacher:
            loss_dict = {'student': avg_epoch_loss_student, 'teacher': sum(epoch_loss_teacher)/len(epoch_loss_teacher)}
            writer.add_scalars("train/epoch_loss", loss_dict, total_steps_train)
            writer.add_scalar("train/epoch_loss_consistency", sum(epoch_loss_consistency)/len(epoch_loss_consistency), total_steps_train)
            if args.classification:
                acc_dict = {'student': sum(epoch_acc_student)/len(epoch_acc_student), 'teacher': sum(epoch_acc_teacher)/len(epoch_acc_teacher)}
                writer.add_scalars("train/epoch_pixel_acc", acc_dict, total_steps_train)
                acc_dict = {'student': sum(epoch_mean_acc_student)/len(epoch_mean_acc_student), 'teacher': sum(epoch_mean_acc_teacher)/len(epoch_mean_acc_teacher)}
                writer.add_scalars("train/epoch_mean_acc", acc_dict, total_steps_train)
                # class acc
                student_mean = np.nanmean(epoch_class_acc_student, axis=0)
                teacher_mean = np.nanmean(epoch_class_acc_teacher, axis=0)
                acc_dict = {}
                for class_ind in range(0, args.force_n_classes):
                    acc_dict['student/'+str(class_ind)] = student_mean[class_ind]
                    acc_dict['teacher/'+str(class_ind)] = teacher_mean[class_ind]
                writer.add_scalars("train/epoch_class_acc", acc_dict, total_steps_train)

        else:
            writer.add_scalar("train/epoch_loss", avg_epoch_loss_student, total_steps_train)
            if args.classification:
                writer.add_scalar("train/epoch_pixel_acc", sum(epoch_acc_student)/len(epoch_acc_student), total_steps_train)
                writer.add_scalar("train/epoch_mean_acc", sum(epoch_mean_acc_student)/len(epoch_mean_acc_student), total_steps_train)
                student_mean = np.nanmean(epoch_class_acc_student, axis=0)
                acc_dict = {}
                for class_ind in range(0, args.force_n_classes):
                    acc_dict[str(class_ind)] = student_mean[class_ind]
                writer.add_scalars("train/epoch_class_acc", acc_dict, total_steps_train)
        # Output class power costs
        if args.regression and args.force_n_classes > 0:
            power_dict = {}
            for ind, val in enumerate(output_student_power.squeeze()):
                power_dict[str(ind)] = val
            writer.add_scalars("params/class_cost_student", power_dict, total_steps_train)
            if args.likelihood_loss:
                var_dict = {}
                for ind, val in enumerate(output_student_power_var.squeeze()):
                    var_dict[str(ind)] = val.sqrt()
                writer.add_scalars("params/class_cost_std_student", var_dict, total_steps_train)
            if args.mean_teacher:
                power_dict = {}
                for ind, val in enumerate(output_teacher_power.squeeze()):
                    power_dict[str(ind)] = val
                writer.add_scalars("params/class_cost_teacher", power_dict, total_steps_train)
                if args.likelihood_loss:
                    var_dict = {}
                    for ind, val in enumerate(output_teacher_power_var.squeeze()):
                        var_dict[str(ind)] = val.sqrt()
                    writer.add_scalars("params/class_cost_std_teacher", var_dict, total_steps_train)


        # Clear loss for next loss print iteration.
        epoch_loss_student = []
        if args.mean_teacher:
            epoch_loss_teacher = []
        epoch_loss_consistency = []
        if args.classification:
            epoch_acc_student = []
            epoch_mean_acc_student = []
            if args.mean_teacher:
                epoch_acc_teacher = []
                epoch_mean_acc_teacher = []
        # Print current loss. 
        print(f'loss: {avg_epoch_loss_student:0.4} (epoch: {epoch}, step: {step})', 
                "// Train: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size), 
                "// Load: %.4f s" % (sum(time_load) / len(time_load) / args.batch_size),
                "// Iter: %.4f s" % (sum(time_iter) / len(time_iter) / args.batch_size))

        if step == 0:
            time_iter.clear()
        time_iter.append(time.time() - start_time)
        # Save time for image loading duration.
        start_time = time.time()

            
        average_epoch_loss_train = avg_epoch_loss_student   
        
        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model_student.eval()
        epoch_loss_student_val = []
        if args.mean_teacher:
            model_teacher.eval()
            epoch_loss_teacher_val = []
        if args.classification:
            epoch_acc_student_val = []
            epoch_mean_acc_student_val = []
            epoch_class_acc_student_val = np.empty((len(dataset_val), args.force_n_classes))
            if args.mean_teacher:
                epoch_acc_teacher_val = []
                epoch_mean_acc_teacher_val = []
                epoch_class_acc_teacher_val = np.empty((len(dataset_val), args.force_n_classes))
        time_val = []


        for step, (images1, images2, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images1 = images1.cuda()
                if args.mean_teacher:
                    images2 = images2.cuda()
                labels = labels.cuda()
 
            with torch.no_grad():
                inputs1 = Variable(images1)    
                if args.mean_teacher:
                    inputs2 = Variable(images2)    
                targets = Variable(labels)

                if args.force_n_classes:
                    if args.likelihood_loss:
                        output_student_prob, output_student_power, output_student_power_var = model_student(inputs1) 
                    else:
                        output_student_prob, output_student_power = model_student(inputs1) 
                    max_prob_student, output_student = getMaxProbValue(output_student_prob, output_student_power, apply_softmax)
                    if args.mean_teacher:
                        if args.likelihood_loss:
                            output_teacher_prob, output_teacher_power, output_teacher_power_var = model_teacher(inputs2) 
                        else:
                            output_teacher_prob, output_teacher_power = model_teacher(inputs2) 
                        max_prob_teacher, output_teacher = getMaxProbValue(output_teacher_prob, output_teacher_power, apply_softmax)
                    # Compute weighted power consumption
                    if args.regression:
                        sum_dim = output_student_prob.dim()-3
                        weighted_sum_output = (output_student_prob * output_student_power).sum(dim=sum_dim, keepdim=True)
                else:
                    output_student = model_student(inputs1)
                    if args.mean_teacher:
                        output_teacher = model_teacher(inputs2)

                if args.force_n_classes > 0:
                    if args.classification:
                        loss_student = criterion_val(output_student_prob, targets)
                        if args.mean_teacher:
                            loss_teacher = criterion_val(output_teacher_prob, targets)
                    else:   # regression
                        if args.likelihood_loss:
                            loss_student = criterion_val(output_student_prob, output_student_power, output_student_power_var, targets)
                            if args.mean_teacher:
                                loss_teacher = criterion_val(output_teacher_prob, output_teacher_power, output_teacher_power_var, targets)
                        else:
                            loss_student = criterion_val(output_student_prob, output_student_power, targets)
                            if args.mean_teacher:
                                loss_teacher = criterion_val(output_teacher_prob, output_teacher_power, targets)
                else :
                    if args.likelihood_loss:
                        output_var_student = output_student[:,1,:,:].pow(2)
                        output_student = output_student[:,0,:,:]
                        loss_student = criterion_val(output_student, output_var_student, targets)
                        if args.mean_teacher:
                            output_var_teacher = output_teacher[:,1,:,:].pow(2)
                            output_teacher = output_teacher[:,0,:,:]
                            loss_teacher = criterion_val(output_teacher, output_var_teacher, targets)

                    else:
                        loss_student = criterion_val(output_student, targets)
                        if args.mean_teacher:
                            loss_teacher = criterion_val(output_teacher, targets)
                epoch_loss_student_val.append(loss_student.data.item())
                if args.mean_teacher:
                    epoch_loss_teacher_val.append(loss_teacher.data.item())
                if args.classification:
                    acc_student = criterion_acc(output_student_prob, targets)
                    mean_acc_student, epoch_class_acc_student_val[step,:] = criterion_mean_acc(output_student_prob, targets)
                    epoch_acc_student_val.append(acc_student.data.item())
                    epoch_mean_acc_student_val.append(mean_acc_student.data.item())
                    if args.mean_teacher:
                        acc_teacher = criterion_acc(output_teacher_prob, targets)
                        mean_acc_teacher, epoch_class_acc_teacher_val[step,:] = criterion_mean_acc(output_teacher_prob, targets)
                        epoch_acc_teacher_val.append(acc_teacher.data.item())
                        epoch_mean_acc_teacher_val.append(mean_acc_teacher.data.item())
                time_val.append(time.time() - start_time)

                # Plot images
                if args.visualize and step % steps_img_val == 0:
                    
                    step_vis_no = total_steps_val + len(epoch_loss_student_val)
                    start_time_plot = time.time()
                    image1 = inputs1[0].cpu().data
                    writer.add_image("val/input_student", image1, step_vis_no)
                    if args.mean_teacher:
                        image2 = inputs2[0].cpu().data
                        writer.add_image("val/input_teacher", image2, step_vis_no)
                    if args.regression:
                        writer.add_image("val/output_student", color_transform_output(output_student[0].cpu().data), step_vis_no)
                        if args.mean_teacher:
                            writer.add_image("val/output_teacher", color_transform_output(output_teacher[0].cpu().data), step_vis_no)
                        if args.likelihood_loss and args.force_n_classes == 0:
                            writer.add_image("val/std_dev_student", output_var_student[0].sqrt().cpu().data, step_vis_no)
                            if args.mean_teacher:
                                writer.add_image("val/std_dev_teacher", output_var_teacher[0].sqrt().cpu().data, step_vis_no)
                    if args.force_n_classes > 0:
                        writer.add_image("val/classes_student", color_transform_classes_prob(output_student_prob[0].cpu().data), step_vis_no)
                        writer.add_image("val/max_class_probability_student", max_prob_student[0], step_vis_no)
                        if args.mean_teacher:
                            writer.add_image("val/classes_teacher", color_transform_classes_prob(output_teacher_prob[0].cpu().data), step_vis_no)
                            writer.add_image("val/max_class_probability_teacher", max_prob_teacher[0], step_vis_no)
                        if args.regression:
                            writer.add_image("val/weighted_output_student", color_transform_output(weighted_sum_output[0].cpu().data), step_vis_no)
                    if args.classification:
                        writer.add_image("val/target", color_transform_classes(targets[0].cpu().data), step_vis_no)
                    else:
                        writer.add_image("val/target", color_transform_target(targets[0].cpu().data), step_vis_no)
                    print ("Time to paint images: ", time.time() - start_time_plot)
                # Plot histograms
                if args.visualize and steps_hist > 0 and step % steps_hist == 0:
                    image1 = inputs1[0].cpu().data

                    hist_ind = int(step / steps_hist)

                    if args.force_n_classes > 0:
                        _, hist_array = output_student_prob[0].cpu().data.max(dim=0, keepdim=True)
                        # Use train steps so we can compare with class power plot
                        writer.add_histogram("val/hist_"+str(hist_ind), hist_array.numpy().flatten(), total_steps_train, hist_bins)  
                        writer.add_image("val_"+str(hist_ind)+"/classes_student", color_transform_classes_prob(output_student_prob[0].cpu().data), total_steps_train)
                        writer.add_image("val_"+str(hist_ind)+"/max_class_probability_student", max_prob_student[0], total_steps_train)
                        if args.mean_teacher:
                            writer.add_image("val_"+str(hist_ind)+"/classes_teacher", color_transform_classes_prob(output_teacher_prob[0].cpu().data), total_steps_train)
                            writer.add_image("val_"+str(hist_ind)+"/max_class_probability_teacher", max_prob_teacher[0], total_steps_train)

                    writer.add_image("val_"+str(hist_ind)+"/output_student", color_transform_output(output_student[0].cpu().data), total_steps_train)
                    if args.mean_teacher:
                        writer.add_image("val_"+str(hist_ind)+"/output_teacher", color_transform_output(output_teacher[0].cpu().data), total_steps_train)

                    if epoch == start_epoch:
                        writer.add_image("val_"+str(hist_ind)+"/input", image1, total_steps_train)  # Visualize image used to compute histogram
                       
        total_steps_val += len(epoch_loss_student_val)
        avg_loss_student_val = sum(epoch_loss_student_val) / len(epoch_loss_student_val)
        print(f'VAL loss_teacher: {avg_loss_student_val:0.4} (epoch: {epoch}, step: {total_steps_val})', 
                "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
        if args.mean_teacher:
            avg_loss_teacher_val = sum(epoch_loss_teacher_val) / len(epoch_loss_teacher_val)
            loss_dict = {'student': avg_loss_student_val, 'teacher': avg_loss_teacher_val}
            writer.add_scalars("val/epoch_loss", loss_dict, total_steps_val)
            if args.classification:
                acc_dict = {'student': sum(epoch_acc_student_val) / len(epoch_acc_student_val), 'teacher':sum(epoch_acc_teacher_val) / len(epoch_acc_teacher_val)}
                writer.add_scalars("val/epoch_pixel_acc", acc_dict, total_steps_val)
                acc_dict = {'student': sum(epoch_mean_acc_student_val) / len(epoch_mean_acc_student_val), 
                            'teacher':sum(epoch_mean_acc_teacher_val) / len(epoch_mean_acc_teacher_val)}
                writer.add_scalars("val/epoch_mean_acc", acc_dict, total_steps_val)
                student_mean = np.nanmean(epoch_class_acc_student_val, axis=0)
                teacher_mean = np.nanmean(epoch_class_acc_teacher_val, axis=0)
                acc_dict = {}
                for class_ind in range(0, args.force_n_classes):
                    acc_dict['student/'+str(class_ind)] = student_mean[class_ind]
                    acc_dict['teacher/'+str(class_ind)] = teacher_mean[class_ind]
                writer.add_scalars("val/epoch_class_acc", acc_dict, total_steps_val)
        else:
            writer.add_scalar("val/epoch_loss", avg_loss_student_val, total_steps_val)
            if args.classification:
                writer.add_scalar("val/epoch_pixel_acc", sum(epoch_acc_student_val) / len(epoch_acc_student_val), total_steps_val)
                writer.add_scalar("val/epoch_mean_acc", sum(epoch_mean_acc_student_val) / len(epoch_mean_acc_student_val), total_steps_val)
                student_mean = np.nanmean(epoch_class_acc_student_val, axis=0)
                acc_dict = {}
                for class_ind in range(0, args.force_n_classes):
                    acc_dict[str(class_ind)] = student_mean[class_ind]
                writer.add_scalars("val/epoch_class_acc", acc_dict, total_steps_val)


        # remember best valIoU and save checkpoint
        if args.classification:
            current_acc = sum(epoch_mean_acc_student_val) / len(epoch_mean_acc_student_val)
        else:   # regression
            current_acc = -avg_loss_student_val
            if args.mean_teacher:
                current_acc = -avg_loss_teacher_val

        epoch_loss_student_val = []
        if args.mean_teacher:
            epoch_loss_teacher_val = []
        if args.classification:
            epoch_acc_student_val = []
            epoch_mean_acc_student_val = []
            if args.mean_teacher:
                epoch_acc_teacher_val = []
                epoch_mean_acc_teacher_val = []

        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model_student),
            'state_dict': model_student.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            if args.mean_teacher:
                torch.save(model_teacher.state_dict(), filename)
            else:
                torch.save(model_student.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            if args.mean_teacher:
                torch.save(model_teacher.state_dict(), filenamebest)
            else:
                torch.save(model_student.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, current_acc))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, current_acc))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, avg_loss_student_val, usedLr ))
    
    return(model_student, model_teacher)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    # Check that we don't do classification and regression
    assert bool(args.classification) or bool(args.regression), "Did not specify --regression or --classification"
    assert bool(args.classification) != bool(args.regression), "Can only do regression or classification, not both. "
    if args.classification:
        assert args.force_n_classes > 0, "Did not specify number of classes for classification."

    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    

    print("========== NETWORK TRAINING ===========")
    pretrainedEnc = None
    if args.pretrainedEncoder:
        print("Loading pretrained encoder")
        from erfnet_imagenet import ERFNet as ERFNet_imagenet
        pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
        pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
        pretrainedEnc = next(pretrainedEnc.children()).features.encoder
        if (not args.cuda):
            pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
    
    model_student = model_file.Net( encoder=pretrainedEnc, softmax_classes=args.force_n_classes, likelihood_loss=args.likelihood_loss, 
                                    spread_class_power=args.spread_init, late_dropout_prob=args.late_dropout_prob)  #Add decoder to encoder
    model_teacher = model_file.Net( encoder=pretrainedEnc, softmax_classes=args.force_n_classes, likelihood_loss=args.likelihood_loss, 
                                    spread_class_power=args.spread_init, late_dropout_prob=args.late_dropout_prob)  #Add decoder to encoder

    if args.cuda:
        def make_cuda(model):
            return torch.nn.DataParallel(model).cuda()

        model_student = make_cuda(model_student)
        model_teacher = make_cuda(model_teacher)

    if args.pretrained:
        pretrained = torch.load(args.pretrained)['state_dict']
        model_student.load_state_dict(pretrained, strict=False)
        model_teacher.load_state_dict(pretrained, strict=False)
        print("Loaded pretrained model")

    #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model_student, model_teacher = train(args, model_student, model_teacher, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help="Use CUDA")  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet_classification", help="Model name to be used")

    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/", help="Directory for dataset")
    parser.add_argument('--height', type=int, default=512, help="Height of input images")
    parser.add_argument('--num-epochs', type=int, default=150, help="Number of epochs before termination")
    parser.add_argument('--num-workers', type=int, default=8, help="Number of data loader workers")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for train/eval")   
    parser.add_argument('--epochs-save', type=int, default=10, help="Save network every N epochs")    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True, help="Output dir for saving network")
    parser.add_argument('--pretrainedEncoder', help="File path to pretrained encoder") #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--pretrained', help="File path to pretrained network") #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true', help="Visualize network output")
    parser.add_argument('--force-n-classes', type=int, default=0, help="Force network to discretize output into N classes")   # Force network to discretize output into classes with discrete output power
    parser.add_argument('--likelihood-loss', action='store_true', default=False, help="Estimate class value variance")   # Learn a Gaussian for every class instead of just mean.
    parser.add_argument('--spread-init', action='store_true', default=False, help="Initialize class values with a spread")    # Spread initial class power over interval [0.7,...,2.0]
    parser.add_argument('--fix-class-power', action='store_true', default=False, help="Fix class values and only learn class membership")    # Fix class power so that it is not optimized
    parser.add_argument('--late-dropout-prob', type=float, default=0.1, help="Dropout probability in last layer if class discretization is used")    # Specify dropout prob in last layer after softmax
    parser.add_argument('--subsample', type=int, default=1, help="Only use every nth image of the dataset") # Only use every nth image of a dataset.
    parser.add_argument('--mean-teacher', action='store_true', default=False, help="Use mean-teacher for training")    # Disable mean teacher
    parser.add_argument('--regression', action='store_true', default=False, help="Training for a regression problem (can also have discretized classes)") # Plot power consumption or whatever scalar value
    parser.add_argument('--classification', action='store_true', default=False, help="Training for a classification problem.") # Plot power consumption or whatever scalar value
    parser.add_argument('--label-name', type=str, default="class", help="Suffix for label file names")
    parser.add_argument('--consistency-weight', type=float, default=1.0, help="Weight of consistency loss.")

    parser.add_argument('--resume', action='store_true', help="Resume from previous checkpoint")    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
