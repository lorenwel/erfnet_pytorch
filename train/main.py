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
from transform import Relabel, ToLabel, Colorize, ColorizeMinMax, ColorizeWithProb, ColorizeClasses, FloatToLongLabel, ToFloatLabel, getMaxProbValue
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile

NUM_CHANNELS = 3
# NUM_CLASSES = 20 #pascal=22, cityscapes=20
NUM_HISTOGRAMS = 5
NUM_IMG_PER_EPOCH = 5
# Optimizer params.
LEARNING_RATE=5e-4
BETAS=(0.9, 0.999)
OPT_EPS=1e-08
WEIGHT_DECAY=1e-6

DISCOUNT_RATE_START=0.1
DISCOUNT_RATE=0.01
MAX_CONSISTENCY_EPOCH=30
DISCOUNT_RATE_START_EPOCH=50

color_transform_target = Colorize(1.0, 2.0, remove_negative=True, extend=True, white_val=1.0)  # min_val, max_val, remove negative
color_transform_output = Colorize(1.0, 2.0, remove_negative=False, extend=True, white_val=1.0)  # Automatic color based on tensor min/max val
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

        self.color_augmentation = ColorJitter(brightness=0.5,
                                              contrast=0.5,
                                              saturation=0.5,
                                              hue=0.07)
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
            affine_angle = random.uniform(-self.affine_angle, self.affine_angle)
            shear_angle = random.uniform(-self.shear_angle, self.shear_angle)
            flip = random.random() < 0.5
            # Do image transformation 
            input_crop = self.transform_augmentation(input, flip, rotation_angle, affine_angle, shear_angle)
            target_crop = self.transform_augmentation(target, flip, rotation_angle, affine_angle, shear_angle)
            crop_val = np.random.randint(6)

            # Crop
            # Assumes base image is 480x640
            img_size = np.array([32, 24]) * (20-crop_val)
            hor_pos = int(np.random.rand() * (640 - img_size[0]))
            input_crop = input_crop.crop((hor_pos, 480 - img_size[1], hor_pos + img_size[0], 480))
            # input.show()
            target_crop = target_crop.crop((hor_pos, 480 - img_size[1], hor_pos + img_size[0], 480))
            target_test = np.array(target_crop, dtype="float32")
            # Condition to make sure we have crop containing footprints
            if np.any(target_test > 0.0):
                input = input_crop
                target = target_crop
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


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

class MSELossPosElements(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.MSELoss(False, False)

    def forward(self, output_prob, output_cost, targets):
        shape = output_prob.size()
        cur_loss = self.loss(output_cost.expand(shape), targets.expand(shape))
        # cur_loss = self.loss(output_prob.expand(shape), targets.expand(shape))
        weighted_loss = cur_loss * output_prob
        # only compute loss for places where label exists.
        masked_loss = weighted_loss.masked_select(torch.gt(targets, 0.0))
        return masked_loss.mean()

class L1LossClassProbMasked(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.L1Loss(False, False)

    def forward(self, output_prob, output_cost, targets):
        shape = output_prob.size()
        cur_loss = self.loss(output_cost.expand(shape), targets.expand(shape))
        # cur_loss = self.loss(output_prob.expand(shape), targets.expand(shape))
        weighted_loss = cur_loss * output_prob
        # only compute loss for places where label exists.
        masked_loss = weighted_loss.sum(dim=1, keepdim=True).masked_select(torch.gt(targets, 0.0))
        return masked_loss.mean()

class L1LossMasked(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.L1Loss(False, False)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets).masked_select(torch.gt(targets, 0.0)).mean()

class L1LossTraversability(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.non_trav_weight = torch.autograd.Variable(torch.from_numpy(np.array([0.1], dtype="float32"))).cuda()

        self.loss = torch.nn.L1Loss(False, False)

    def forward(self, outputs, targets):
        return (1-outputs[torch.gt(targets, 0.0)]).abs().mean() + (outputs[torch.le(targets, 0.0)]).abs().mean()*self.non_trav_weight


class L1Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.L1Loss(True, True)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

class MSELossWeighted(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.MSELoss(False, False)

    def forward(self, outputs, targets, weight):
        return (self.loss(outputs, targets) * weight).mean()

def copyWeightsToModelNoGrad(model_source, model_target):
    for source, target in zip(model_source.parameters(), model_target.parameters()):
        target = source.clone() # Using detach here doesn't work.
        # Setting requires_grad False here doesn't work.
    for target in model_target.parameters():
        target.requires_grad=False

def copyWeightsToModelWithDiscount(model_source, model_target, discount_factor):
    for source, target in zip(model_source.parameters(), model_target.parameters()):
        target.data = target.data * (1-discount_factor) + discount_factor * source.data


best_acc = 0

def train(args, model_student, model_teacher, enc=False):
    global best_acc

    weight = torch.ones(1)
    
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    # Set data loading variables
    co_transform = MyCoTransform(enc, augment=True, height=480)#1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=480)#1024)
    dataset_train = self_supervised_power(args.datadir, co_transform, 'train', file_format="csv", subsample=args.subsample)
    # dataset_train = self_supervised_power(args.datadir, None, 'train')
    dataset_val = self_supervised_power(args.datadir, None, 'val', file_format="csv", subsample=args.subsample)

    if args.force_n_classes > 0:
        color_transform_classes = ColorizeClasses(args.force_n_classes)  # Automatic color based on max class probability

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        weight = weight.cuda()

    # Set Loss functions
    if args.force_n_classes > 0:
        criterion = L1LossClassProbMasked() # L1 loss weighted with class prob with averaging over mini-batch
    else:
        criterion = L1LossMasked()     

    criterion_trav = L1LossTraversability()
    criterion_autoenc = L1Loss()
    criterion_consistency = MSELossWeighted()
    criterion_val = L1LossMasked()     # L1 loss with averaging over mini-batch
    print(type(criterion))

    savedir = f'../save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model_student))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model_student.parameters(), LEARNING_RATE, BETAS,  eps=OPT_EPS, weight_decay=WEIGHT_DECAY)
    if args.alternate_optimization:
        params_prob = [param for name, param in model.named_parameters() if name != "module.class_power"]
        params_power  = [param for name, param in model.named_parameters() if name == "module.class_power"]
        optimizer_prob = Adam(params_prob, LEARNING_RATE, BETAS,  eps=OPT_EPS, weight_decay=WEIGHT_DECAY)
        optimizer_power = Adam(params_power, LEARNING_RATE, BETAS,  eps=OPT_EPS, weight_decay=WEIGHT_DECAY)

    start_epoch = 1

    if args.pretrained:
        pretrained = torch.load(args.pretrained)
        model_student.load_state_dict(pretrained)
        model_teacher.load_state_dict(pretrained)
        print("Loaded pretrained model")
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

    # Initialize teacher with same weights as student. 
    copyWeightsToModelNoGrad(model_student, model_teacher)

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2
    if args.alternate_optimization:
        scheduler_prob = lr_scheduler.LambdaLR(optimizer_prob, lr_lambda=lambda1)                             ## scheduler 2
        scheduler_power = lr_scheduler.LambdaLR(optimizer_power, lr_lambda=lambda1)                             ## scheduler 2

    if args.visualize:
        board = Dashboard(args.port)
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

        if epoch < MAX_CONSISTENCY_EPOCH:
            cur_consistency_weight = epoch / MAX_CONSISTENCY_EPOCH
        else:
            cur_consistency_weight = 1.0

        if args.no_mean_teacher:
            cur_consistency_weight = 0.0

        if args.alternate_optimization:
            if epoch % 2 == 0:
                scheduler_power.step(epoch)
            else:
                scheduler_prob.step(epoch)
        else:
            scheduler.step(epoch)    ## scheduler 2

        average_loss_student_val = 0
        average_loss_teacher_val = 0

        epoch_loss_student = []
        epoch_loss_teacher = []
        epoch_loss_trav_student = []
        epoch_loss_trav_teacher = []
        epoch_loss_autoenc_student = []
        epoch_loss_autoenc_teacher = []
        epoch_loss_consistency = []
        time_train = []
        time_load = []
        time_iter = [0.0]
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      


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
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images1 = images1.cuda()
                images2 = images2.cuda()
                labels = labels.cuda()

            inputs1 = Variable(images1)
            inputs2 = Variable(images2, volatile=True)
            targets = Variable(labels)
            if (args.force_n_classes) > 0:
                # Forced into discrete classes. 
                output_student_prob, output_student_trav, output_student_autoenc, output_student_power = model_student(inputs1, only_encode=enc)
                output_teacher_prob, output_teacher_trav, output_teacher_autoenc, output_teacher_power = model_teacher(inputs2, only_encode=enc)
                if args.alternate_optimization:
                    if epoch % 2 == 0:
                        optimizer_power.zero_grad()
                    else:
                        optimizer_prob.zero_grad()
                else:
                    optimizer.zero_grad()
                loss_student_pred = criterion(output_student_prob, output_student_power, targets)
                loss_teacher_pred = criterion(output_teacher_prob, output_teacher_power, targets)
                loss_consistency = criterion_consistency(output_student_prob, output_teacher_prob, cur_consistency_weight)
            else:
                # Straight regressoin
                output_student, output_student_trav, output_student_autoenc = model_student(inputs1, only_encode=enc)
                output_teacher, output_teacher_trav, output_teacher_autoenc = model_teacher(inputs2, only_encode=enc)
                optimizer.zero_grad()
                loss_student_pred = criterion(output_student, targets)
                loss_teacher_pred = criterion(output_teacher, targets)
                loss_consistency = criterion_consistency(output_student, output_teacher, cur_consistency_weight)

            # Loss independent of how scalar value is determined
            loss_student_trav = criterion_trav(output_student_trav, targets)
            loss_teacher_trav = criterion_trav(output_teacher_trav, targets)
            loss_student_autoenc = criterion_autoenc(output_student_autoenc, inputs1)
            loss_teacher_autoenc = criterion_autoenc(output_teacher_autoenc, inputs2)


            #print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            # Do backward pass.
            loss_student_pred.backward(retain_graph=True)
            loss_student_trav.backward(retain_graph=True)
            if epoch>0 and not args.no_mean_teacher:
                loss_student_autoenc.backward(retain_graph=True)
                loss_consistency.backward()
            else: 
                loss_student_autoenc.backward()


            if args.alternate_optimization:
                if epoch % 2 == 0:
                    optimizer_power.step()
                else:
                    optimizer_prob.step()
            else:
                optimizer.step()

            # Average over first 50 epochs.
            if epoch < DISCOUNT_RATE_START_EPOCH:
                cur_discount_rate = DISCOUNT_RATE_START
            else:
                cur_discount_rate = DISCOUNT_RATE
            copyWeightsToModelWithDiscount(model_student, model_teacher, cur_discount_rate)

            # copyWeightsToModelWithDiscount(model_student, model_teacher, DISCOUNT_RATE)

            epoch_loss_student.append(loss_student_pred.data[0])
            epoch_loss_teacher.append(loss_teacher_pred.data[0])
            epoch_loss_trav_student.append(loss_student_trav.data[0])
            epoch_loss_trav_teacher.append(loss_teacher_trav.data[0])
            epoch_loss_autoenc_student.append(loss_student_autoenc.data[0])
            epoch_loss_autoenc_teacher.append(loss_teacher_autoenc.data[0])
            epoch_loss_consistency.append(loss_consistency.data[0])
            time_train.append(time.time() - start_time)

            # if (doIouTrain):
            #     #start_time_iou = time.time()
            #     iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
            #     #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            #print(outputs.size())
            if args.visualize and step % steps_img_train == 0:
                step_vis_no = total_steps_train + len(epoch_loss_student)

                # Figure out and compute tensor to visualize. 
                if args.force_n_classes > 0:
                    # Compute weighted power consumption
                    sum_dim = output_student_prob.dim()-3
                    weighted_sum_output = (output_student_prob * output_student_power).sum(dim=sum_dim, keepdim=True)
                    if (isinstance(output_student_prob, list)):
                        max_prob, vis_output = getMaxProbValue(output_student_prob[0][0].cpu().data, output_student_power[0][0].cpu().data)
                        max_prob_teacher, vis_output_teacher = getMaxProbValue(output_teacher_prob[0][0].cpu().data, output_teacher_power[0][0].cpu().data)
                        writer.add_image("train/2_classes", color_transform_classes(output_student_prob[0][0].cpu().data), step_vis_no)
                        writer.add_image("train/3_max_class_probability", max_prob[0][0], step_vis_no)
                        writer.add_image("train/4_weighted_output", color_transform_output(weighted_sum_output[0][0].cpu().data), step_vis_no)
                    else:
                        max_prob, vis_output = getMaxProbValue(output_student_prob[0].cpu().data, output_student_power[0].cpu().data)
                        max_prob_teacher, vis_output_teacher = getMaxProbValue(output_teacher_prob[0].cpu().data, output_teacher_power[0].cpu().data)
                        writer.add_image("train/2_classes", color_transform_classes(output_student_prob[0].cpu().data), step_vis_no)
                        writer.add_image("train/3_max_class_probability", max_prob[0], step_vis_no)
                        writer.add_image("train/4_weighted_output", color_transform_output(weighted_sum_output[0].cpu().data), step_vis_no)
                else:
                    if (isinstance(output_teacher, list)):
                        vis_output = output_student[0][0].cpu().data
                        vis_output_teacher = output_teacher[0][0].cpu().data
                    else:
                        vis_output = output_student[0].cpu().data
                        vis_output_teacher = output_teacher[0].cpu().data

                if (isinstance(output_teacher_trav, list)):
                    trav_output = output_student_trav[0][0].cpu().data
                    trav_output_teacher = output_teacher_trav[0][0].cpu().data
                    autoenc_output = output_student_autoenc[0][0].cpu().data
                    autoenc_output_teacher = output_teacher_autoenc[0][0].cpu().data
                else:
                    trav_output = output_student_trav[0].cpu().data
                    trav_output_teacher = output_teacher_trav[0].cpu().data
                    autoenc_output = output_student_autoenc[0].cpu().data
                    autoenc_output_teacher = output_teacher_autoenc[0].cpu().data

                start_time_plot = time.time()
                image1 = inputs1[0].cpu().data
                image2 = inputs2[0].cpu().data
                # board.image(image, f'input (epoch: {epoch}, step: {step})')
                writer.add_image("train/1_input_student", image1, step_vis_no)
                writer.add_image("train/1_input_teacher", image2, step_vis_no)
                writer.add_image("train/5_output_student", color_transform_output(vis_output), step_vis_no)
                writer.add_image("train/5_output_teacher", color_transform_output(vis_output_teacher), step_vis_no)
                writer.add_image("train/7_output_trav_student", trav_output, step_vis_no)
                writer.add_image("train/7_output_trav_teacher", trav_output_teacher, step_vis_no)
                writer.add_image("train/7_output_autoenc_student", autoenc_output, step_vis_no)
                writer.add_image("train/7_output_autoenc_teacher", autoenc_output_teacher, step_vis_no)
                # board.image(color_transform_target(targets[0].cpu().data),
                #     f'target (epoch: {epoch}, step: {step})')
                writer.add_image("train/6_target", color_transform_target(targets[0].cpu().data), step_vis_no)
                print ("Time to paint images: ", time.time() - start_time_plot)
                

        len_epoch_loss = len(epoch_loss_student)
        for ind, val in enumerate(epoch_loss_student):
            writer.add_scalar("train/instant_loss_student", val, total_steps_train + ind)
        for ind, val in enumerate(epoch_loss_teacher):
            writer.add_scalar("train/instant_loss_teacher", val, total_steps_train + ind)
        for ind, val in enumerate(epoch_loss_trav_student):
            writer.add_scalar("train/instant_loss_trav_student", val, total_steps_train + ind)
        for ind, val in enumerate(epoch_loss_trav_teacher):
            writer.add_scalar("train/instant_loss_trav_teacher", val, total_steps_train + ind)
        for ind, val in enumerate(epoch_loss_autoenc_student):
            writer.add_scalar("train/instant_loss_autoenc_student", val, total_steps_train + ind)
        for ind, val in enumerate(epoch_loss_autoenc_teacher):
            writer.add_scalar("train/instant_loss_autoenc_teacher", val, total_steps_train + ind)
        for ind, val in enumerate(epoch_loss_consistency):
            writer.add_scalar("train/instant_loss_consistency", val, total_steps_train + ind)
        total_steps_train += len_epoch_loss
        avg_loss_teacher = sum(epoch_loss_teacher)/len(epoch_loss_teacher)
        writer.add_scalar("train/epoch_loss_student", sum(epoch_loss_student)/len(epoch_loss_student), total_steps_train)
        writer.add_scalar("train/epoch_loss_teacher", avg_loss_teacher, total_steps_train)
        writer.add_scalar("train/epoch_loss_trav_student", sum(epoch_loss_trav_student)/len(epoch_loss_trav_student), total_steps_train)
        writer.add_scalar("train/epoch_loss_trav_teacher", sum(epoch_loss_trav_teacher)/len(epoch_loss_trav_teacher), total_steps_train)
        writer.add_scalar("train/epoch_loss_autoenc_student", sum(epoch_loss_autoenc_student)/len(epoch_loss_autoenc_student), total_steps_train)
        writer.add_scalar("train/epoch_loss_autoenc_teacher", sum(epoch_loss_autoenc_teacher)/len(epoch_loss_autoenc_teacher), total_steps_train)
        writer.add_scalar("train/epoch_loss_consistency", sum(epoch_loss_consistency)/len(epoch_loss_consistency), total_steps_train)
        # Clear loss for next loss print iteration.
        # Output class power costs
        power_dict = {}
        if args.force_n_classes > 0:
            for ind, val in enumerate(output_teacher_power.squeeze()):
                power_dict[str(ind)] = val
            writer.add_scalars("params/class_cost", power_dict, total_steps_train)
        epoch_loss_student = []
        epoch_loss_teacher = []
        epoch_loss_consistency = []
        # Print current loss. 
        print(f'loss: {avg_loss_teacher:0.4} (epoch: {epoch}, step: {step})', 
                "// Train: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size), 
                "// Load: %.4f s" % (sum(time_load) / len(time_load) / args.batch_size),
                "// Iter: %.4f s" % (sum(time_iter) / len(time_iter) / args.batch_size))

        if step == 0:
            time_iter.clear()
        time_iter.append(time.time() - start_time)
        # Save time for image loading duration.
        start_time = time.time()

            
        average_epoch_loss_train = avg_loss_teacher   
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model_student.eval()
        model_teacher.eval()
        epoch_loss_student_val = []
        epoch_loss_teacher_val = []
        epoch_loss_trav_student_val = []
        epoch_loss_trav_teacher_val = []
        epoch_loss_autoenc_student_val = []
        epoch_loss_autoenc_teacher_val = []
        time_val = []


        for step, (images1, images2, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images1 = images1.cuda()
                images2 = images2.cuda()
                labels = labels.cuda()

            inputs1 = Variable(images1, volatile=True)    #volatile flag makes it free backward or outputs for eval
            inputs2 = Variable(images2, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)

            if args.force_n_classes:
                output_student_prob, output_student_trav, output_student_autoenc, output_student_power = model_student(inputs1, only_encode=enc) 
                output_teacher_prob, output_teacher_trav, output_teacher_autoenc, output_teacher_power = model_teacher(inputs2, only_encode=enc) 
                max_prob, output_student = getMaxProbValue(output_student_prob, output_student_power)
                max_prob, output_teacher = getMaxProbValue(output_teacher_prob, output_teacher_power)
                # Compute weighted power consumption
                sum_dim = output_student_prob.dim()-3
                weighted_sum_output = (output_student_prob * output_student_power).sum(dim=sum_dim, keepdim=True)
            else:
                output_student, output_student_trav, output_student_autoenc = model_student(inputs1, only_encode=enc)
                output_teacher, output_teacher_trav, output_teacher_autoenc = model_teacher(inputs2, only_encode=enc)

            loss_student = criterion_val(output_student, targets)
            loss_teacher = criterion_val(output_teacher, targets)
            loss_student_trav = criterion_trav(output_student_trav, targets)
            loss_teacher_trav = criterion_trav(output_teacher_trav, targets)
            loss_student_autoenc = criterion_autoenc(output_student_autoenc, inputs1)
            loss_teacher_autoenc = criterion_autoenc(output_teacher_autoenc, inputs2)
            epoch_loss_student_val.append(loss_student.data[0])
            epoch_loss_teacher_val.append(loss_teacher.data[0])
            epoch_loss_trav_student_val.append(loss_student_trav.data[0])
            epoch_loss_trav_teacher_val.append(loss_teacher_trav.data[0])
            epoch_loss_autoenc_student_val.append(loss_student_autoenc.data[0])
            epoch_loss_autoenc_teacher_val.append(loss_teacher_autoenc.data[0])
            time_val.append(time.time() - start_time)


            #Add batch to calculate TP, FP and FN for iou estimation
            # if (doIouVal):
            #     #start_time_iou = time.time()
            #     iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
            #     #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            # Plot images
            if args.visualize and step % steps_img_val == 0:
                if (isinstance(output_teacher_trav, list)):
                    trav_output = output_student_trav[0][0].cpu().data
                    trav_output_teacher = output_teacher_trav[0][0].cpu().data
                    autoenc_output = output_student_autoenc[0][0].cpu().data
                    autoenc_output_teacher = output_teacher_autoenc[0][0].cpu().data
                else:
                    trav_output = output_student_trav[0].cpu().data
                    trav_output_teacher = output_teacher_trav[0].cpu().data
                    autoenc_output = output_student_autoenc[0].cpu().data
                    autoenc_output_teacher = output_teacher_autoenc[0].cpu().data

                step_vis_no = total_steps_val + len(epoch_loss_student_val)
                start_time_plot = time.time()
                image1 = inputs1[0].cpu().data
                image2 = inputs2[0].cpu().data
                # board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                writer.add_image("val/1_input_student", image1, step_vis_no)
                writer.add_image("val/1_input_teacher", image2, step_vis_no)
                if isinstance(output_teacher, list):   #merge gpu tensors
                    # board.image(color_transform_output(outputs[0][0].cpu().data),
                    # f'VAL output (epoch: {epoch}, step: {step})')
                    writer.add_image("val/5_output_teacher", color_transform_output(output_teacher[0][0].cpu().data), step_vis_no)
                    writer.add_image("val/5_output_student", color_transform_output(output_student[0][0].cpu().data), step_vis_no)
                    if args.force_n_classes > 0:
                        writer.add_image("val/2_classes", color_transform_classes(output_teacher_prob[0][0].cpu().data), step_vis_no)
                        writer.add_image("val/3_max_class_probability", max_prob[0][0], step_vis_no)
                        writer.add_image("val/4_weighted_output", color_transform_output(weighted_sum_output[0][0].cpu().data), step_vis_no)

                else:
                    # board.image(color_transform_output(outputs[0].cpu().data),
                    # f'VAL output (epoch: {epoch}, step: {step})')
                    writer.add_image("val/5_output_teacher", color_transform_output(output_teacher[0].cpu().data), step_vis_no)
                    writer.add_image("val/5_output_student", color_transform_output(output_student[0].cpu().data), step_vis_no)
                    if args.force_n_classes > 0:
                        writer.add_image("val/2_classes", color_transform_classes(output_teacher_prob[0].cpu().data), step_vis_no)
                        writer.add_image("val/3_max_class_probability", max_prob[0], step_vis_no)
                        writer.add_image("val/4_weighted_output", color_transform_output(weighted_sum_output[0].cpu().data), step_vis_no)
                # board.image(color_transform_target(targets[0].cpu().data),
                #     f'VAL target (epoch: {epoch}, step: {step})')
                writer.add_image("val/7_output_trav_student", trav_output, step_vis_no)
                writer.add_image("val/7_output_trav_teacher", trav_output_teacher, step_vis_no)
                writer.add_image("val/7_output_autoenc_student", autoenc_output, step_vis_no)
                writer.add_image("val/7_output_autoenc_teacher", autoenc_output_teacher, step_vis_no)
                writer.add_image("val/6_target", color_transform_target(targets[0].cpu().data), step_vis_no)
                print ("Time to paint images: ", time.time() - start_time_plot)
            # Plot histograms
            if args.force_n_classes > 0 and args.visualize and steps_hist > 0 and step % steps_hist == 0:
                hist_ind = int(step / steps_hist)
                if (isinstance(output_teacher_prob, list)):
                    _, hist_array = output_teacher_prob[0][0].cpu().data.max(dim=0, keepdim=True)
                else:
                    _, hist_array = output_teacher_prob[0].cpu().data.max(dim=0, keepdim=True)

                writer.add_histogram("val/hist_"+str(hist_ind), hist_array.numpy().flatten(), total_steps_train, hist_bins)  # Use train steps so we can compare with class power plot
                if epoch == start_epoch:
                    writer.add_image("val/hist/input_"+str(hist_ind), image2)  # Visualize image used to compute histogram
                       
        total_steps_val += len(epoch_loss_student_val)
        avg_loss_teacher_val = sum(epoch_loss_teacher_val) / len(epoch_loss_teacher_val)
        print(f'VAL loss_teacher: {avg_loss_teacher_val:0.4} (epoch: {epoch}, step: {total_steps_val})', 
                "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
        writer.add_scalar("val/epoch_loss_student", sum(epoch_loss_student_val) / len(epoch_loss_student_val), total_steps_val)
        writer.add_scalar("val/epoch_loss_teacher", avg_loss_teacher_val, total_steps_val)
        writer.add_scalar("val/epoch_loss_trav_student", sum(epoch_loss_trav_student_val) / len(epoch_loss_trav_student_val), total_steps_val)
        writer.add_scalar("val/epoch_loss_trav_teacher", sum(epoch_loss_trav_teacher_val) / len(epoch_loss_trav_teacher_val), total_steps_val)
        writer.add_scalar("val/epoch_loss_autoenc_student", sum(epoch_loss_autoenc_student_val) / len(epoch_loss_autoenc_student_val), total_steps_val)
        writer.add_scalar("val/epoch_loss_autoenc_teacher", sum(epoch_loss_autoenc_teacher_val) / len(epoch_loss_autoenc_teacher_val), total_steps_val)
        epoch_loss_student_val = []
        epoch_loss_teacher_val = []

        average_epoch_loss_val = avg_loss_teacher_val
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = average_epoch_loss_val
        else:
            current_acc = iouVal 
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
            'arch': str(model_teacher),
            'state_dict': model_teacher.state_dict(),
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
            torch.save(model_teacher.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model_teacher.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model_student, model_teacher)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model_student = model_file.Net(softmax_classes=args.force_n_classes, spread_class_power=args.spread_init, fix_class_power=args.fix_class_power, late_dropout_prob=args.late_dropout_prob)
    model_teacher = model_file.Net(softmax_classes=args.force_n_classes, spread_class_power=args.spread_init, fix_class_power=args.fix_class_power, late_dropout_prob=args.late_dropout_prob)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model_student = torch.nn.DataParallel(model_student).cuda()
        model_teacher = torch.nn.DataParallel(model_teacher).cuda()
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model_student = load_my_state_dict(model_student, torch.load(args.state))
        model_teacher = load_my_state_dict(model_teacher, torch.load(args.state))

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder
    
    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """

    #train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model_student, model_teacher = train(args, model_student, model_teacher, True) #Train encoder
    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
        else:
            pretrainedEnc = next(model_student.children()).encoder
        model_student = model_file.Net( encoder=pretrainedEnc, softmax_classes=args.force_n_classes, spread_class_power=args.spread_init, fix_class_power=args.fix_class_power, late_dropout_prob=args.late_dropout_prob)  #Add decoder to encoder
        model_teacher = model_file.Net( encoder=pretrainedEnc, softmax_classes=args.force_n_classes, spread_class_power=args.spread_init, fix_class_power=args.fix_class_power, late_dropout_prob=args.late_dropout_prob)  #Add decoder to encoder
        if args.cuda:
            def make_cuda(model):
                return torch.nn.DataParallel(model).cuda()

            model_student = make_cuda(model_student)
            model_teacher = make_cuda(model_teacher)
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model_student, model_teacher = train(args, model_student, model_teacher, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet_self_supervised_power")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--pretrained') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--force-n-classes', type=int, default=0)   # Force network to discretize output into classes with discrete output power
    parser.add_argument('--spread-init', action='store_true', default=False)    # Spread initial class power over interval [0.7,...,2.0]
    parser.add_argument('--fix-class-power', action='store_true', default=False)    # Fix class power so that it is not optimized
    parser.add_argument('--late-dropout-prob', type=float, default=0.3)    # Specify dropout prob in last layer after softmax
    parser.add_argument('--alternate-optimization', action='store_true', default=False) # Alternate optimizing class segmentation and class score every epoch
    parser.add_argument('--subsample', type=int, default=1) # Only use every nth image of a dataset.
    parser.add_argument('--no-mean-teacher', action='store_true', default=False)

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=False)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
