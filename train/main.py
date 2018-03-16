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
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from scipy import ndimage
from numpy.lib.stride_tricks import as_strided

from tensorboardX import SummaryWriter

from dataset import VOC12,cityscapes,self_supervised_power
from transform import Relabel, ToLabel, Colorize, ColorizeMinMax, ColorizeWithProb, ColorizeClasses, FloatToLongLabel, ToFloatLabel, getMaxProbValue
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile

NUM_CHANNELS = 3
# NUM_CLASSES = 20 #pascal=22, cityscapes=20
NUM_SOFTMAX = 20

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
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        # start_time = time.time()

        if(self.augment):
            # Appearance changes.
            rand_val = np.random.randn(4)/4 + 1.0    # Generate rand around 1
            sharpness = ImageEnhance.Sharpness(input)
            color = ImageEnhance.Color(sharpness.enhance(rand_val[0]))
            brightness = ImageEnhance.Brightness(color.enhance(rand_val[1]))
            contrast = ImageEnhance.Contrast(brightness.enhance(rand_val[2]))
            input = contrast.enhance(rand_val[3])

            # Smooth out target. 
            filter_size = 9
            target_array = np.array(target, dtype="float32")
            mask = target_array < 0
            target_filtered = np.copy(target_array)
            target_filtered[mask] = np.nan
            M = target_array.shape[0] - filter_size + 1
            N = target_array.shape[1] - filter_size + 1
            target_strided = as_strided(target_filtered, (M, N, filter_size, filter_size), 2*target_filtered.strides)
            target_strided = target_strided.copy().reshape((M, N, filter_size*filter_size))
            edge_length = int((filter_size-1)/2)
            target_filtered[edge_length:M+edge_length, edge_length:N+edge_length] = np.nanmean(target_strided, axis=2)
            target_filtered[np.isnan(target_filtered)] = -1.0
            target_filtered[mask] = target_array[mask]
            target = Image.fromarray(target_filtered, 'F')

            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill= -1.0) #pad label filling with -1
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

            # Random crop while preserving aspect ratio and divisibility by 8
            while True:
                crop_val = np.random.randint(6)
                # Assumes base image is 480x640
                img_size = np.array([32, 24]) * (20-crop_val)
                hor_pos = int(np.random.rand() * (640 - img_size[0]))
                input_crop = input.crop((hor_pos, 480 - img_size[1], hor_pos + img_size[0], 480))
                # input.show()
                target_crop = target.crop((hor_pos, 480 - img_size[1], hor_pos + img_size[0], 480))
                target_test = np.array(target_crop, dtype="float32")
                # Condition to make sure we have crop containing footprints
                if (img_size[0] == 640 or np.any(target_test > 0.0)):
                    input = input_crop
                    target = target_crop
                    break


        # print("Augmentation took ", time.time() - start_time, " seconds")

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToFloatLabel()(target)
        # print (target)
        # target = Relabel(255, 19)(target)

        return input, target


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
        masked_loss = weighted_loss.masked_select(torch.gt(targets, 0.0))
        return masked_loss.mean()

class L1LossMasked(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.L1Loss(False, False)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets).masked_select(torch.gt(targets, 0.0)).mean()


best_acc = 0

def train(args, model, enc=False):
    global best_acc

    weight = torch.ones(1)
    
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    # Set data loading variables
    co_transform = MyCoTransform(enc, augment=True, height=480)#1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=480)#1024)
    dataset_train = self_supervised_power(args.datadir, co_transform, 'train')
    # dataset_train = self_supervised_power(args.datadir, None, 'train')
    dataset_val = self_supervised_power(args.datadir, co_transform_val, 'val')

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
        myfile.write(str(model))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-6)      ## scheduler 2

    start_epoch = 1

    if args.pretrained:
        pretrained = torch.load(args.pretrained)
        start_epoch = pretrained['epoch']
        model.load_state_dict(pretrained['state_dict'])
        optimizer.load_state_dict(pretrained['optimizer'])
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
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)
        writer = SummaryWriter()
        log_base_dir = writer.file_writer.get_logdir() + "/"
        print("Saving tensorboard log to: " + log_base_dir)
        total_steps_train = 0
        total_steps_val = 0

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        # Create new run in summary writer. 
        if args.split_epoch_vis:
            writer.close()
            writer = SummaryWriter(log_base_dir + "epoch_" + str(epoch))

        average_loss_train = 0
        average_loss_val = 0

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      


        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            if (args.force_n_classes) > 0:
                # Forced into discrete classes. 
                output_prob, output_power = model(inputs, only_encode=enc)
                optimizer.zero_grad()
                loss = criterion(output_prob, output_power, targets)
            else:
                # Straight regressoin
                output = model(inputs, only_encode=enc)
                optimizer.zero_grad()
                loss = criterion(output, targets)

            #print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            # Do backward pass. 
            loss.backward()

            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name, " grad is bad")
            #     else:
            #         print(name, " grad is good")
            #         print(param.grad)

            optimizer.step()

            epoch_loss.append(loss.data[0])
            time_train.append(time.time() - start_time)

            # if (doIouTrain):
            #     #start_time_iou = time.time()
            #     iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
            #     #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            #print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                if args.split_epoch_vis:
                    step_vis_no = step
                else:
                    step_vis_no = total_steps_train + len(epoch_loss)

                # Figure out and compute tensor to visualize. 
                if args.force_n_classes > 0:
                    if (isinstance(output_prob, list)):
                        vis_output = getMaxProbValue(output_prob[0][0].cpu().data, output_power[0][0].cpu().data)
                        writer.add_image("train/classes", color_transform_classes(output_prob[0][0].cpu().data), step_vis_no)
                    else:
                        vis_output = getMaxProbValue(output_prob[0].cpu().data, output_power[0].cpu().data)
                        writer.add_image("train/classes", color_transform_classes(output_prob[0].cpu().data), step_vis_no)
                else:
                    if (isinstance(output, list)):
                        vis_output = output[0][0].cpu().data
                    else:
                        vis_output = output[0].cpu().data

                start_time_plot = time.time()
                image = inputs[0].cpu().data
                # board.image(image, f'input (epoch: {epoch}, step: {step})')
                writer.add_image("train/input", image, step_vis_no)

                writer.add_image("train/output", color_transform_output(vis_output), step_vis_no)
                # board.image(color_transform_target(targets[0].cpu().data),
                #     f'target (epoch: {epoch}, step: {step})')
                writer.add_image("train/target", color_transform_target(targets[0].cpu().data), step_vis_no)
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                len_epoch_loss = len(epoch_loss)
                average_loss_train = (average_loss_train * step + sum(epoch_loss)) / (step + len_epoch_loss)
                for ind, val in enumerate(epoch_loss):
                    writer.add_scalar("train/instant_loss", val, total_steps_train + ind)
                total_steps_train += len_epoch_loss
                writer.add_scalar("train/average_loss", average_loss_train, total_steps_train)
                # Clear loss for next loss print iteration.
                # Output class power costs
                power_dict = {}
                if args.force_n_classes > 0:
                    for ind, val in enumerate(output_power.squeeze()):
                        power_dict[str(ind)] = val
                    writer.add_scalars("params/class_cost", power_dict, total_steps_train)
                epoch_loss = []
                # Print current loss. 
                print(f'loss: {average_loss_train:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []


        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)

            if args.force_n_classes:
                output_prob, output_power = model(inputs, only_encode=enc) 
                output = getMaxProbValue(output_prob, output_power)
            else:
                output = model(inputs, only_encode=enc)

            loss = criterion_val(output, targets)
            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)


            #Add batch to calculate TP, FP and FN for iou estimation
            # if (doIouVal):
            #     #start_time_iou = time.time()
            #     iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
            #     #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                if args.split_epoch_vis:
                    step_vis_no = step
                else:
                    step_vis_no = total_steps_val + len(epoch_loss_val)
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                # board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                writer.add_image("val/input", image, step_vis_no)
                if isinstance(output, list):   #merge gpu tensors
                    # board.image(color_transform_output(outputs[0][0].cpu().data),
                    # f'VAL output (epoch: {epoch}, step: {step})')
                    writer.add_image("val/output", color_transform_output(output[0][0].cpu().data), step_vis_no)
                    if args.force_n_classes > 0:
                        writer.add_image("val/classes", color_transform_classes(output_prob[0][0].cpu().data), step_vis_no)
                else:
                    # board.image(color_transform_output(outputs[0].cpu().data),
                    # f'VAL output (epoch: {epoch}, step: {step})')
                    writer.add_image("val/output", color_transform_output(output[0].cpu().data), step_vis_no)
                    if args.force_n_classes > 0:
                        writer.add_image("val/classes", color_transform_classes(output_prob[0].cpu().data), step_vis_no)
                # board.image(color_transform_target(targets[0].cpu().data),
                #     f'VAL target (epoch: {epoch}, step: {step})')
                writer.add_image("val/target", color_transform_target(targets[0].cpu().data), step_vis_no)
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                len_epoch_loss_val = len(epoch_loss_val)
                average_loss_val = (average_loss_val * step + sum(epoch_loss_val)) / (step + len_epoch_loss_val)
                total_steps_val += len_epoch_loss_val
                epoch_loss_val = []
                       
        print(f'VAL loss: {average_loss_val:0.4} (epoch: {epoch}, step: {total_steps_val})', 
                "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
        writer.add_scalar("val/average_loss", average_loss_val, total_steps_val)

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
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
            'arch': str(model),
            'state_dict': model.state_dict(),
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
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
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
    
    return(model)   #return model (convenience for encoder-decoder training)

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
    model = model_file.Net(softmax_classes=args.force_n_classes, spread_class_power=args.spread_init, fix_class_power=args.fix_class_power)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
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
        model = load_my_state_dict(model, torch.load(args.state))

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
        model = train(args, model, True) #Train encoder
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
            pretrainedEnc = next(model.children()).encoder
        model = model_file.Net( encoder=pretrainedEnc, softmax_classes=args.force_n_classes, spread_class_power=args.spread_init, fix_class_power=args.fix_class_power)  #Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)   #Train decoder
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
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--pretrained') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--force-n-classes', type=int, default=0)   # Force network to discretize output into classes with discrete output power
    parser.add_argument('--split-epoch-vis', action='store_true', default=False)    # Split tensorboard output by epoch
    parser.add_argument('--spread-init', action='store_true', default=False)    # Spread initial class power over interval [0.7,...,2.0]
    parser.add_argument('--fix-class-power', action='store_true', default=False)    # Fix class power so that it is not optimized

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=False)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
