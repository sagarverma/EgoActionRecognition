# -*- coding: utf-8 -*-
"""
Transfer Learning tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train your network using
transfer learning. You can read more about the transfer learning at `cs231n
notes <http://cs231n.github.io/transfer-learning/>`__

Quoting this notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios looks as follows:

-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from config import GTEA as DATA
from folder import ImagePreloader
import random
import cv2

mean = DATA.flow['mean']
std = DATA.flow['std']
lr = DATA.flow['lr']
momentum = DATA.flow['momentum']
step_size = DATA.flow['step_size']
gamma = DATA.flow['gamma']
num_epochs = DATA.flow['num_epochs']
data_dir = DATA.flow['data_dir']
train_csv = DATA.flow['train_csv']
test_csv = DATA.flow['test_csv']
num_classes = DATA.flow['num_classes']
batch_size = DATA.flow['batch_size']
weights_dir = DATA.flow['weights_dir']
plots_dir = DATA.flow['plots_dir']

#plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 testidation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for testidation

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, test in enumerate(images):                                          
        weight[idx] = weight_per_class[test[1]]                                  
    return weight

class HumaraFlipJoNaHogaFlop(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        
        if random.random() < 0.5:   
            img = np.asarray(img)
            img = img[:, ::-1, :]  # flip for mirrors
            rgb = cv2.merge([255-img[:,:,0],img[:,:,1],img[:,:,2]])
            
            return rgb
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

transformer = io.Transformer({'data': (1, 3, 300, 300)})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([128, 128, 128]))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_is_flow('data', True)

trans = io.Transformer({'data': (1, 3, 300, 300)})
trans.set_transpose('data', (2, 0, 1))
trans.set_mean('data', np.array([128, 128, 128]))
trans.set_raw_scale('data', 255)
trans.set_channel_swap('data', (2, 1, 0))

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(300),
        transformer.preprocess(),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(300),
        trans.preprocess(),
        transforms.ToTensor()
    ]),
}


image_datasets = {'train': ImagePreloader(data_dir + 'FusionSeg_flow/', data_dir + train_csv, data_transforms['train'], loader=load_image_for_flow), 'test': ImagePreloader(data_dir + 'FusionSeg_flow/', data_dir + test_csv, data_transforms['test'], loaded=load_image_for_flow)}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


#second method
# data_dir = '/home/shubham/Egocentric/dataset/temp/GTea_preprocessed_rgb_10_actions_ctrl_crop_280_450'

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}
# print (image_datasets)
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
#                                              shuffle=True, num_workers=4)
# #print (dataloaders)
#               for x in ['train', 'test']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
# class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

######################################################################

        
def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    ##------my code------------##
    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []
    test_loss = 100
    ##--------finish--------------##
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and testidation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to etestuate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))
            
            ##----------my code --------------##
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                print ('##############################################################')
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))                
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                print (" {} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
                print ('##############################################################')

       
            plt.plot(range(len(train_loss)),train_loss,label="Train")
            plt.plot(range(len(test_loss)),test_loss,label="Test")
            plt.legend(bbox_to_anchor=(.90, 1), loc=2, borderaxespad=0.)
            plt.savefig(data_dir + plots_dir + file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) +"_loss.png")
            plt.cla()
            plt.clf()
            plt.plot(range(len(train_acc)),train_acc,label="Train")
            plt.plot(range(len(test_acc)),test_acc,label="Test")
            plt.legend(bbox_to_anchor=(.90, 1), loc=2, borderaxespad=0.)
            plt.savefig(data_dir + plots_dir + file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) +"_acc.png")
            plt.cla()
            plt.clf()


            ##--------- my code finish--------------##
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = model.state_dict()
                torch.save(model, data_dir + weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
                torch.save(model.state_dict() , data_dir + weights_dir + 'weights_state_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
            

    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best test Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # #model.load_state_dict(best_model_wts)
    # model = torch.load('weights_'+file_name+'_lr_001.pt')
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(7,1)
        self.fc = nn.Linear(2048, num_classes)
        #self.dropout= nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = self.avg_pool(x)
        #print(x.size())
        x = x.view(-1, 2048)
        x = self.fc(x)
        #x=self.dropout(x)
        return x

if __name__ == '__main__':

    file_name = __file__.split('/')[-1].split('.')[0]
    print (file_name)
    model_conv = torchvision.models.resnet50(pretrained=True)
    #for param in model_conv.parameters():
    #    param.requires_grad = False
    model_conv = ResNet50Bottom(model_conv)
    print(model_conv)
    
    # Parameters of newly constructed modules have requires_grad=True by default
    #num_ftrs = model_conv.fc.in_features
    #model_conv.fc = nn.Linear(num_ftrs, 10)
    
    
    if use_gpu:
        model_conv = model_conv.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=momentum)
    #optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.000001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv)


    ######################################################################
    # Train and etestuate
    # ^^^^^^^^^^^^^^^^^^
    #
    # On CPU this will take about half the time compared to previous scenario.
    # This is expected as gradients don't need to be computed for most of the
    # network. However, forward does need to be computed.
    #

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=num_epochs)

    ######################################################################
    #
    visualize_model(model_conv)
    plt.ioff()
    plt.show()

