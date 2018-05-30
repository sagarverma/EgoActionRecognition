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
from utils.folder import ImagePreloader
import random

use_gpu = torch.cuda.is_available()
DEVICE = 1

#Data statistics
mean = DATA.flow['mean']
std = DATA.flow['std']
num_classes = DATA.flow['num_classes']
class_map = DATA.flow['class_map']

#Training parameters
lr = DATA.flow['lr']
momentum = DATA.flow['momentum']
step_size = DATA.flow['step_size']
gamma = DATA.flow['gamma']
num_epochs = DATA.flow['num_epochs']
batch_size = DATA.flow['batch_size']
data_transforms= DATA.flow['data_transforms']
#Directory names
data_dir = DATA.flow['data_dir']
png_dir = DATA.flow['png_dir']
weights_dir = DATA.flow['weights_dir']

#csv files
train_csv = DATA.flow['train_csv']
test_csv = DATA.flow['test_csv']

class ResNet50Bottom(nn.Module):
    """
    Model definition.
    """
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(7,1)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        x = self.fc(x)
        return x
        
def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    """
        Training model with given criterion, optimizer for num_epochs. 
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True) 
            else:
                model.train(False) 

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda(DEVICE))
                    labels = Variable(labels.cuda(DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / (dataset_sizes[phase] * 1.0)
            epoch_acc = running_corrects / (dataset_sizes[phase] * 1.0)

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


            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model = torch.load(data_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
    
    return model

image_datasets = {'train': ImagePreloader(data_dir + png_dir, data_dir + train_csv, class_map, data_transforms['train']), 
                    'test': ImagePreloader(data_dir + png_dir, data_dir + test_csv, class_map, data_transforms['test'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model_conv = torchvision.models.resnet50(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = True
model_conv = ResNet50Bottom(model_conv)

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)
print (model_conv)
#Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

#Train model
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)