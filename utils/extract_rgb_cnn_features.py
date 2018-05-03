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
import torch.utils.data as data
import torch.nn.functional as F
from config import GTEA as DATA 
from utils.folder import ImagePreloader

#Data statistics
mean = DATA.rgb['mean']
std = DATA.rgb['std']
num_classes = DATA.rgb['num_classes']
class_map = DATA.rgb['class_map']

#Training parameters
lr = DATA.rgb['lr']
momentum = DATA.rgb['momentum']
step_size = DATA.rgb['step_size']
gamma = DATA.rgb['gamma']
num_epochs = DATA.rgb['num_epochs']
batch_size = DATA.rgb['batch_size']
data_transforms= DATA.rgb['data_transforms']

#Directory names
data_dir = DATA.rgb['data_dir']
png_dir = DATA.rgb['png_dir']
features_2048_dir = DATA.rgb['features_2048_dir']
weights_dir = DATA.rgb['weights_dir']
plots_dir = DATA.rgb['plots_dir']

#csv files
train_csv = DATA.rgb['train_csv']
test_csv = DATA.rgb['test_csv']

class ResNet50Bottom(nn.Module):
    """
        Model definition.
    """
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(10,1)

    def forward(self, x):
        #print (x.size())
        x = x.view(-1, 3, 300, 300)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        return x

class Net(nn.Module):
    def __init__(self, model_conv):
        super(Net, self).__init__()
        self.resnet50Bottom = ResNet50Bottom(model_conv)
        #self.lstmNet = LSTMNet(input_size, hidden_size)
        
    def forward(self, inp):
            outputs = self.resnet50Bottom(inp)
            #outputs = self.lstmNet(feature_sequence)
            return outputs
        
    
#Dataload and generator initialization
image_datasets = {'train': ImagePreloader(data_dir + 'pngs/', data_dir + train_csv, class_map, data_transforms['train']),
                     'test': ImagePreloader(data_dir + 'pngs/', data_dir + test_csv, class_map, data_transforms['test'])}
#print ((image_datasets['train'][0][2]))
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
size_train= (dataset_sizes['train'])
size_test= dataset_sizes['test']
print (size_train)
print (size_test)

class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

file_name = __file__.split('/')[-1].split('.')[0]

model_conv = torch.load(data_dir + weights_dir + 'weights_resnet_50_lr_0.001_momentum_0.9_step_size_7_gamma_1_num_classes_10_batch_size_128.pt')
for param in model_conv.parameters():
    param.requires_grad = False
    
#hidden_size = 512
#input_size = 2048
model = Net(model_conv)
#print (model)

model = model.cuda()
batch_size_temp=1
for i in range(0,size_train,batch_size_temp):
    batch_img=[]

    for j in range(batch_size_temp):
       batch_img.append(image_datasets['train'][i+j][0])
            
    batch_img = torch.stack(batch_img)
    batch_img=Variable(batch_img.cuda())
    feature_vec=model(batch_img)
    feature_vec_numpy=((feature_vec).data).cpu().numpy()
    print (np.shape(feature_vec_numpy))
    
    
    for j in range(batch_size_temp):
        path=(image_datasets['train'][i+j][2])
        path=path.replace('pngs','cnn_features')
        path=path.replace('png','npy')
        print (path)
        temp=feature_vec_numpy[j,:]
        np.save(path, temp)
        
    
    
for i in range(0,size_test,batch_size_temp):
    batch_img=[]

    for j in range(batch_size_temp):
       batch_img.append(image_datasets['test'][i+j][0])
            
    batch_img = torch.stack(batch_img)
    batch_img=Variable(batch_img.cuda())
    feature_vec=model(batch_img)
    feature_vec_numpy=((feature_vec).data).cpu().numpy()
    #print (np.shape(feature_vec_numpy))
    
    
    for j in range(batch_size_temp):
        path=(image_datasets['test'][i+j][2])
        path=path.replace('pngs','cnn_features')
        path=path.replace('png','npy')
        print (path)
        temp=feature_vec_numpy[j,:]
        np.save(path, temp)


#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.lstmNet.parameters(), lr=lr, momentum=momentum)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
