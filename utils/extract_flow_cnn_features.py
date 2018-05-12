import sys
sys.path.insert(0, '/home/shubham/GTEA/codes/ego_action_recognition/')
import shutil
import torch
from os import listdir
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
from utils.folder import VideoPreloader
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
features_2048_dir = DATA.flow['features_2048_dir']
weights_dir = DATA.flow['weights_dir']
plots_dir = DATA.flow['plots_dir']

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

    def forward(self, x):
        #print (x.size())
        #x = x.view(-1, 3, 300, 300)
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
        
    
videos=listdir(data_dir+png_dir)
for video in videos:
    image_datasets=VideoPreloader(data_dir+png_dir, video, class_map, data_transforms['test'])
    #dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    size_train=len(image_datasets)
    #print (np.shape(image_datasets[0][0]), (image_datasets[0][1]))
    use_gpu = torch.cuda.is_available()
    file_name = __file__.split('/')[-1].split('.')[0]
    model_conv = torch.load(data_dir + weights_dir + 'weights_flow_cnn_lr_0.001_momentum_0.9_step_size_200_gamma_0.1_num_classes_11_batch_size_64.pt')
    for param in model_conv.parameters():
        param.requires_grad = False
        
    model = Net(model_conv)
    model = model.cuda()
    batch_size_temp=1
    video_feature=[]
    for i in range(0,size_train,batch_size_temp):
        batch_img=[]
        for j in range(batch_size_temp):
            batch_img.append(image_datasets[i+j][0])
        batch_img = torch.stack(batch_img)
        batch_img=Variable(batch_img.cuda())
        feature_vec=model(batch_img)
        feature_vec=feature_vec.view(2048)
        feature_vec_numpy=((feature_vec).data).cpu().numpy()
        video_feature.append(feature_vec_numpy)

       
    video_feature=np.asarray(video_feature)
    print (np.shape(video_feature))  
    path=data_dir+features_2048_dir+video+'.npy'
    print (path)
    np.save(path, video_feature)
