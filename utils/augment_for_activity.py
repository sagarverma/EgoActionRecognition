
from __future__ import print_function, division

import sys
sys.path.append('/home/shubham/ego_action_recognition/')

from os import listdir, path, mkdir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import functional

import numpy as np

from config import GTEA as DATA
from folder import ImagePreloader, pil_loader

mean = DATA.rgb['mean']
std = DATA.rgb['std']
lr = DATA.rgb['lr']
momentum = DATA.rgb['momentum']
step_size = DATA.rgb['step_size']
gamma = DATA.rgb['gamma']
num_epochs = DATA.rgb['num_epochs']
data_dir = DATA.rgb['data_dir']
png_dir = DATA.rgb['png_dir']
features_2048_dir = DATA.rgb['features_2048_dir']
train_csv = DATA.rgb['train_csv']
test_csv = DATA.rgb['test_csv']
num_classes = DATA.rgb['num_classes']
batch_size = DATA.rgb['batch_size']
weights_dir = DATA.rgb['weights_dir']
plots_dir = DATA.rgb['plots_dir']
class_map = DATA.rgb['class_map']
data_transforms = DATA.rgb['data_transforms']


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(10,1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        return x

if __name__ == '__main__':
    model_conv = torchvision.models.resnet50(pretrained=True)
    model_conv = ResNet50Bottom(model_conv)
    model_conv = torch.load(data_dir + weights_dir + 'weights_resnet_50_lr_0.001_momentum_0.9_step_size_7_gamma_1_num_classes_10_batch_size_128.pt')
    model_conv = model_conv.cuda()
    
    trf = data_transforms['test']
    trf = data_
    videos = listdir(data_dir + png_dir)
    
    for video in videos:
        if 'S4' not in video:
            
            if not path.exists(data_dir + features_2048_dir + video):
                mkdir(data_dir + features_2048_dir + video)
            
            images = listdir(data_dir + png_dir + video)
    
            for image in images:
                img = pil_loader(data_dir + png_dir + video + '/' + image)
                inp = torch.stack([trf(img)])            
                inp = Variable(inp.cuda())
            
                output = model_conv(inp)
                output = output.data.cpu().numpy()[0]
            
                np.save(data_dir + features_2048_dir + video + '/' + image[:-4] + '.npy', output)
            