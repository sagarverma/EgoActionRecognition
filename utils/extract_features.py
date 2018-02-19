
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
from torchvision.transforms import functional
import matplotlib.pyplot as plt
import time
import os
from config import GTEA as DATA
from folder import ImagePreloader
from os import listdir
from PIL import Image
import random

mean = DATA.rgb['mean']
std = DATA.rgb['std']
lr = DATA.rgb['lr']
momentum = DATA.rgb['momentum']
step_size = DATA.rgb['step_size']
gamma = DATA.rgb['gamma']
num_epochs = DATA.rgb['num_epochs']
data_dir = DATA.rgb['data_dir']
train_csv = DATA.rgb['train_csv']
val_csv = DATA.rgb['val_csv']
num_classes = DATA.rgb['num_classes']
batch_size = DATA.rgb['batch_size']
weights_dir = DATA.rgb['weights_dir']
plots_dir = DATA.rgb['plots_dir']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
        
class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(10,1)
        self.fc = nn.Linear(2048, 10)
        #self.dropout= nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = self.avg_pool(x)
        #print(x.size())
        x = x.view(-1, 2048)
        #x = self.fc(x)
        #x=self.dropout(x)
        return x

transofrm = transforms.Compose([
        transforms.CenterCrop([280,450]),
        transforms.CenterCrop(224),
        transforms.Resize(300),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

if __name__ == '__main__':
    model_conv = torchvision.models.resnet50(pretrained=True)
    model_conv = ResNet50Bottom(model_conv)
    model_conv=(torch.load('/home/shubham/Egocentric/weights/weights_modified_resnet_50_fine_tune_gtea_10_classes_resized_to_300_by_400_lr_001.pt'))
    
    model_conv = model_conv.cuda()
    
    root = '/home/shubham/Egocentric/dataset/GTea/pngs/'
    out = '/home/shubham/Egocentric/dataset/GTea/rgb_2048_features/'
    videos = listdir(root)
    
    file_names = []
    
    
    for video in videos:
        if 'S4' not  in video:
            images = listdir(root + video)
    
            for image in images:
                file_names.append(root + video + '/' + image)
    
    for a in range(10,20):
        irand = random.randint(0, 280 - 224)
        jrand = random.randint(0, 450 - 224)
        flip = random.random()
        for i in range(0,len(file_names),512):
            
            batch = []
            batch_fn = []
            
            for image in file_names[i:i+512]:
                img = Image.open(image)
                img = img.convert('RGB')
                img = functional.center_crop(img, (280, 450))
                #random crop
                img = functional.crop(img, irand, jrand, 224, 224)
                #resize
                img = functional.resize(img, 300)
                #horizontal flip
                if flip < 0.5:
                    img = functional.hflip(img)
                #to_tensor
                tensor = functional.to_tensor(img)
                #normalize
                tensor = functional.normalize(tensor, mean, std)
                batch.append(tensor)
                batch_fn.append(image)
            
            batch = Variable(torch.stack(batch).cuda())
            outputs = model_conv(batch)
            outputs = outputs.data.cpu().numpy()
            print(outputs.shape, a)
            
            for j in range(len(batch_fn)):
                np.save(out + batch_fn[j].split('/')[-2] + '/' + batch_fn[j].split('/')[-1][:-4] + '_' + str(a).zfill(2) + '_.npy', outputs[j])
        