import sys
sys.path.append('.')

import shutil, time,  os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.transforms import functional

import torchvision
from torchvision import datasets, models, transforms

from config import GTEA as DATA

from utils.folder import ImagePreloader, pil_loader

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
features_2048_dir = DATA.rgb_lstm['features_2048_dir']
weights_dir = DATA.rgb['weights_dir']

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
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        return x

class Net(nn.Module):
    def __init__(self, model_conv):
        super(Net, self).__init__()
        self.resnet50Bottom = ResNet50Bottom(model_conv)

    def forward(self, inp):
            outputs = self.resnet50Bottom(inp)
            return outputs

model_conv = torch.load(weights_dir + 'weights_rgb_cnn_lr_0.001_momentum_0.9_step_size_15_gamma_1_num_classes_10_batch_size_128.pt')
for param in model_conv.parameters():
    param.requires_grad = False

model = Net(model_conv)
model = model.cuda()

videos = os.listdir(data_dir + png_dir)
for video in videos:
    images = os.listdir(data_dir + png_dir + video)

    for image in images:
        img = pil_loader(data_dir + png_dir + video + '/' + image)
        img = functional.center_crop(img, (224, 224))
        img = functional.resize(img, 300)
        tensor = functional.to_tensor(img)
        tensor = functional.normalize(tensor, mean, std)
        tensor = torch.stack([tensor]).cuda()
        feature = model(tensor)
        feature = feature.view(2048)
        feature = ((feature).data).cpu().numpy()

        path = data_dir + features_2048_dir + video + '/' + image[:-4] + '.npy'
        np.save(path, feature)
