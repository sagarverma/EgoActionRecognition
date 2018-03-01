from __future__ import print_function, division
import sys
sys.path.append('../')
sys.path.append('../utils')
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
from utils.folder import SequencePreloader, NpySequencePreloader
import csv

#Directory names
data_dir = DATA.rgb['data_dir']
png_dir = DATA.rgb['png_dir']
features_2048_dir = DATA.rgb_lstm['features_2048_dir']
weights_dir = DATA.rgb['weights_dir']
plots_dir = DATA.rgb['plots_dir']


class_map = DATA.rgb_lstm['class_map']

class ResNet50Bottom(nn.Module):
    """
        Model definition.
    """
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(10,1)

    def forward(self, x):
        x = x.view(-1, 3, 300, 300)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, sequence_length, 2048)
        return x
        
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.out = nn.Linear(hidden_size, 11)
    
    def forward(self, inp):
        x = self.rnn(inp)[0]
        x = x.permute(1,0,2)
        
        outputs = []
        for i in range(x.size()[0]):
            outputs.append(np.argmax(self.out(x[i]).data.cpu().numpy()))
            
        return outputs

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.lstmNet = LSTMNet(input_size, hidden_size)
        
    def forward(self, inp):
        outputs = self.lstmNet(inp)
        return outputs
        
        
model = Net(2048, 512)
model = torch.load('../../Egocentric/dataset/GTEA/weights/RGB/weights_rgb_lstm_lr_0.001_momentum_0.9_step_size_20_gamma_0.1_seq_length_11_num_classes_11_batch_size_32.pt')


videos = os.listdir(data_dir + features_2048_dir)

accuracy = 0
tot = 0
for video in videos:
    if 'S4' in video:
        features = os.listdir(data_dir + features_2048_dir + video)
        features.sort()

        inp = []
        for feature in features:
            npy = np.load(data_dir + features_2048_dir + video + '/' + feature)
            inp.append(npy)
            
        inp = Variable(torch.from_numpy(np.asarray([inp])).cuda())
        output = model(inp)
    
        r = csv.reader(open(data_dir + 'gtea_labels_cleaned/' + video + '.txt', 'r'), delimiter = ' ')
    
        ground_truth = [0 for x in range(len(output))]
        for row in r:
            if len(row) > 0:
                for i in range(int(row[2])-1, int(row[3])):
                    ground_truth[i] = class_map[row[0]]
                
        
        for i in range(len(output)):
            if output[i] == ground_truth[i]:
                accuracy += 1
            tot += 1
            

print(accuracy/(tot * 1.0))
                
