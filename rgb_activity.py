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
from utils.folder import NpyVideoPreloader

#Data statistics
mean = DATA.rgb['mean']
std = DATA.rgb['std']
num_classes = DATA.rgb_activity['num_classes']
class_map = DATA.rgb_activity['class_map']

#Training parameters
lr = DATA.rgb_activity['lr']
momentum = DATA.rgb_activity['momentum']
step_size = DATA.rgb_activity['step_size']
gamma = DATA.rgb_activity['gamma']
num_epochs = DATA.rgb_activity['num_epochs']
batch_size = DATA.rgb_activity['batch_size']
data_transforms= DATA.rgb['data_transforms']

#Directory names
data_dir = DATA.rgb['data_dir']
png_dir = DATA.rgb['png_dir']
weights_dir = DATA.rgb['weights_dir']
plots_dir = DATA.rgb['plots_dir']
features_2048_dir = DATA.rgb_activity['features_2048_dir']

#csv files
train_csv = DATA.rgb_activity['train_csv']
test_csv = DATA.rgb_activity['test_csv']

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
        #self.out = nn.Linear(hidden_size, 11)
    
    def forward(self, inp):
        x = self.rnn(inp)[0]
        #x = x.permute(1,0,2)
        
        #outputs = []
        #for i in range(x.size()[0]):
        #    outputs.append(np.argmax(self.out(x[i]).data.cpu().numpy()))
            
        return x

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.lstmNet = model_action.lstmNet
        
    def forward(self, inp):
        outputs = self.lstmNet(inp)
        return outputs
        
class ActivityNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ActivityNet, self).__init__()
        self.activityrnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)
        
    def forward(self, actions):
        activities = self.activityrnn(actions)[0]
        activities = activities.permute(1,0,2)
        
        outputs = Variable(torch.zeros(activities.size()[0], activities.size()[1], num_classes)).cuda()
        for i in range(activities.size()[0]):
            outputs[i] = self.out(activities[i])
            
        outputs = outputs.mean(0)
        return outputs
        
        
class Net2(nn.Module):
    def __init__(self, model_action, input_size, hidden_size):
        super(Net2, self).__init__()
        self.actionNet = Net(model_action)
        self.activityNet = ActivityNet(input_size, hidden_size)
        
    def forward(self, inp):
        actions = self.actionNet(inp)
        activity = self.activityNet(actions)
        
        return activity
        
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
    
    for epoch in range(0,num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to etestuate mode

            running_loss = 0.0
            running_corrects = 0
            
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

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
                print ('saving model.....')
                torch.save(model, data_dir + weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
        print()

    time_elapsed = time.time() - sinceview(1,-1)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    
    model = torch.load(data_dir + weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
    

    

#Dataload and generator initialization
video_datasets = {'train': NpyVideoPreloader(data_dir, features_2048_dir, train_csv, class_map), 'test': NpyVideoPreloader(data_dir, features_2048_dir, test_csv, class_map)}
dataloaders = {x: torch.utils.data.DataLoader(video_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6) for x in ['train', 'test']}
dataset_sizes = {x: len(video_datasets[x]) for x in ['train', 'test']}


use_gpu = torch.cuda.is_available()

file_name = __file__.split('/')[-1].split('.')[0]


hidden_size = 256
input_size = 512
model_action = torch.load(data_dir + weights_dir + 'weights_rgb_lstm_lr_0.001_momentum_0.9_step_size_20_gamma_0.1_seq_length_11_num_classes_11_batch_size_32.pt')

model = Net2(model_action, input_size, hidden_size)
print(model)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.activityNet.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

        
        