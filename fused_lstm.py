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
from utils.folder import BiModeSequencePreloader, BiModeNpySequencePreloader

DEVICE = 1

#Data statistics
rgb_mean = DATA.rgb['mean']
rgb_std = DATA.rgb['std']
flow_mean = DATA.flow['mean']
flow_std = DATA.flow['std']
num_classes = DATA.flow_lstm['num_classes']
class_map = DATA.flow_lstm['class_map']

#Training parameters
lr = DATA.flow_lstm['lr']
momentum = DATA.flow_lstm['momentum']
step_size = DATA.flow_lstm['step_size']
gamma = DATA.flow_lstm['gamma']
num_epochs = DATA.flow_lstm['num_epochs']
batch_size = DATA.flow_lstm['batch_size']
sequence_length = DATA.flow_lstm['sequence_length']

#Directory names
data_dir = DATA.flow['data_dir']
flow_png_dir = DATA.flow['png_dir']
rgb_png_dir = DATA.rgb['png_dir']
weights_dir = DATA.flow['weights_dir']
flow_features_2048_dir = DATA.flow_lstm['features_2048_dir']
rgb_features_2048_dir = DATA.rgb_lstm['features_2048_dir']

#csv files
train_csv = DATA.flow_lstm['train_csv']
test_csv = DATA.flow_lstm['test_csv']

class RGBResNet50Bottom(nn.Module):
    """
        Model definition.
    """
    def __init__(self, original_model):
        super(RGBResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(10,1)

    def forward(self, x):
        x = x.view(-1, 3, 300, 300)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        return x

class FlowResNet50Bottom(nn.Module):
    """
        Model definition.
    """
    def __init__(self, original_model):
        super(FlowResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(7,1)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        return x

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.rgb_rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.flow_rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.out = nn.Linear(hidden_size + hidden_size, num_classes)

    def forward(self, rgb_inp, flow_inp):
        x_rgb = self.rgb_rnn(rgb_inp)[0]
        x_flow = self.flow_rnn(flow_inp)[0]


        fused_outputs = torch.cat([x_rgb, x_flow], dim=2)
        outputs = self.out(fused_outputs)

        outputs_mean = Variable(torch.zeros(outputs.size()[0], num_classes)).cuda(DEVICE)
        for i in range(outputs.size()[0]):
            outputs_mean[i] = outputs[i].mean(dim=0)

        return outputs_mean

class Net(nn.Module):
    def __init__(self, rgb_model_conv, flow_model_conv, input_size, hidden_size):
        super(Net, self).__init__()
        self.rgbResnet50Bottom = RGBResNet50Bottom(rgb_model_conv)
        self.flowResNet50Bottom = FlowResNet50Bottom(flow_model_conv)
        self.lstmNet = LSTMNet(input_size, hidden_size)

    def forward(self, rgb_inp, flow_inp, phase):
        if phase == 'train':
            rgb_features = []
            flow_features = []

            batch_size = rgb_inp.size()[0]
            sequence_length = rgb_inp.size()[1]

            rgb_inp = rgb_inp.view(-1, 3, 300, 300)
            flow_inp = flow_inp.view(-1, 3, 224, 224)
            for i in range(0, rgb_inp.size()[0], 128):
                rgb_features.append(self.rgbResnet50Bottom(rgb_inp[i:i+128]))
                flow_features.append(self.flowResNet50Bottom(flow_inp[i:i+128]))

            rgb_feature_sequence = torch.cat(rgb_features, dim=0)
            flow_feature_sequence = torch.cat(flow_features, dim=0)

            rgb_feature_sequence = rgb_feature_sequence.view(batch_size, sequence_length, 2048)
            flow_feature_sequence = flow_feature_sequence.view(batch_size, sequence_length, 2048)

            outputs = self.lstmNet(rgb_feature_sequence, flow_feature_sequence)
            return outputs
        else:
            outputs = self.lstmNet(rgb_inp, flow_inp)
            return outputs

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
                rgb_inputs, flow_inputs, labels = data

                rgb_inputs = Variable(rgb_inputs.cuda(DEVICE))
                flow_inputs = Variable(flow_inputs.cuda(DEVICE))
                labels = Variable(labels.cuda(DEVICE))

                optimizer.zero_grad()
                outputs = model(rgb_inputs, flow_inputs, phase)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)[0].cpu().numpy()

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
                print ('saving model.....')
                torch.save(model, weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_seq_length_' + str(sequence_length) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
        print()

    time_elapsed = time.time() - sinceview(1,-1)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model = torch.load( weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_seq_length_' + str(sequence_length) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')

    return model

#Dataload and generator initialization
rgb_preloader_param = {'train': [data_dir + rgb_png_dir, data_dir + train_csv, rgb_mean, rgb_std, [280, 450], [224, 224], 300], \
                       'test': [data_dir + rgb_features_2048_dir, data_dir + flow_features_2048_dir, data_dir + test_csv]}
flow_preloader_param = {'train': [data_dir + flow_png_dir, data_dir + train_csv, flow_mean, flow_std, [405, 720], [224, 224], 300], \
                        'test': [data_dir + flow_features_2048_dir, data_dir + test_csv]}

sequence_datasets = {'train': BiModeSequencePreloader(rgb_preloader_param['train'], flow_preloader_param['train']), \
                    'test': BiModeNpySequencePreloader(flow_preloader_param['test'], flow_preloader_param['test'])}

dataloaders = {x: torch.utils.data.DataLoader(sequence_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6) for x in ['train', 'test']}
dataset_sizes = {x: len(sequence_datasets[x]) for x in ['train', 'test']}


file_name = __file__.split('/')[-1].split('.')[0]

ResNet50Bottom = RGBResNet50Bottom
rgb_model_conv = torch.load(weights_dir + 'weights_rgb_cnn_lr_0.001_momentum_0.9_step_size_15_gamma_1_num_classes_10_batch_size_128.pt')

ResNet50Bottom = FlowResNet50Bottom
flow_model_conv = torch.load(weights_dir + 'weights_flow_cnn_lr_0.001_momentum_0.9_step_size_3_gamma_0.1_num_classes_11_batch_size_64.pt')

#print(model_conv)
for param in rgb_model_conv.parameters():
    param.requires_grad = False

for param in flow_model_conv.parameters():
    param.requires_grad = False

hidden_size = 256
input_size = 2048
model = Net(rgb_model_conv, flow_model_conv, input_size, hidden_size)
print (model)
model = model.cuda(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.lstmNet.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
