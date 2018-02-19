
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
from folder2 import SequencePreloader, NpySequencePreloader

mean = DATA.rgb['mean']
std = DATA.rgb['std']
lr = DATA.rgb_lstm['lr']
momentum = DATA.rgb_lstm['momentum']
step_size = DATA.rgb_lstm['step_size']
gamma = DATA.rgb_lstm['gamma']
num_epochs = DATA.rgb_lstm['num_epochs']
data_dir = DATA.rgb_lstm['data_dir']
train_csv = DATA.rgb_lstm['train_csv']
val_csv = DATA.rgb_lstm['val_csv']
num_classes = DATA.rgb_lstm['num_classes']
batch_size = DATA.rgb_lstm['batch_size']
weights_dir = DATA.rgb_lstm['weights_dir']
plots_dir = DATA.rgb_lstm['plots_dir']
sequence_length = DATA.rgb_lstm['sequence_length']


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.Bottom = nn.Sequential(*list(original_model.children())[:-2])
        self.avg_pool = nn.AvgPool2d(7,1)
        
    def forward(self, x):
        x = x.view(-1,3,224,224)
        x = self.Bottom(x)
        x = self.avg_pool(x)
        x = x.view(-1, sequence_length, 2048)
        return x

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)
    
    def forward(self, inp):
        x = self.rnn(inp)[0]
        x = x.permute(1,0,2)
        
        outputs = Variable(torch.zeros(x.size()[0], x.size()[1], num_classes)).cuda()
        for i in range(sequence_length):
            outputs[i] = self.out(x[i])
         
        outputs = outputs.mean(0)
        #print(outputs.size())
        return outputs

class Net(nn.Module):
    def __init__(self, model_conv, input_size, hidden_size):
        super(Net, self).__init__()
        self.resnet50Bottom = ResNet50Bottom(model_conv)
        self.lstmNet = LSTMNet(input_size, hidden_size)
        
    def forward(self, inp, phase):
        feature_sequence = self.resnet50Bottom(inp)
        outputs = self.lstmNet(feature_sequence)
        return outputs
        
sequence_datasets = {'train': SequencePreloader(mean, std, 'train', data_dir, train_csv), 'val': SequencePreloader(mean, std, 'val', data_dir, val_csv)}
#sequence_datasets = {'train': NpySequencePreloader(data_dir, train_csv), 'val': NpySequencePreloader(data_dir, val_csv)}
dataloaders = {x: torch.utils.data.DataLoader(sequence_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6) for x in ['train', 'val']}
dataset_sizes = {x: len(sequence_datasets[x]) for x in ['train', 'val']}
class_names = sequence_datasets['train'].classes

use_gpu = torch.cuda.is_available()



def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    ##------my code------------##
    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []
    val_loss = 100
    ##--------finish--------------##
    for epoch in range(0,num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                
                inputs, labels = data
                
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs, phase)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                #print(loss.data[0]/batch_size, torch.sum(preds == labels.data)/batch_size)

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
            plt.savefig(file_name+"_loss.png")
            plt.cla()
            plt.clf()
            plt.plot(range(len(train_acc)),train_acc,label="Train")
            plt.plot(range(len(test_acc)),test_acc,label="Test")
            plt.legend(bbox_to_anchor=(.90, 1), loc=2, borderaxespad=0.)
            plt.savefig(file_name+"_acc.png")
            plt.cla()
            plt.clf()


            ##--------- my code finish--------------##
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = model.state_dict()
                print ('saving model.....')
                torch.save(model, 'weights_' + file_name + '_epoch_' + str(epoch) + '_lr_' + str(lr) + '.pt')
        print()

    time_elapsed = time.time() - sinceview(1,-1)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    model = torch.load('weights_' + file_name + '_epoch_' + str(epoch) + '_lr_' + str(lr) + '_' + str(batch_size) + '_' + str(sequence_length) + 'cnn_10_class_ctrl_crop.pt')
    return model



if __name__ == '__main__':

    file_name = __file__.split('/')[-1].split('.')[0]
    print (file_name)
    
    
    #model_conv = torchvision.models.resnet50(pretrained=True)
    
    #model_conv = ResNet50Bottom(model_conv)
    model_conv=torch.load('/home/shubham/Egocentric/weights/weights_resnet_50_fine_tune_gtea_10_classes_ctrl_crop_lr_001.pt')
    for param in model_conv.parameters():
        param.requires_grad = False
        
    #model_conv = model_conv.cuda()
    
    #print(model_conv)

    hidden_size = 512
    input_size=2048
    model = Net(model_conv, input_size,hidden_size)
    
    print(model)
    #model = torch.load('weights_lstm_epoch_10_lr_0.001.pt')
    
    #criterion = nn.CrossEntropyLoss
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer = optim.SGD(model.lstmNet.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model = train_model(model, criterion, optimizer,
                             exp_lr_scheduler, num_epochs=num_epochs)

    ######################################################################
    #
    visualize_model(model)
    plt.ioff()
    plt.show()
