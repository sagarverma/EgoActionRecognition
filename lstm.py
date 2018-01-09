
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
from folder import SequencePreloader

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

class SimpleRNN(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.LSTM(input_size, hidden_size, 1)
        self.dropout= nn.Dropout(p=0.2)
        self.out = nn.Linear(hidden_size,11 )
        

    def step(self, input, hidden=None):
        #input = self.inp(input.view(1, -1)).unsqueeze(1)
        print (input.size())
        output, hidden = self.rnn(input.view(1,-1).unsqueeze(1), hidden)
        print (output.size())
        output = self.out(output.squeeze(11))
        
        output=F.softmax(11)
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden


sequence_datasets = {'train': SequencePreloader(data_dir, train_csv), 'val': SequencePreloader(data_dir, val_csv)}
dataloaders = {x: torch.utils.data.DataLoader(sequence_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
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
    for epoch in range(num_epochs):
        
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
                #print (inputs.size())
                inputs=inputs.permute(1, 0, 2)
                #print (inputs.size())
                #fg
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

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
                torch.save(model, 'weights_'+file_name+'_lr_001.pt')
        print()

    time_elapsed = time.time() - sinceview(1,-1)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    model = torch.load('weights_'+file_name+'_lr_001.pt')
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few sequences
#

def visualize_model(model, num_sequences=6):
    sequences_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            sequences_so_far += 1
            ax = plt.subplot(num_sequences//2, 2, sequences_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if sequences_so_far == num_sequences:
                return


# Define the model
    
class SimpleRNN(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.LSTMCell(input_size, hidden_size, 1)
        self.out = nn.Linear(hidden_size,11 )
        self.dropout= nn.Dropout(p=0.4)
        

    def step(self, input, h, c):
        #input = self.inp(input.view(1, -1)).unsqueeze(1)
        h, c = self.rnn(input, (h, c))
        h=self.dropout(h)
        c=self.dropout(c)
        output = self.out(h)
        return output, h, c 

    def forward(self, inputs):
        steps = len(inputs)
        outputs = []
        h = Variable(torch.randn(inputs.size()[1], self.hidden_size)).cuda()
        c = Variable(torch.randn(inputs.size()[1], self.hidden_size)).cuda()
        for i in range(steps):
            input = inputs[i]
            output, h, c = self.step(input, h, c)
            outputs.append(output)
            
        outputs = torch.stack(outputs)
        outputs = torch.mean(outputs, dim=0)
        #outpus = F.softmax(outputs)
        #print(outputs.size())
        return outputs
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp   
if __name__ == '__main__':

    file_name = __file__.split('/')[-1].split('.')[0]
    print (file_name)
#    model_conv = torchvision.models.resnet50(pretrained=True)
#    for param in model_conv.parameters():
#        param.requires_grad = False
#
#    # Parameters of newly constructed modules have requires_grad=True by default
#    num_ftrs = model_conv.fc.in_features
#    model_conv.fc = nn.Linear(num_ftrs, 10)
    hidden_size = 512
    input_size=2048
    model = SimpleRNN(input_size,hidden_size)
    params=get_n_params(model)
    print ('Number of Parameters: ', params)
    
    #criterion = nn.CrossEntropyLoss
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=.1)


    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # On CPU this will take about half the time compared to previous scenario.
    # This is expected as gradients don't need to be computed for most of the
    # network. However, forward does need to be computed.
    #

    model = train_model(model, criterion, optimizer,
                             exp_lr_scheduler, num_epochs=200)

    ######################################################################
    #
    visualize_model(model)
    plt.ioff()
    plt.show()
