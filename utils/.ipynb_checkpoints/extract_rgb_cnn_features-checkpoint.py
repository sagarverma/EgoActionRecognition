import sys
sys.path.append('.')

import shutil, time, os, csv

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
features_2048x10x10_dir = DATA.rgb_lstm['features_2048x10x10_dir']
weights_dir = DATA.rgb['weights_dir']
label_dir = DATA.rgb['label_dir']

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
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        x = self.fc(x)
        x = F.softmax(x)
        x = x.view(-1)
        return x

class Net(nn.Module):
    def __init__(self, model_conv):
        super(Net, self).__init__()
        self.resnet50Bottom = ResNet50Bottom(model_conv)

    def forward(self, inp):
        outputs = self.resnet50Bottom(inp)
        return outputs
"""
model_conv = torch.load(weights_dir + 'weights_rgb_cnn_lr_0.001_momentum_0.9_step_size_15_gamma_1_num_classes_10_batch_size_128.pt')


#model = Net(model_conv)
model = model_conv.cuda()

truth = 0
tot = 0

videos = os.listdir(data_dir + png_dir)
for video in videos:print (
    images = os.listdiprint (r(data_dir + png_dir + video)
    images.sort()

    gt = [class_map['x'] for x in range(len(images))]

    r = csv.reader(open(data_dir + label_dir + video + '.txt', 'rb'), delimiter=' ')
    actions = {}
    for row in r:
        if len(row) > 0:
            if row[0] not in actions:
                actions[row[0]] = 1
            for i in range(int(row[2])-1, int(row[3])):
                gt[i] = class_map[row[0]]

    features_out = []
    for image in images:
        img = pil_loader(data_dir + png_dir + video + '/' + image)
        img = functional.center_crop(img, (224, 224))
        img = functional.resize(img, 300)
        img = functional.to_tensor(img)
        img = functional.normalize(img, mean, std)
        img = torch.stack([img]).cuda()
        feature = model(img)
        #feature = feature.view(2048,10,10)
        feature = ((feature).data).cpu().numpy()
        features_out.append(feature)


    for i in range(len(gt)):
        if gt[i] == np.argmax(features_out[i]):
            truth += 1
        tot += 1

    path = '../../dataset/action_set/data/features/' + video + '.npy'
    print (np.asarray(features_out).shape)
    print truth, tot
    np.save(path, np.asarray(features_out))

print truth, tot
"""
"""
fout_g = open('../../dataset/action_set/data/grammar.txt','wb')
videos = os.listdir(data_dir + png_dir)
maxx = 0
for video in videos:
    images = os.listdir(data_dir + png_dir + video)
    gt = ['x' for x in range(len(images))]

    r = csv.reader(open(data_dir + label_dir + video + '.txt', 'rb'), delimiter=' ')
    actions = {}
    print (video)
    
    for row in r:
        if len(row) > 0:
            maxx = max(maxx, (int(row[3])-int(row[2])))
            if row[0] not in actions:
                actions[row[0]] = 1
                    
            for i in range(int(row[2])-1, int(row[3])):
                gt[i] = row[0]    

    fout = open('../../dataset/action_set/data/transcripts/' + video + '.txt', 'w')
    fout.write('x\n')
    for k in actions.keys():
        if k != 'x':
            fout.write(k + '\n')
    fout.write('x\n')
    fout.close()

    fout = open('../../dataset/action_set/data/groundTruth/' + video + '.txt', 'w')
    for g in gt:
        fout.write(g + '\n')
    fout.close()

    old_g = ''
    for g in gt:
        if  g != old_g:
            fout_g.write(g + ' ')
            old_g = g
    fout_g.write('\n')

fout_g.close()
print maxx
"""
## code to take ground truth from LSTM output and generate transcript, grammar and ground truth
dest_path = '../../dataset/action_set/data/'
soc_path = '../../dataset/visualization/npy/seq_11/'
fout_g = open(dest_path + 'grammar.txt','wb')
videos = os.listdir(soc_path)
class_map = {0:'x',1:'fold', 2:'pour', 3:'put', 4:'scoop', 5:'shake', 6:'spread', 7:'stir', 8:'take', 9: 'open', 10:'close'}
for video in videos:
    if 'npy' in video:
        data = np.load(soc_path + video)
        label = data[:,1]
        fout = open(dest_path + 'groundTruth/' + video[:-20] + '.txt', 'w')
        fout_t = open(dest_path + 'transcripts/' + video[:-20] + '.txt', 'w')
        unique_action = set()
        gt = []
        for i in range(label.shape[0]):
            fout.write(class_map[label[i]]+'\n')
            gt.append(class_map[label[i]])
            unique_action.add(class_map[label[i]])
        fout.close()
        old_g = ''
        if 'S1' in video:
            print video
            for g in gt:
                if  g != old_g:
                    fout_g.write(g + ' ')
                    old_g = g
            fout_g.write('\n')
            for item in unique_action:
                fout_t.write(item+'\n')

## code to compare grammar and predicted action sequence 
def levenshtein(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


soc_path = '../../dataset/visualization/npy/seq_11/'
videos = os.listdir(soc_path)
class_map = {0:'x',1:'fold', 2:'pour', 3:'put', 4:'scoop', 5:'shake', 6:'spread', 7:'stir', 8:'take', 9: 'open', 10:'close'}

activities= set()
for video in videos:
    activities.add(video[3:-23])
GT = [] 
fout_g = open(dest_path + 'grammar.txt','r')
gt_class_map = {0:'cofHoney',1:'Hotdog', 2:'peanut', 3:'coffee', 4:'pealate', 5:'tea', 6:'cheese'}
GT.append(fout_g.readlines())
gt_temp = []
for i in range(7):
    titu = (GT[0][i].split(' '))
    titu = titu[:-1]
    gt_temp.append(titu)
GT = gt_temp
action_class_map = {'x':'A', 'bg':'B', 'fold':'C', 'pour':'D', 'put':'E', 'scoop':'F', 'shake':'G', 'spread':'H', 'stir':'I', 'take':'J', 'open': 'K', 'close':'L'}
for video in videos:
    if 'npy' in video and 'S4' not in video:
        print (video[:-20])
        data = np.load(soc_path + video)
        label = data[:,1]
        gt = []
        for i in range(label.shape[0]):
            gt.append(class_map[label[i]])
        predicted = []
        old_g = ''
        for g in gt:
            if  g != old_g:
                predicted.append(g)
                old_g = g
        old_common_ele = 0
        new_common_ele = 0
        index_label = 0
        for i in range(len(GT)):
            list_common = []
            mp = ''.join([action_class_map[p] for p in predicted])
            mgt = ''.join([action_class_map[gt] for gt in GT[i]])
            #print mp
            #print mgt
            dist = levenshtein(mp, mgt)
            old_common_ele = new_common_ele
            new_common_ele = dist
            if new_common_ele < old_common_ele:
                index_label = i
        print (predicted)
        print (GT[index_label])
        print (gt_class_map[index_label])
        #break    
