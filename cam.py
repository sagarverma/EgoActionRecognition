# the test script
# load the trained model then forward pass on a given image


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
import csv

def load_model(modelID, categories):
    if modelID == 1:
        weight_file = '/home/shubham/Egocentric/dataset/GTea/weights/RGB/weights_resnet_50_lr_0.001_momentum_0.9_step_size_7_gamma_1_num_classes_10_batch_size_128.pt'
        #weight_file = '../Egocentric/weights/weights_resnet_50_fine_tune_gtea_10_classes_lr_001.pt'
    
        
        model = torchvision.models.resnet50()
    
        model = ResNet50Bottom(model)
        #print(model._modules['features'])
        
        model = torch.load(weight_file)
        
        print(model)
        #model._modules['layer4'].register_forward_hook(hook_feature)
        #model._modules.get('avgpool').register_forward_hook(hook_feature)
        model._modules['features'][-1].register_forward_hook(hook_feature)
        model._modules.get('avg_pool').register_forward_hook(hook_feature)
    
    return model

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (300, 300)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf_old = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tf_proposed =  trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return tf_proposed

class ResNet50Bottom_second(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom_second, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

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
        x = self.fc(x)
        #x=self.dropout(x)
        return x
        
dataset = 'GTEA'
modelID = 1

class_map = {5:'take', 1: 'pour', 8: 'put', 0: 'close', 7: 'shake', 2: 'open', 3: 'spread', 4: 'scoop', 9: 'stir', 10: 'bg', 6: 'fold'}

categories = ['take', 'close', 'put', 'pour', 'bg', 'scoop', 'stir', 'fold', 'spread', 'shake']

# load the labels
model = load_model(modelID, categories)

# load the model
features_blobs = []

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.cpu().numpy()
weight_softmax[weight_softmax<0] = 0

r = csv.reader(open('../Egocentric/dataset/GTea/test_flow.csv'), delimiter=' ')

for row in r:
    action_name = class_map[int(row[1])]
    
    image_path = '../Egocentric/dataset/GTea/pngs/' + row[0]
    
    img = cv2.imread(image_path)
    cimg = img[91:91+224, 248:248+224]
    #rimg = cv2.resize(img, (224,224))
    rimg = cv2.resize(cimg, (300,300))
    input_img = V(tf(rimg).unsqueeze(0), volatile=True)
    
    # forward pass
    logit = model.forward(input_img.cuda())
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    
    print('RESULT ON ' + image_path, action_name)
    
    
    # output the prediction of action category
    print('--Top Actions:')
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
    

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    #print(CAMs[0].shape)
    # render the CAM and output
    
    height, width, _ = rimg.shape
    print(height, width, CAMs[0].shape)
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + rimg * 0.5
    cv2.imwrite('cam_results/' + action_name +'_' + row[0].replace('/','_'), result)