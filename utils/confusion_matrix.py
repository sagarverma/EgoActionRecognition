#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:48:14 2018

@author: shubham
"""
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import csv
predicted=[]
gt=[]
#np.set_printoptions(suppress=True)

with open('/home/shubham/Egocentric/lisa-caffe-public/python/results.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         values= row.pop().split( ',')
         predicted.append(values[0])
         gt.append(values[1])
         
class_names = []
for x in gt:
    if x not in class_names:
        class_names.append(x)
print class_names
class_names.sort()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    cnf_matrix_huji = np.asarray([[91,0,4,0,0,0,0,0,0,0,0,0,0],
                    [0,93,0,1,0,1,3,0,0,0,0,0,0],
                    [5,0,87,0,0,0,5,0,0,0,0,0,0],
                    [0,0,0,97,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,98,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,92,7,0,0,0,0,0,0],
                    [0,1,2,0,2,1,91,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,99,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,93,0,0,0,2],
                    [0,0,0,0,0,0,0,0,0,96,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,97,0,0],
                    [0,0,0,1,0,0,0,0,1,0,0,91,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0,95]], dtype='float')
    
    print(cnf_matrix_huji,np.asarray(cm,dtype='float'))

    cnf_matrix = np.zeros((24,24))
    cnf_matrix[0:13,0:13] = cnf_matrix_huji
    cnf_matrix[13:24,13:24] = cm * 100.0
    
    cm = cnf_matrix

    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    xclasses = []
    for i in range(24):
        if i%2 ==0 :
            xclasses.append(classes[i])
        else:
            xclasses.append('')
            
    yclasses = []
    for i in range(24):
        if i%2 == 1 :
            yclasses.append(classes[i])
        else:
            yclasses.append('')
            
    plt.xticks(tick_marks, xclasses, rotation=45, size=18)
    plt.yticks(tick_marks, yclasses, size=18)

    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), size=15,
                 horizontalalignment="center", verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix_gtea = confusion_matrix(gt, predicted)
np.set_printoptions(precision=2)


print(cnf_matrix_gtea)
class_names = ['Walking', 'Driving', 'Standing', 'Biking', 'Static', 'Riding Bus', 'Sitting', 'Running', 'Stair Climbing', 'Skiing', 'Horseback', 'Sailing','Boxing'] + class_names

print(len(class_names))

# Plot normalized confusion matrix
plt.figure(figsize=[10,10])
plot_confusion_matrix(cnf_matrix_gtea, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


plt.savefig('cm.pdf',format="pdf",transparent=True, bbox_inches='tight', \
                        pad_inches=0)

