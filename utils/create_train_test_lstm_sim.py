from os import listdir
import csv
import sys
import itertools
import random
from random import shuffle
sys.path.append('.')
import config.GTEA as DATA
class_map = DATA.rgb['class_map']
label_dir = DATA.rgb['data_dir'] + DATA.rgb['label_dir']
label_files = listdir(label_dir)
out_dir = DATA.rgb['data_dir']
sequence_length = DATA.rgb_lstm['sequence_length']
print ("###################### running binary labels code #################")
window=6
stride=1
print ("############ Window Size = {} ############ Stride = {} ##############".format(window, stride))
w_train = csv.writer(open(out_dir + 'train_sim.csv','wb'))
w_test = csv.writer(open(out_dir + 'test_sim.csv','wb'))
## --------- code to calculate unique obejcts ------------------##
# objects =[]
# for label in label_files:
#     r = csv.reader(open(label_dir + label, 'r'), delimiter = ' ')
#     for row in r:
#         if len(row) > 1:
#             if row[1] not in objects:
#                 #print (row[0], row[1])
#                 objects.append(row[1])
#print (len(objects),objects)
count_zero=0;
count_ones=0;
all_rows=[]
for label in label_files:
    if 'S4' not in label:
        r = csv.reader(open(label_dir + label, 'r'), delimiter = ' ')
        tot_rows=[]
        for row_ in r:
            if len(row_) > 1:
                tot_rows.append(row_)
        r = csv.reader(open(label_dir + label, 'r'), delimiter = ' ')
        for row in r:
            if len(row) > 1:
                if tot_rows[-1]==row:
                    all_rows.append([label[:-4], row, str(1)])
                else:
                    all_rows.append([label[:-4], row, str(0)])
                for i in range(int(row[2]), int(row[3])-window, stride ):
                    #print (i)
                    #print ([label[:-4] + '/'+str(item).zfill(10) + '.png' for item in range(i,i + window)] + [str(0)])
                    w_train.writerow([label[:-4] + '/'+str(item).zfill(10) + '.png' for item in range(i,i + window)] + [str(0)])
                    count_zero=count_zero+1
                if tot_rows[-1] != row:
                    for i in range((int(row[3])-window+1), int(row[3])+1, stride ):
                        #print ([label[:-4] + '/'+str(item).zfill(10) + '.png' for item in range(i,i + window)] + [str(1)])
                        w_train.writerow([label[:-4] + '/'+str(item).zfill(10) + '.png' for item in range(i,i + window)] + [str(1)])
                        count_ones=count_ones+1

print (count_ones, count_zero)
all_comb=[]
for comb in itertools.combinations(range(len(all_rows)), 2):
    all_comb.append(comb)
shuffle(all_comb)

for comb in all_comb:
    if count_ones==count_zero:
        break
    if ((int(all_rows[comb[0]][2]))==0 and (int(all_rows[comb[1]][2]))==0):
        #print ( (int(all_rows[comb[0]][2])) , (int(all_rows[comb[1]][2])))
        count_ones=count_ones+1
        #print (all_rows[comb[0]][0],all_rows[comb[0]][1][2],'to', all_rows[comb[0]][1][3], all_rows[comb[1]][0], all_rows[comb[1]][1][2],'to',all_rows[comb[1]][1][3])
        #print ([all_rows[comb[0]][0] + '/'+str(item+1).zfill(10) + '.npy' for item in range(int(all_rows[comb[0]][1][3])-(window/2),int(all_rows[comb[0]][1][3]))] +  [all_rows[comb[1]][0] + '/'+str(item).zfill(10) + '.npy' for item in range(int(all_rows[comb[1]][1][2]),int(all_rows[comb[1]][1][2])+(window/2))]+ [str(1)]  )
        w_train.writerow([all_rows[comb[0]][0] + '/'+str(item+1).zfill(10) + '.png' for item in range(int(all_rows[comb[0]][1][3])-(window/2),int(all_rows[comb[0]][1][3]))]\
         +  [all_rows[comb[1]][0] + '/'+ str(item).zfill(10) + '.png' for item in range(int(all_rows[comb[1]][1][2]),int(all_rows[comb[1]][1][2])+(window/2))]+ [str(1)])
print (count_ones, count_zero)

for label in label_files:
    if 'S4' in label:
        #print (label)
        r = csv.reader(open(label_dir + label, 'r'), delimiter = ' ')
        tot_rows=[]
        for row_ in r:
            if len(row_) > 1:
                tot_rows.append(row_)
        r = csv.reader(open(label_dir + label, 'r'), delimiter = ' ')
        for row in r:
            if len(row) > 1:
                for i in range(int(row[2]), int(row[3])-window, stride):
                    #print ([label[:-4] + '/'+str(item).zfill(10) + '.npy' for item in range(i,i + window)] + [str(0)])
                    w_test.writerow([label[:-4] + '/'+str(item).zfill(10) + '.npy' for item in range(i,i + window)] + [str(0)])
                #print ("====================================================")
                #print ([label[:-4] + '/'+str(item+1).zfill(10) + '.npy' for item in range(int(row[3])-(window/2) , int(row[3]) + (window/2))]  + [str(1)])
                if tot_rows[-1] != row:
                    for i in range((int(row[3])-window+1), int(row[3])+1, stride ):
                        #print ([label[:-4] + '/'+str(item).zfill(10) + '.png' for item in range(i,i + window)] + [str(1)])
                        w_test.writerow([label[:-4] + '/'+str(item).zfill(10) + '.npy' for item in range(i,i + window)] + [str(1)])
