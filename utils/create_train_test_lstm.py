from os import listdir
import csv
import sys
sys.path.append('/home/shubham/ego_action_recognition/config/')
import GTEA as DATA
class_map=DATA.rgb['class_map'] 
label_dir=DATA.rgb['data_dir']+DATA.rgb['label_dir']
label_files=listdir(label_dir)
out_dir = DATA.rgb['data_dir']

w_train=csv.writer(open(out_dir+'train_lstm.csv','wb'))
w_test=csv.writer(open(out_dir+'test_lstm.csv','wb'))
sequence_length=11
for label in label_files:
    if 'S4' not in label:
        r = csv.reader(open(label_dir + label, 'r'), delimiter=' ')
        for row in r:
            if len(row) > 1:
                for i in range(int(row[2]), int(row[3])+2-sequence_length):
                    print ([label[:-4]+'/'+str(item).zfill(10)+'.png' for item in range(i,i+sequence_length)]+[class_map[row[0]]])
                    w_train.writerow([label[:-4]+'/'+str(item).zfill(10)+'.png' for item in range(i,i+sequence_length)]+ [class_map[row[0]]])
    
        
for label in label_files:
    if 'S4' in label:
        r = csv.reader(open(label_dir + label, 'r'), delimiter=' ')
        for row in r:
            if len(row)>1:
                for i in range(int(row[2]), int(row[3])+2-sequence_length):
                    #print ([str(item).zfill(10)+'.png' for item in range(i,i+sequence_length)]+[class_map[row[0]]])
                    w_test.writerow([label[:-4]+'/'+str(item).zfill(10)+'.npy' for item in range(i,i+sequence_length)]+ [class_map[row[0]]])                
              

    