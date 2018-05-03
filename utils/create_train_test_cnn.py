import sys
sys.path.append('../config')

from os import listdir
import csv
import GTEA as DATA
#class_map = DATA.rgb['class_map']
class_map = DATA.flow['class_map']
label_dir = DATA.rgb['data_dir'] + DATA.rgb['label_dir']
out_dir = DATA.rgb['data_dir']

label_files = listdir(label_dir)

#w_train = csv.writer(open(out_dir + 'train.csv', 'w'))
#w_test = csv.writer(open(out_dir + 'test.csv', 'w'))
w_train = csv.writer(open(out_dir + 'train_flow.csv', 'w'))
w_test = csv.writer(open(out_dir + 'test_flow.csv', 'w'))


for label_file in label_files:
    if 'S4' not in label_file:
        print(label_file)
        r = csv.reader(open(label_dir + label_file, 'r'), delimiter=' ')
        
        for row in r:
            
            if len(row) > 1:
                for i in range(int(row[2]), int(row[3]) + 1):
                    w_train.writerow([label_file[:-4] + '/' + str(i).zfill(10) + '.png', class_map[row[0]]])
            

for label_file in label_files:
    if 'S4' in label_file:
        print(label_file)
        r = csv.reader(open(label_dir + label_file, 'r'), delimiter=' ')
        
        for row in r:
            
            if len(row) > 1:
                for i in range(int(row[2]), int(row[3]) + 1):
                    w_test.writerow([label_file[:-4] + '/' + str(i).zfill(10) + '.png', class_map[row[0]]])