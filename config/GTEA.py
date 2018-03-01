from torchvision import  transforms

rgb = {
 'mean': [0.485, 0.456, 0.406],
 'std': [0.229, 0.224, 0.225],
 'lr': 0.001,
 'momentum': 0.9,
 'step_size': 7,
 'gamma': 1,
 'num_epochs': 500,
 'data_dir': '/home/shubham/Egocentric/dataset/GTEA/',
 'png_dir': 'pngs/',
 'label_dir': 'gtea_labels_cleaned/',
 'features_2048_dir' : 'rgb_2048_features/',
 'num_classes': 10,
 'batch_size': 128,
 'train_csv': 'train.csv',
 'test_csv': 'test.csv',
 'weights_dir': 'weights/RGB/',
 'plots_dir': 'plots/RGB/',
 'class_map': {'x':0, 'bg':0, 'fold':1, 'pour':2, 'put':3, 'scoop':4, 'shake':5, 'spread':6, 'stir':7, 'take':8, 'open': 9, 'close':9},
 'data_transforms' : {
  'train': transforms.Compose([
       transforms.CenterCrop([280, 450]),
       transforms.RandomCrop(224),
       transforms.Resize(300),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
       
   ]),
   'test': transforms.Compose([
       transforms.CenterCrop(224),
       transforms.Resize(300),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
   ]),
  }
}

rgb_lstm = {'lr':0.001,
 'momentum':0.9,
 'step_size':20,
 'gamma':0.1,
 'num_epochs':500,
 'data_dir':'/home/shubham/Egocentric/dataset/GTEA/',
 'label_dir': 'gtea_labels_cleaned/',
 'features_2048_dir' : 'rgb_2048_features/',
 'num_classes':11,
 'batch_size':32,
 'train_csv':'train_lstm.csv',
 'test_csv':'test_lstm.csv',
 'weights_dir':'weights/RGB/',
 'plots_dir':'plots/RGB/',
 'sequence_length':11,
 'class_map': {'x':0, 'bg':0, 'fold':1, 'pour':2, 'put':3, 'scoop':4, 'shake':5, 'spread':6, 'stir':7, 'take':8, 'open': 9, 'close':10}
}


rgb_activity = {'lr':0.001,
 'momentum':0.9,
 'step_size':20,
 'gamma':0.1,
 'num_epochs':30,
 'data_dir':'/home/shubham/Egocentric/dataset/GTEA/',
 'features_2048_dir':'rgb_2048_features/',
 'num_classes':7,
 'batch_size':1,
 'train_csv':'train_activity.csv',
 'test_csv':'test_activity.csv',
 'weights_dir':'weights/RGB/',
 'plots_dir':'plots/RGB/',
 'class_map':{'Cheese': 0, 'Coffee': 1, 'CofHoney': 2, 'Hotdog': 3, 'Pealate': 4, 'Peanut': 5, 'Tea': 6}
}

flow = {'mean':[0.5, 0.5, 0.5],
 'std':[1, 1, 1],
 'lr':0.001,
 'momentum':0.9,
 'step_size':200,
 'gamma':1,
 'num_epochs':5000,
 'data_dir':'/home/shubham/Egocentric/dataset/GTea/',
 'num_classes':11,
 'batch_size':128,
 'train_csv':'train_label_Gtea_flow_11_classes.csv',
 'test_csv':'test_label_Gtea_flow_11_classes.csv',
 'weights_dir':'weights/FLOW/',
 'plots_dir':'plots/FLOW/'
}
