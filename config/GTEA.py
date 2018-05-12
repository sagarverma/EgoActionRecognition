from torchvision import  transforms

rgb = {
 'mean': [0.485, 0.456, 0.406],
 'std': [0.229, 0.224, 0.225],
 'lr': 0.001,
 'momentum': 0.9,
 'step_size': 7,
 'gamma': 1,
 'num_epochs': 500,
 'data_dir': '../../GTEA/',
 'png_dir': 'dataset/pngs/',
 'label_dir': 'dataset/gtea_labels_cleaned/',
 'features_2048_dir' : 'dataset/rgb_2048_features/',
 'feature_2048_dir_conv_lstm' : 'rgb_10x10x2048_features/', 
 'num_classes': 10,
 'batch_size': 128,
 'train_csv': 'dataset/train_rgb_cnn.csv',
 'test_csv': 'dataset/test_rgb_cnn.csv',
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
 'data_dir':'/home/shubham/GTEA/',
 'label_dir': 'dataset/gtea_labels_cleaned/',
 'features_2048_dir' : 'cnn_features/',
 'num_classes':11,
 'batch_size':8,
 'train_csv':'dataset/train_lstm.csv',
 'test_csv':'dataset/test_lstm.csv',
 'weights_dir':'weights/RGB/',
 'plots_dir':'plots/RGB/',
 'sequence_length':100,
 'class_map': {'x':0, 'bg':0, 'fold':1, 'pour':2, 'put':3, 'scoop':4, 'shake':5, 'spread':6, 'stir':7, 'take':8, 'open': 9, 'close':10}
}


flow = {'mean':[0.5, 0.5, 0.5],
 'std':[1, 1, 1],
 'lr':0.001,
 'momentum':0.9,
 'step_size':200,
 'gamma':0.1,
 'num_epochs':300,
 'data_dir':'/home/shubham/GTEA/',
 'num_classes':11,
 'batch_size':64,
 'train_csv':'dataset/train_flow_cnn.csv',
 'test_csv':'dataset/test_flow_cnn.csv',
 'png_dir':'dataset/flows/',
 'weights_dir':'weights/FLOW/',
 'features_2048_dir' : 'dataset/flow_2048_features/',
 'plots_dir':'plots/FLOW/',
 'class_map': {'x':0, 'bg':0, 'fold':1, 'pour':2, 'put':3, 'scoop':4, 'shake':5, 'spread':6, 'stir':7, 'take':8, 'open': 9, 'close':10},
 'data_transforms' : {
  'train': transforms.Compose([
       transforms.Resize([300,300]),
       transforms.RandomCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.5, 0.5, 0.5],[1,1,1])
       
   ]),
   'test': transforms.Compose([
       transforms.Resize([224,224]),
       transforms.ToTensor(),
       transforms.Normalize([0.5, 0.5, 0.5],[1,1,1])
   ]),
  }
}
