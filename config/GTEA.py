from torchvision import  transforms

rgb = {
 'mean': [0.485, 0.456, 0.406],
 'std': [0.229, 0.224, 0.225],
 'lr': 0.001,
 'momentum': 0.9,
 'step_size': 20,
 'gamma': 1,
 'num_epochs': 500,
 'data_dir': '../../dataset/',
 'png_dir': 'pngs/',
 'label_dir': 'gtea_labels_cleaned/',
 'num_classes': 10,
 'batch_size': 128,
 'train_csv': 'train_rgb_cnn.csv',
 'test_csv': 'test_rgb_cnn.csv',
 'weights_dir': '../../weights/',
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

rgb_lstm = {
    'lr': 0.001,
    'momentum': 0.9,
    'step_size': 20,
    'gamma': 1,
    'num_epochs': 500,
    'data_dir': '../../dataset/',
    'features_2048_dir': 'rgb_2048_features/',
    'png_dir': 'pngs/',
    'num_classes': 11,
    'batch_size': 128,
    'sequence_length': 11,
    'train_csv': 'train_lstm.csv',
    'test_csv': 'test_lstm.csv',
    'class_map': {'x':0, 'bg':0, 'fold':1, 'pour':2, 'put':3, 'scoop':4, 'shake':5, 'spread':6, 'stir':7, 'take':8, 'open': 9, 'close':10}
}

rgb_sim = {
    'lr': 0.001,
    'momentum': 0.9,
    'step_size': 20,
    'gamma': 1,
    'num_epochs': 500,
    'data_dir': '../../dataset/',
    'features_2048x10x10_dir': 'rgb_2048x10x10_features/',
    'weights_dir': '../../weights/',
    'num_classes': 2,
    'batch_size': 128,
    'train_csv': 'train_sim.csv',
    'test_csv': 'test_sim.csv',
    'class_map': {'sim':0, 'dsim':1}
}

flow = {'mean': [0.5, 0.5, 0.5],
 'std': [1, 1, 1],
 'lr': 0.001,
 'momentum': 0.9,
 'step_size': 3,
 'gamma': 0.1,
 'num_epochs': 300,
 'data_dir': '../../dataset/',
 'png_dir': 'flows/',
 'num_classes': 11,
 'batch_size': 64,
 'train_csv': 'train_flow_cnn.csv',
 'test_csv': 'test_flow_cnn.csv',
 'weights_dir': '../../weights/',
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

flow_lstm = {
    'lr': 0.01,
    'momentum': 0.9,
    'step_size': 20,
    'gamma': 1,
    'num_epochs': 500,
    'data_dir': '../../dataset/',
    'features_2048_dir': 'flow_2048_features/',
    'png_dir': 'flows/',
    'num_classes': 11,
    'batch_size': 128,
    'sequence_length': 11,
    'train_csv': 'train_lstm.csv',
    'test_csv': 'test_lstm.csv',
    'class_map': {'x':0, 'bg':0, 'fold':1, 'pour':2, 'put':3, 'scoop':4, 'shake':5, 'spread':6, 'stir':7, 'take':8, 'open': 9, 'close':10}
}
