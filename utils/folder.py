import torch.utils.data as data
import torch
import sys
from PIL import Image
import os
import os.path
import csv
from random import shuffle 
import numpy as np
import random
from torchvision.transforms import functional
from torch.autograd import Variable

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(images_list):
    
    classes = {}
    class_id = 0
    for image in images_list:
        if image[1] not in classes:
            classes[image[1]] = class_id
            class_id += 1
            
    return classes.keys(), classes

def make_seq_dataset(data_png_dir, sequence_list):
    sequences = []
    
    for seq in sequence_list:
        sequence = []
        for image in seq[0]:
            sequence.append(data_png_dir + image)
        
        sequences.append((sequence, seq[1]))
        
    return sequences
    
def make_dataset(dir, images_list, class_to_idx):
    images = []
    
    for image in images_list:
        images.append((dir + image[0], int(image[1])))

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
        
        
def npy_seq_loader(seq):
    out = []
    
    for s in seq:
        out.append(np.load(s))
        
    out = np.asarray(out)
        
    return out

def sequence_loader(seq, mean, std):
    irand = random.randint(0, 280 - 224)
    jrand = random.randint(0, 450 - 224)
    flip = random.random()
    batch = []
    for path in seq:
        img = Image.open(path)
        img = img.convert('RGB')
        img = functional.center_crop(img, (280, 450))
        img = functional.crop(img, irand, jrand, 224, 224)
        img = functional.resize(img, 300)
        if flip < 0.5:
            img = functional.hflip(img)
        tensor = functional.to_tensor(img)
        tensor = functional.normalize(tensor, mean, std)
        batch.append(tensor)

    batch = torch.stack(batch)
    
    return batch
        
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def load_image_for_flow(imgfile):
    image = io.load_image(imgfile)
    #samples = caffe.io.oversample([image,[227,227])
    
    return image

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImagePreloader(data.Dataset):

    def __init__(self, root, csv_file, class_map, transform=None, target_transform=None,
                 loader=default_loader):
                     
        r = csv.reader(open(csv_file, 'r'), delimiter=',')
    
        images_list = []
        
        for row in r:
            images_list.append([row[0],row[1]])


        #shuffle(images_list)
            
        classes, class_to_idx = class_map.keys(), class_map
        imgs = make_dataset(root, images_list, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
        
class SequencePreloader(data.Dataset):

    def __init__(self, mean, std, data_dir, png_dir, csv_file, 
                    class_map, transform=None, target_transform=None, loader=sequence_loader):
                        
        r = csv.reader(open(data_dir + csv_file, 'r'), delimiter=',')
        sequence_list = []
        for row in r:
            sequence_list.append([row[0:-1], int(row[-1])])
            
        classes, class_to_idx = class_map.keys(), class_map
        seqs = make_seq_dataset(data_dir + png_dir, sequence_list)
        if len(seqs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.data_dir = data_dir
        self.seqs = seqs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mean = mean
        self.std = std
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.seqs[index]
        seq = self.loader(path, self.mean, self.std)
        if self.transform is not None:
            seq = self.transform(seq)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return seq, target

    def __len__(self):
        return len(self.seqs)
        
        
class NpySequencePreloader(data.Dataset):

    def __init__(self, data_dir, features_2048_dir, csv_file, class_map):
                     
        r = csv.reader(open(data_dir+csv_file, 'r'), delimiter=',')
    
        sequence_list = []
        
        for row in r:
            sequence_list.append([row[0:-1], int(row[-1])])
            
        classes, class_to_idx = find_classes(sequence_list)
        seqs = make_seq_dataset(data_dir + features_2048_dir, sequence_list)
        if len(seqs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.data_dir = data_dir
        self.seqs = seqs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = npy_seq_loader
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        paths, target = self.seqs[index]
        seq = self.loader(paths)
        return seq, target

    def __len__(self):
        return len(self.seqs)
        