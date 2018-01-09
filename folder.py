import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import csv
from random import shuffle 
import numpy as np

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


def make_dataset(dir, images_list, class_to_idx):
    images = []
    
    for image in images_list:
        images.append((dir + image[0], class_to_idx[image[1]]))

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def sequence_loader(path):
    seq = np.load(path)
    if seq.shape[0] < 11:
        curr_shape = seq.shape[0]
        for i in range(0,11-curr_shape):
            seq = np.append(seq, [seq[-1]], axis=0)
    
    seq = torch.from_numpy(seq)  
    return seq
    
    
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImagePreloader(data.Dataset):

    def __init__(self, root, csv_file, transform=None, target_transform=None,
                 loader=default_loader):
                     
        r = csv.reader(open(csv_file, 'r'), delimiter=',')
    
        images_list = []
        
        for row in r:
            images_list.append([row[0],row[1]])


        #shuffle(images_list)
            
        classes, class_to_idx = find_classes(images_list)
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

    def __init__(self, root, csv_file, transform=None, target_transform=None,
                 loader=sequence_loader):
                     
        r = csv.reader(open(csv_file, 'r'), delimiter=',')
    
        sequence_list = []
        
        for row in r:
            sequence_list.append([row[0],row[1]])


        #shuffle(images_list)
            
        classes, class_to_idx = find_classes(sequence_list)
        seqs = make_dataset(root, sequence_list, class_to_idx)
        if len(seqs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.seqs = seqs
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
        path, target = self.seqs[index]
        seq = self.loader(path)
        if self.transform is not None:
            seq = self.transform(seq)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return seq, target

    def __len__(self):
        return len(self.seqs)