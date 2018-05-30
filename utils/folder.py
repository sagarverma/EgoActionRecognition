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
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import random
from os import listdir


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
        images.append((dir + image[0], int(image[1])))

    return images

def make_sequence_dataset(dir, sequences_list):
    sequences = []

    for sequence in sequences_list:
        images = []
        for image in sequence:
            images.append(dir + image)

        sequences.append([images, int(sequences_list[1])])

    return sequences

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

def sequence_loader(paths, mean, std, inp_size, rand_crop_size, resize_size):
    irand = random.randint(0, inp_size[0] - rand_crop_size[0])
    jrand = random.randint(0, inp_size[1] - rand_crop_size[1])
    flip = random.random()
    batch = []
    for path in paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = functional.center_crop(img, (inp_size[0], inp_size[1]))
        img = functional.crop(img, irand, jrand, rand_crop_size[0], rand_crop_size[1])
        img = functional.resize(img, resize_size)
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


        shuffle(images_list)
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

    def __init__(self, root, csv_file, sequence_length, mean, std, inp_size, rand_crop_size, resize_size):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        sequences_list = []

        for row in r:
            sequences_list.append([row[0:-1],row[-1]])

        shuffle(sequences_list)
        sequences = make_sequence_dataset(root, sequences_list)


        self.root = root
        self.sequences = sequences
        self.loader = sequence_loader
        self.mean = mean
        self.std = std
        self.inp_size = inp_size
        self.rand_crop_size = rand_crop_size
        self.resize_size = resize_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        paths, target = self.sequences[index]
        sequence = self.loader(paths, self.mean, self.std, self.inp_size, self.rand_crop_size, self.resize_size)
        return sequence, target

        return img, target

    def __len__(self):
        return len(self.imgs)

class NpySequencePreloader(data.Dataset):

    def __init__(self, root, features_dir, csv_file):

        r = csv.reader(open(root + csv_file, 'r'), delimiter=',')

        sequence_list = []
        for row in r:
            sequence_list.append([row[0:-1], int(row[-1])])

        seqs = make_seq_dataset(root + features_dir, sequence_list)

        self.root = root
        self.sequences = sequences
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
