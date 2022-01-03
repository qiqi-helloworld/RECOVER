import os
import torchvision.datasets as datasets

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pickle
from preprocess import get_transform
from utils import *
import math
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import os

__DATASETS_DEFAULT_PATH = './data/'


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=False, datasets_path=__DATASETS_DEFAULT_PATH):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)

    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
    elif name == 'svhn':
        return datasets.SVHN(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name =='chexpert':
        return ChexpertSmall(root=root, mode=split, transform=transform)


def get_balanced_dataset(name, transform):
    train_data = get_dataset(name, 'train', transform['train'])
    if name == 'svhn':
        val_data = get_dataset(name, 'test', transform['eval'])
    elif name == 'chexpert':
        val_data = get_dataset(name, 'valid', transform['eval'])
    else:
        val_data = get_dataset(name, 'val', transform['eval'])
    return train_data, val_data

def get_imbalanced_dataset(name, im_ratio, transform):

    im_train_data_file = './data/' + name + '/im_train_data' + "_" + str(im_ratio)
    if not os.path.isfile(im_train_data_file):
        train_data= get_dataset(name, 'train', transform['train'])
        label_list = []
        for (input, label) in train_data:
            label_list.append(label)
            # print(label_list.index(0))
        np_label = np.array(label_list)

        label_stats = Counter(np_label)
        #saved_indexes_start = math.floor((len(np_label) * (1 - im_ratio)) // len(np.unique(np_label)))
        #print(saved_indexes_start, len(np_label) // len(np.unique(np_label)))
        saved_indexes = []
        for i in range(len(np.unique(np_label))):
            if i < len(np.unique(np_label)) // 2:
                saved_indexes_start = math.floor(label_stats[i]* (1 - im_ratio))
                saved_indexes = saved_indexes + list(np.where(np_label == i)[0][saved_indexes_start:])
            else:
                saved_indexes = saved_indexes + list(np.where(np_label == i)[0])

        imbalanced_train_data = torch.utils.data.Subset(train_data, saved_indexes)
        print(len(imbalanced_train_data))

        f = open(im_train_data_file, 'wb')
        pickle.dump(imbalanced_train_data, f)
        f.close()

    val_data_file = './data/' + name + '/val_data'
    if not os.path.isfile(val_data_file):
        if name == 'svhn' or name == 'stl10':
            val_data = get_dataset(name, 'test', transform['eval'])
        else:
            val_data = get_dataset(name, 'val', transform['eval'])
        f = open('./data/'+ name + '/val_data', 'wb')
        pickle.dump(val_data, f)
        f.close()

    f = open('./data/' + name + '/im_train_data'+ "_" + str(im_ratio), 'rb')
    train_data = pickle.load(f)
    f.close()
    f = open('./data/' + name + '/val_data', 'rb')
    val_data = pickle.load(f)
    f.close()

    return  train_data, val_data


class LT(Dataset):
    '''
    ImageNet, ImageNet-LT, iNaturalist2018, iNaturalist2019 Imbalanced Dataset Contruction
    '''
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        print("LT:", txt)
        if 'amax' in os.uname()[1]:
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))
        else:
            if 'test' in txt and 'ImageNet' in txt:
                with open(txt) as f:
                    for line in f:
                        img_name = '/'.join([line.split()[0].split('/')[0], line.split()[0].split('/')[2]])
                        self.img_path.append(os.path.join(root, img_name))
                        self.labels.append(int(line.split()[1]))
            else:
                with open(txt) as f:
                    for line in f:
                        self.img_path.append(os.path.join(root, line.split()[0]))
                        self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label  # , index


if __name__ == '__main__':
    transform = {
          'train': get_transform('mnist',
                                 input_size=28, augment=True),
         'eval': get_transform('mnist',
                               input_size=28, augment=False)
     }


    print("IMRATIO 0.02")
    train_data, val_data = get_imbalance_dataset('mnist', 0.02, transform)
    label_list = []
    for (input, label) in train_data:
        label_list.append(label)
    print(Counter(np.array(label_list)))
    val_label_list = []
    for (input, label) in val_data:
        val_label_list.append(label)
    print(Counter(np.array(val_label_list)))


    print("IMRATIO 0.05")
    train_data, val_data = get_imbalance_dataset('mnist', 0.05, transform)
    label_list = []
    for (input, label) in train_data:
        label_list.append(label)
    print(Counter(np.array(label_list)))
    val_label_list = []
    for (input, label) in val_data:
        val_label_list.append(label)
    print(Counter(np.array(val_label_list)))

    print("IMRATIO 0.2")
    train_data, val_data = get_imbalance_dataset('mnist', 0.2, transform)
    label_list = []
    for (input, label) in train_data:
        label_list.append(label)
    print(Counter(np.array(label_list)))
    val_label_list = []
    for (input, label) in val_data:
        val_label_list.append(label)
    print(Counter(np.array(val_label_list)))

    print("IMRATIO 0.5")
    train_data, val_data = get_imbalance_dataset('mnist', 0.5, transform)
    label_list = []
    for (input, label) in train_data:
        label_list.append(label)
    print(Counter(np.array(label_list)))
    val_label_list = []
    for (input, label) in val_data:
        val_label_list.append(label)
    print(Counter(np.array(val_label_list)))

    print("IMRATIO 1")
    train_data, val_data = get_imbalance_dataset('mnist', 1, transform)
    label_list = []
    for (input, label) in train_data:
        label_list.append(label)
    print(Counter(np.array(label_list)))
    val_label_list = []
    for (input, label) in val_data:
        val_label_list.append(label)
    print(Counter(np.array(val_label_list)))
    # train_data, val_data = get_imbalance_dataset("cifar10", 0.02, transform)
    # train_data, val_data = get_imbalance_dataset("cifar10", 0.05, transform)
    # train_data, val_data = get_imbalance_dataset("cifar10", 0.1, transform)
    # train_data, val_data = get_imbalance_dataset("cifar10", 0.2, transform)
    # print("Length of Train Data: ", len(train_data))
    # print("Length of Validation Data: ", len(val_data))
    # for i, (inputs, target) in enumerate(train_loader):
    #     if i == 0:
    #         print("targets:", target>100)
    #         print(inputs)
    #         print(inputs[0].size())