__author__ = 'Qi'
# Created by on 1/10/21.
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter
from dataset import *
from preprocess import get_transform_medium_scale_data, get_data_transform_ImageNet_iNaturalist18

RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'ImageNet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'ImageNet_LT': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}



def myDataLoader_imagenet(args, data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True):
    assert phase in {'train', 'val', 'test'}
    if 'LT' in args.dataset:
        key = 'ImageNet_LT'
        txt = f'./data/ImageNet_LT/ImageNet_LT_{phase}.txt'
    else:
        key = 'ImageNet'
        txt = f'./data/ImageNet/ImageNet_{phase}.txt'

    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']


    if phase == 'val' and args.stages == 2:
        transform = get_data_transform_ImageNet_iNaturalist18('train', rgb_mean, rgb_std)
    else:
        transform = get_data_transform_ImageNet_iNaturalist18(phase, rgb_mean, rgb_std)

    set_imagenet = LT(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_imagenet)}')
    # if phase == 'test' and test_open:
    #     open_txt = './data/%s/%s_open.txt' % (dataset, dataset)
    #     print('Testing with open sets from %s' % open_txt)
    #     open_set_ = INaturalist('./data/%s/%s_open' % (dataset, dataset), open_txt, transform)
    #     set_ = ConcatDataset([set_, open_set_])
    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          sampler=sampler_dic['sampler'](set_imagenet, **sampler_dic['params']))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# /dual_data/not_backed_up/imagenet-2012/ilsvrc
def myDataLoader_iNaturalist18(args, data_root, batch_size, phase, sampler_dic=None, num_workers=4, shuffle=True, imb_factor = 0.01):

    assert  phase in {'train', 'val'} , "There is no test phase for iNaturalist18"
    key = 'iNaturalist18'
    txt = f'./data/iNaturalist18/iNaturalist18_{phase}.txt'

    print(f'===> Loading iNaturalist10 data from {txt}')
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']


    if phase == 'val' and args.stages == 2:
        transform = get_data_transform_ImageNet_iNaturalist18('train', rgb_mean, rgb_std, key = key)
    else:
        transform = get_data_transform_ImageNet_iNaturalist18(phase, rgb_mean, rgb_std, key = key)


    set_imagenet = LT(data_root, txt, transform)
    print(f'===> {phase} data length {len(set_imagenet)}')

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          sampler=sampler_dic['sampler'](set_imagenet, **sampler_dic['params']))
    else:
        print('No sampler.')
        print('Shuffle is %s.' % shuffle)
        return DataLoader(dataset=set_imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_train_val_test_loader(args, train_sampler = None):

    sampler_dic = None
    test_loader = None

    if args.dataset == 'imagenet-LT':
        if 'argon' in os.uname()[1]:
            args.data_root ="/nfsscratch/qqi7/imagenet/"
        elif 'amax' in os.uname()[1]:
            args.data_root = "/data/imagenet/imagenet/"
        else:
            args.data_root = '/dual_data/not_backed_up/imagenet-2012/ilsvrc'

        train_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train', sampler_dic = sampler_dic, num_workers = args.works, shuffle = True)
        val_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size if args.batch_size != 1 else 64, 'val', sampler_dic = sampler_dic, num_workers = args.works, shuffle = False)
        test_loader =  myDataLoader_imagenet(args, args.data_root, args.batch_size if args.batch_size != 1 else 64, 'test', sampler_dic = sampler_dic, num_workers = args.works, shuffle = False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'imagenet':
        args.data_root = '/dual_data/not_backed_up/imagenet-2012/ilsvrc'
        train_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size, 'train', sampler_dic=sampler_dic,
                                          num_workers=args.works, shuffle=True)
        val_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size // 4, 'val', sampler_dic=sampler_dic,
                                        num_workers=args.works, shuffle=False)
        test_loader = myDataLoader_imagenet(args, args.data_root, args.batch_size // 4, 'test', sampler_dic=sampler_dic,
                                         num_workers=args.works, shuffle=False)
        args.cls_num_list = get_cls_num_list(args)

    elif args.dataset == 'iNaturalist18':
        if 'argon' in os.uname()[1]:
            args.data_root = "/nfsscratch/qqi7/iNaturalist2018/"
        else:
            args.data_root = "/dual_data/not_backed_up/iNaturalist2018/"

        train_loader = myDataLoader_iNaturalist18(args, args.data_root, args.batch_size, 'train', sampler_dic=None,
                                           num_workers=args.works, shuffle=True)
        val_loader = myDataLoader_iNaturalist18(args, args.data_root, 256, 'val', sampler_dic=None,
                                         num_workers=args.works, shuffle=False)

        args.cls_num_list = get_cls_num_list(args)

    else:

        default_transform = {
            'train': get_transform_medium_scale_data(args.dataset,
                                                     input_size=args.input_size, isTrain=True),
            'eval': get_transform_medium_scale_data(args.dataset,
                                                    input_size=args.input_size, isTrain=False)
        }


        if args.dataset == 'C2':
            traindir = './data/C2/im_train_' + str(args.im_ratio) + '/'
            valdir = './data/C2/val/'
            torch.manual_seed(777)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(traindir,
                                     transforms.Compose([
                                         # transforms.Lambda(shear),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir,
                                     transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])),
                batch_size=args.batch_size // 2,
                shuffle=False,
                num_workers=8,
                pin_memory=True)

        elif args.dataset == 'melanoma':
            traindir = './data/C2/im_train_' + str(args.im_ratio) + '/'
            valdir = './data/C2/val/'
            torch.manual_seed(777)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(traindir,
                                     transforms.Compose([
                                         # transforms.Lambda(shear),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir,
                                     transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])),
                batch_size=args.batch_size // 2,
                shuffle=False,
                num_workers=8,
                pin_memory=True)
        else:



            transform = default_transform
            train_data, val_data = get_imbalanced_dataset(args.dataset, args.im_ratio, transform)
            torch.manual_seed(777)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.works, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False,
                                                     num_workers=args.works, pin_memory=True)

    return train_loader, val_loader, test_loader




def network_frozen(args, model):
    for param_name, param in model.named_parameters():
        if 'fc' not in param_name:
            param.requires_grad = False

    if args.frozen_aside_fc and args.frozen_aside_last_block:
        last_block_number = 0
        if args.model == "resnet152":
            last_block_number = 2
        elif args.model == 'resnet50':
            last_block_number = 3
        elif args.model == 'resnet10':
            last_block_number = 0

        last_block_pattern = 'layer4.' + str(last_block_number)

        # last_block_pattern = 'layer4.'
        if args.model == 'resnet32':
            last_block_pattern = 'layer3.4'

        for param_name, param in model.named_parameters():  # (self.networks[key]):  # frozen the first 3 block
            # import pdb; pdb.set_trace()
            # Freeze all parameters except self attention parameters
            # block components:
            #    -- layer1
            #    -- layer2
            #    -- layer3
            #    -- layer4
            #    -- fc
            #    -- fc
            if 'fc' not in param_name and last_block_pattern not in param_name:
                    param.requires_grad = False


def number_of_bp_layers(args, model):
    bp_layers = 0
    all_layers = 0
    for param_name, param in model.named_parameters():
        all_layers += 1
        if param.requires_grad:
            bp_layers += 1
    return bp_layers, all_layers


def get_num_classes(args):
    num_classes = 0
    if args.dataset == 'ina':
        num_classes = 1010
    elif args.dataset == 'imagenet-LT':
        num_classes = 1000
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    elif args.dataset == 'places-LT':
        num_classes = 365
    elif args.dataset == 'covid-LT':
        num_classes = 4
    elif args.dataset == 'iNaturalist18':
        num_classes = 8142
    return num_classes


