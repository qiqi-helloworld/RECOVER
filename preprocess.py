import torch
import torchvision.transforms as transforms
import random
import numpy as np

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

__tiny_imagenet_stats = {'mean': [0.4802, 0.4481, 0.3975],
                   'std': [0.2302, 0.2265, 0.2262]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

__cifar10_stats = {'mean': [0.4914, 0.4822, 0.4465],
                      'std': [0.2023, 0.1994, 0.2010]}

__cifar100_stats = {'mean': [0.5071, 0.4867, 0.4408],
                       'std': [0.2675, 0.2565, 0.2761]}



def get_transform_medium_scale_data(name='cifar10', input_size=None,
                  scale_size=None, normalize=None, isTrain=True):

    if 'cifar10' == name:
        input_size = input_size or 32
        normalize = normalize or __cifar10_stats
        if isTrain:
            scale_size = scale_size or 40 # this variable is already not useful anymore
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize, fill=127)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)
    elif 'cifar100' == name:
        input_size = input_size or 32
        normalize = normalize or __cifar100_stats
        if isTrain:
            scale_size = scale_size or 40
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize, fill=127)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)


    elif name == 'mnist':
        normalize = {'mean': [0.5], 'std': [0.5]}
        input_size = input_size or 28
        if isTrain:
            scale_size = scale_size or 32
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize)
        else:
            scale_size = scale_size or 28
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

    elif name == 'tiny-imagenet-200':

        normalize = normalize or __tiny_imagenet_stats
        scale_size = scale_size or 256
        input_size = input_size or 224
        if isTrain:
            return inception_preproccess(input_size, normalize=normalize)
        else:
            return scale_crop_tiny_imagenet_32X32(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

    elif name == 'stl10':
        return transforms.Compose([
                                 #transforms.Grayscale(),
                                   transforms.ToTensor()
                            #       transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
                                    ])
    elif name == 'svhn':
        if isTrain:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])
        else:
            return transforms.Compose([
                transforms.ToTensor()
                ])
    # elif name == 'chexpert':
    #     return transforms.Compose([
    #         transforms.Resize(scale_size) if scale_size else transforms.Lambda(lambda x: x),
    #         transforms.CenterCrop(320 if not scale_size else scale_size),
    #         lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),  # tensor in [0,1]
    #         transforms.Normalize(mean=[0.5330], std=[0.0349]),  # whiten with dataset mean and std
    #         lambda x: x.expand(3, -1, -1)])


def get_data_transform_ImageNet_iNaturalist18(split, rgb_mean, rbg_std, key = 'ImageNet'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) ,
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]



def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size), #center crop
        transforms.ToTensor(),
        transforms.Normalize(**normalize), # Normalization Data
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    return transforms.Compose(t_list)



def scale_crop_tiny_imagenet_32X32(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    return transforms.Compose(t_list)



def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    return transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats, fill=0):
    #padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        #transforms.Pad(padding, fill=fill), # fill = 127
        transforms.RandomCrop(input_size, padding=4), # 32
        transforms.RandomHorizontalFlip(), # Random Crop
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])



class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.copy()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):


    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
