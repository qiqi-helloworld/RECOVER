from models import *
import argparse
import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
from data import get_imbalance_dataset
import models
from torch.autograd import Variable
from preprocess import get_transform
from utils import *
import copy
import re
from ast import literal_eval
from torch.nn.utils import clip_grad_norm
from math import ceil
import shutil
import numpy as np

#model_names = sorted(name for name in models.__dict__
#                     if name.islower() and not name.startswith("__")
#                     and callable(models.__dict__[name]))
#print(model_names)


# models_names = sorted(name for name in models.__dict__ if name.islower() and not name.startwith("__"))
# print (model_names)
# model_names = "a"
# print(models.__dict__)
parser = argparse.ArgumentParser(description="Pytorch PLCOVER Training")
parser.add_argument('--results_dir', metavar="RESULTS_DIR", default='./TrainingResults', help = 'results dir')

parser.add_argument('--save', metavar = 'SAVE',  default='',help='save folder')
parser.add_argument('--res_name', default='', type = str, help = 'results file name')
parser.add_argument('--dataset',  metavar='DATASET', default='cifar10',
                    help = 'dataset name or folder')

parser.add_argument('--model', metavar = 'MODEL', default='resnet', help ='model architecture')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help = 'types of tensor - e.g torch.cuda.FloatTensor')
parser.add_argument('--gpus',  default='0', help = 'gpus used for training - e.g 0,1,2,3')
parser.add_argument('--workers', default='8', type = int, metavar='N',
                    help='number of data loading workers (default:256)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help = 'mini-batch size (default:256)')
parser.add_argument('--optimizer', default='SGD',type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--momentum', default=0.9, type = float, metavar="M",
                    help = "momentum parameter of SHB or SNAG")
parser.add_argument('--scale_size', default=32, type=int, help = 'image scale size for data preprocessing')
parser.add_argument('--input_size', default=32, type=int, help = 'the size of image. e.g. 32 for cifar10, 224 for imagenet')
parser.add_argument('--works', default=8, type=int, help = 'number of threads used for loading data')

parser.add_argument('--weight_decay', default=0, type=float, help ='weight decay parameters')
parser.add_argument('--print_freq', '-p', default=50, type = int,
                    help = 'print frequency (default:50)')
# number of restart batches: restart_init_loop * batchsize
parser.add_argument('--restart_init_loop', default=5, type = int,
                    help = 'restart minibatch size = restart_init_loop * batchsize')
parser.add_argument('--start_training_time', type = float, help = 'Overall training start time')
parser.add_argument('--lamda', default=5, type = float, help = 'parameters of regularization')
parser.add_argument('--num_classes', default=10, type = float, help = 'different number of classes for different datasets')

#parser.add_argument('--boolean_flag',
#                   help='This is a boolean flag.',
#                    type=eval,
#                    choices=[True, False],
#                    default='True')
# boolean variable
parser.add_argument('--nesterov', default=False, type=eval, choices=[True, False],
                    help = 'This is used to determine whether we use SNAG')
parser.add_argument('--resume', default=False, type=eval, choices=[True, False],
                    help = 'Training from scratch (False) or from the saved check point')

###Tuning Parameters
parser.add_argument('--a_t', default=0.9, type=float, help = 'momentum parameter')
parser.add_argument('--epochs', default=0, type=int,
                    help = 'number of total epochs')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--loaded_epoch', default=0, type=int, help = "continuing training from a save check point")
parser.add_argument('--stages', default='1，2，3，4', type = str, help = 'start epochs of each stages')
parser.add_argument('--start_epochs', default=0, type=int, help = "start training epochs: default 0 in common training and start from loaded_epochs - 1 after loading the check point ")
parser.add_argument('--nc', default=3, type = int,  help = '# of initial channels')


def main():

    torch.manual_seed(777)
    global args, best_prec1, y_t, z_t
    best_prec1 = 0
    args = parser.parse_args()
    args.start_training_time = time.time()

    ######VERBOSE
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        pass
    args.stages = list(map(int, re.split(',|\[|\]', args.stages)[1:-1]))
    print("stages start epochs:", args.stages)
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, args.res_name + '_results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        torch.cuda.manual_seed(123)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    model = models.__dict__[args.model]

    if args.dataset == 'cifar100':
        args.num_classes = 100
    if args.dataset == 'C2':
        args.num_classes = 2
    if args.dataset == 'mnist':
        args.nc = 1

    if args.dataset == "BINARY_mnist":
        args.nc = 1
        args.num_classes = 2

    print(args.num_classes)
    for i in range(10):
        print("i-th")
        model_new = model(args.num_classes, args.nc)
        if args.gpus and len(args.gpus) > 1:
            model_new = torch.nn.DataParallel(model_new, args.gpus)
        for name, param in model_new.named_parameters():
            if name == "module.conv1.weight":
                print(param[0][0][0])

        if 'cuda' in args.type:
             save_checkpoint_iter({
             'epoch': 0,
             'model': args.model,
             'state_dict': model_new.module.state_dict()
         }, initw=True, num=i, path=save_path)  #### Update old model


if __name__ == '__main__':
    main()