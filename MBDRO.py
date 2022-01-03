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
import torchvision.transforms as transforms
from sklearn import  metrics
from history_code.DRO import DRO
from dataset import get_imbalance_dataset, get_num_classes
from dataloader import get_train_val_test_loader
import wandb


parser = argparse.ArgumentParser(description="Pytorch PLCOVER Training")
parser.add_argument('--results_dir', metavar="RESULTS_DIR", default='./TrainingResults_Sep', help = 'results dir')

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
parser.add_argument('--momentum', default=0, type = float, metavar="M",
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
parser.add_argument('--ith_init_run', default=0, type=int, help = "ith-initial weights")
parser.add_argument('--num_classes', default=10, type=int, help = "classes of different datasets")
parser.add_argument('--im_ratio', default=0.2, type=float, help = "imbalance ratio of datasets")
parser.add_argument('--DR', default=10, type=int, help = 'Decay Rate of Different Stages')
parser.add_argument('--binary', default=False, type=eval, choices=[True, False],
                    help = 'Whether perform binary classification.')
parser.add_argument('--auc', default=False, type = eval, choices=[True, False], help = 'calculating AUC in binary classification')
parser.add_argument('--nc', default=3, type = int,  help = '# of initial channels')
parser.add_argument('--DL', default=3, type = int, help = 'Decay lambda Rate')
parser.add_argument('--curlamda', type=float, default= 200, help = 'The value of current lambda')
parser.add_argument('--init_lamda', type=float, default= 200, help = 'The lambda of first stage')
parser.add_argument('--lamda', default=5, type = float, help = 'parameters of regularization')
parser.add_argument('--obj', default='MBDRO', type=str,
                    help='optimization objective of the loss')

def main():
    physical_start_time = time.time()
    #torch.manual_seed(123)
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    args.start_training_time = time.time()
    wandb.init(config=args, project="mbdro-recover")

    if args.dataset == 'mnist':
        args.input_size = 28

    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    global save_path
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        pass
       # shutil.rmtree(save_path, ignore_errors=True)
       # if not os.path.exists(save_path):
       #  os.makedirs(save_path)



    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, args.res_name +'_results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    logging.info("Total Running Time of the Program: %s ", str((time.time() - physical_start_time) // 60))

    args.stages = list(map(int, re.split(',|\[|\]', args.stages)[1:-1]))
    print("stages start epochs:", args.stages)
    print('learning rate:', args.lr)

    if 'cuda' in args.type:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.gpus = [int(i) for i in args.gpus.split(',')]
        #print(args.gpus[0])
        #torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    logging.info("creating model %s", args.model)
    args.num_classes = get_num_classes(args)
    model = models.__dict__[args.model]
    model_new = model(args.num_classes, args.nc)

    for name, param in model_new.named_parameters():
        if name == "conv1.weight":
            print("conv1.weight:", param[0][0])


    # Data loading code
    torch.manual_seed(777)

    train_loader, val_loader, _ = get_train_val_test_loader(args, None)

    if args.gpus and len(args.gpus) >= 1:
        print("args.type", args.type, "-------------------------------------")
        model_new = torch.nn.DataParallel(model_new)
    #print(model_new)
    #optimizer = torch.optim.SGD(model_new.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    criterion.type(args.type)
    model_new.type(args.type)
    best_prec1 = 0
    wandb.watch(model_new)

    if args.epochs == 200:
        args.stages = [0, 160, 180]
    elif args.epochs == 120:
        args.stages = [0,60]
    print("args.stages:", args.stages)
    print("args.binary:", args.binary, type(args.binary))

    for epoch in range(args.start_epochs, args.epochs):

        adjust_lambda(epoch, args)

        # if epoch == 0: ##################Loade initialization model
        #     if args.binary:
        #         if args.dataset == 'mnist':
        #             init_weights_path = os.path.join(args.results_dir, args.model + "_BINARY_mnist_init_weights")
        #         else:
        #             init_weights_path = os.path.join(args.results_dir, args.model + "_C2_init_weights")
        #     else:
        #         init_weights_path = os.path.join(args.results_dir, args.model + "_" + args.dataset +'_init_weights')
        #     print(init_weights_path)
        #     assert os.path.isdir(init_weights_path), 'Error: no initialized ResNet18 models\' directory found!'
        #
        #     checkpoint = load_checkpoint_init(epoch, ith_init=args.ith_init_run,
        #                                       path=init_weights_path)
        #     if 'cuda' in args.type:
        #         model_new.module.load_state_dict(checkpoint['state_dict'])
        #     else:
        #         model_new.load_state_dict(checkpoint['state_dict'])
        #
        #     print("We are loading the same initializations")
        #     for name, param in model_new.named_parameters():
        #         if name == 'module.conv1.weight':
        #             print("model_new:")
        #             logging.info(param.data[0][0])



        adjust_learning_rate(epoch, args)
        optimizer = torch.optim.SGD(model_new.parameters(), lr=args.curlr)


        wandb.log({"lr": args.curlr, 'lambda': args.curlamda}, step=epoch)

        train_loss, train_prec1, train_prec5, auc_score = train(
            train_loader, model_new, criterion, epoch, optimizer)


        model_new.eval()
        val_loss, val_prec1, val_prec5, auc_score = validate(
             val_loader, model_new, criterion, epoch)

        wandb.log({"train loss": train_loss, 'train acc1': train_prec1, 'train acc5': train_prec5}, step=epoch)
        wandb.log({"test loss": val_loss, 'test acc1': val_prec1, 'test acc5': val_prec5}, step=epoch)

        is_best = val_prec1 > best_prec1
        if 'cuda' in args.type:
            save_checkpoint_epoch({
                'epoch': epoch + 1,
                'model': args.model,
                'batch_size': args.batch_size,
                'state_dict': model_new.module.state_dict()
            }, path=save_path)

        if val_prec1 > best_prec1:
            best_prec1 = val_prec1

        logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec@1 {train_prec1:.3f} \t'
                         'Evaluating Prec@1 {val_prec1:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'AUC Score {auc_score:.4f} \t'
                     'Best Prec@1 {best_prec1:.3f} \n'
                         .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1, auc_score = auc_score, best_prec1 = best_prec1))

        overall_training_time = time.time() - args.start_training_time
        logging.info("Total Running Time of the Program: %s secs", str(overall_training_time))
        results.add(epoch = epoch + 1, train_loss = train_loss, val_loss = val_loss,
                    train_error1 = train_prec1, val_error1 = val_prec1, best_prec1 = best_prec1,
                    train_error5 = train_prec5, val_error5 = val_prec5,
                    overall_training_time = overall_training_time, auc_score = auc_score)
        results.save()

def forward(data_loader, model_new, criterion, epoch=0, training=True, optimizer=None, var = False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    auc_score = 0
    if args.auc:
        target_list = np.array([])
        pred_target_list = np.array([])
    torch.manual_seed(777+epoch)
    max_loss = 0
    max_p = 0
    for i, (inputs, target) in enumerate(data_loader):


        if args.binary:
            target[target < args.num_classes // 2] = 0
            target[target >= args.num_classes // 2] = 1

        data_time.update(time.time() - end)
        if training and i == 0:
            pass

        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var =  Variable(target)
        output = model_new(input_var)
        loss = criterion(output, target_var)
        max_loss = max(max_loss, max(loss))

        dro = DRO()
        loss, p = dro.DRO_KL(loss=loss, p_lambda=args.curlamda)
        max_p = max(max_p, max(p))



        if args.auc:
            target_list = np.concatenate((target_list, target.cpu().numpy()))
            _, pred_target = torch.max(output.data, 1)
            pred_target_list = np.concatenate((pred_target_list , pred_target.cpu().numpy()))

        if type(output) is list:
            output = output[0]


        prec1, prec5= accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Updated new

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(data_loader) - 1:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    if args.auc:
        fpr, tpr, threshold = metrics.roc_curve(target_list, pred_target_list, pos_label=1)
        #print("target_list:", target_list[0:5], "pred_target_list:", pred_target_list[0:5])
        auc_score = metrics.auc(fpr, tpr)

    wandb.log({"Maximum Loss": max_loss.item(), 'p': max(p)}, step=epoch)

    return losses.avg, top1.avg, top5.avg, auc_score


def train(data_loader, model_new, criterion, epoch, optimizer):
    model_new.train()
    return forward(data_loader, model_new, criterion, epoch,
                   training=True, optimizer=optimizer, var = False)


def validate(data_loader, model_new, criterion, epoch):
    return forward(data_loader, model_new, criterion, epoch,
                   training=False, optimizer=None, var = False)



if __name__ == '__main__':
    main()
