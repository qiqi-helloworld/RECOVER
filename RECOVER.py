__author__ = 'Qi'
# Created by on 12/18/21.
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
from dataset import get_imbalance_dataset
import models
from torch.autograd import Variable
from preprocess import get_transform
from utils import *
import copy
import re
import torchvision.transforms as transforms
from ast import literal_eval
from torch.nn.utils import clip_grad_norm
from math import ceil
import shutil
import numpy as np
import torch.cuda
import sys
from collections import defaultdict
from torch.nn.parallel._functions import Broadcast
from torch.autograd import Function
import torch.cuda.comm as comm
from sklearn import metrics
# import wandb


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)


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
parser.add_argument('--momentum', default=0.9, type = float, metavar="M",
                    help = "momentum parameter of SHB or SNAG")
parser.add_argument('--scale_size', default=32, type=int, help = 'image scale size for data preprocessing')
parser.add_argument('--input_size', default=32, type=int, help = 'the size of image. e.g. 32 for cifar10, 224 for imagenet')
parser.add_argument('--works', default=8, type=int, help = 'number of threads used for loading data')

parser.add_argument('--print_freq', '-p', default=50, type = int,
                    help = 'print frequency (default:50)')
# number of restart batches: restart_init_loop * batchsize
parser.add_argument('--restart_init_loop', default=1, type = int,
                    help = 'restart minibatch size = restart_init_loop * batchsize')
parser.add_argument('--start_training_time', type = float, help = 'Overall training start time')
parser.add_argument('--lamda', default= 5 , type = float, help = 'The lambda of second stage')
parser.add_argument('--init_lamda', type=float, default= 200, help = 'The lambda of first stage')
parser.add_argument('--curlamda', type=float, default= 200, help = 'The value of current lambda')

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
parser.add_argument('--DR', default=10, type=int, help = 'Decay Rate of Different Stages')

parser.add_argument('--epochs', default=0, type=int,
                    help = 'number of total epochs')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--curlr', default=0.1, type=float, metavar='CURLR',
                    help='current learning rate')
parser.add_argument('--loaded_epoch', default=0, type=int, help = "continuing training from a save check point")
parser.add_argument('--stages', default='1，2，3，4', type = str, help = 'start epochs of each stages')
parser.add_argument('--start_epochs', default=0, type=int, help = "start training epochs: default 0 in common training and start from loaded_epochs - 1 after loading the check point ")
parser.add_argument('--ith_init_run', default=0, type=int, help = "ith-initial weights")
parser.add_argument('--num_classes', default=10, type=int, help = "classes of different datasets")
parser.add_argument('--im_ratio', default=0.2, type=float, help = "imbalance ratio of datasets")
parser.add_argument('--binary', default=False, type=eval, choices=[True, False],
                    help = 'Whether perform binary classification.')
parser.add_argument('--auc', default=False, type = eval, choices=[True, False], help = 'calculating AUC in binary classification')
parser.add_argument('--alg', default='PTL', type = str, help = 'start epochs of each stages')
parser.add_argument('--y_t', default=0, type = float, help = 'Cumulative y_t')
parser.add_argument('--wd', '--weight_decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')



global GPULEN

args = parser.parse_args()
GPULEN = len(args.gpus.split(","))

def update_states(self, **kwargs):
    for key, value in kwargs.iteritems():
        setattr(self, key, value)

class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination1, destination2, num_inputs, *grads):
        ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]

        # print("len(grads): ", len(grads))
        grads_1 = [grads[i:i + num_inputs]
                   for i in range(0, len(grads) // 2, num_inputs)]
        grads_2 = [grads[i:i + num_inputs]
                  for i in range(len(grads) // 2, len(grads), num_inputs)]
        global LASTHALFGRADIENT
        # \nabla f(w_t, \xi_{t+1}), saved for next iteration
        LASTHALFGRADIENT = comm.reduce_add_coalesced(grads_2, destination2)
        return comm.reduce_add_coalesced(grads_1, destination1) # \nabla f(w_t, \xi_{t}), return back to GPU 0 to update w_t


def backward(ctx, *grad_outputs):
    # print("Hello: using self defined backword.")
    # print('grad_out_outputs:', len(grad_outputs))
    return (None,) + ReduceAddCoalesced.apply(0, GPULEN//2, 59, *grad_outputs)

# resnet20 59,resnet32: 95 resnet50:170

Broadcast.backward = backward

def main():
    torch.manual_seed(777)
    global args, best_prec1
    #best_prec1 = 0
    global z_t
    z_t = dict()

    args = parser.parse_args()
    # wandb.init(config = args, project="mbdro-recover")

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
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, args.res_name + '_results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')
    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        torch.cuda.manual_seed(123)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.gpus = [int(i) for i in args.gpus.split(',')]
        #torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    logging.info("creating model %s", args.model)


    if args.dataset == "cifar100":
        args.num_classes = 100
    if args.dataset == "C2":
        args.num_classes = 2
    if args.binary:
        args.num_classes = 2

    model = models.__dict__[args.model]
    model_new = model(args.num_classes)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion.type(args.type)
    model_new.type(args.type)
    overall_training_time = 0

    global state
    state = dict()


    num_parameters = sum([l.nelement() for l in model_new.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True, scale_size=args.scale_size),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False, scale_size=args.scale_size)
    }


    if args.dataset == 'C2':
        traindir = './data/C2/im_train/'
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
            batch_size= args.batch_size//8,
            shuffle=False,
            num_workers=8,
            pin_memory=False)
    else:
        train_data, val_data = get_imbalance_dataset(args.dataset, args.im_ratio, transform)
        torch.manual_seed(777)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.works, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.works, pin_memory=True)

    if args.gpus and len(args.gpus) >= 1:
        model_new = torch.nn.DataParallel(model_new)

    # Load check points from certain number of epochs.
    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint = load_checkpoint_iter(args.loaded_epoch, path=checkpoint_file)
            args.start_epochs = checkpoint['epoch'] - 1
            model_new.module.load_state_dict(checkpoint['state_dict'])


        network_frozen(args, model_new)
        model_new.apply(_weights_init)

    print("# of the model", len(list(model_new.named_parameters())))
    # for name, param in model_new.named_parameters():
    #     # print(name, param.requires_grad)
    optimizer = torch.optim.SGD(model_new.parameters(), lr=args.curlr)

    print('number of epochs: -------------', args.epochs)
    best_prec1 = 0

    if args.epochs == 200:
        args.stages = [0, 160, 180]
    else:
        args.stages = [0, 60]


    if args.resume:
        print("We are loading the SGD pretrianed model")
        filename = args.dataset + '_' + str(args.im_ratio) + '_checkpoint.pth.tar'
        checkpoint = torch.load(filename)
        if 'cuda' in args.type:
            model_new.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_new.load_state_dict(checkpoint['state_dict'])

    for epoch in range(args.start_epochs, args.epochs):
        if epoch % 50 == 0:
            print(epoch, "/", args.epochs, "epochs finished.")


        if 'cuda' in args.type:
            save_checkpoint_epoch({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_new.module.state_dict()
            }, path=save_path)  #### Update old model

        # decay a_t, args.lr
        adjust_learning_rate(epoch, args, optimizer)
        # if epoch > 0 and epoch in args.stages:
        #     # ind = args.stages.index(epoch)
        #     # restart batch size = args.restart_init_loop * batchsize
        #     # args.restart_init_loop = 2*args.restart_init_loop
        #     args.a_t = args.a_t / args.DR
        #     # args.restart_init_loop = args.restart_init_loop * args.DR

        if epoch <= args.stages[1]:
            args.curlamda =  args.init_lamda
        else:
            args.curlamda = args.lamda


        args.start_training_time = time.time()
        # wandb.log({"lr": args.curlr, 'lambda': args.curlamda}, step=epoch)

        train_loss, train_prec1, train_prec5, _ = train(
            train_loader, model_new, criterion, save_path, epoch, optimizer)
        overall_training_time = overall_training_time + (time.time() - args.start_training_time)


        val_loss, val_prec1, val_prec5, auc_score = validate(
            val_loader, model_new,  criterion, save_path, epoch)


        # wandb.log({"train loss": train_loss, 'train acc1': train_prec1, 'train acc5': train_prec5}, step=epoch)
        # wandb.log({"test loss": val_loss, 'test acc1': val_prec1, 'test acc5': val_prec5}, step=epoch)

        if best_prec1 < val_prec1:
            best_prec1 = val_prec1

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.4f} \t'
                     'AUC score {auc_score:.3f} \t'
                     'Best Validation Accuracy {best_val:.3f}'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             auc_score=auc_score, best_val = best_prec1))

        # wandb.log({"best_prec1": best_prec1}, step=epoch)
        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=train_prec1, val_error1=val_prec1, best_prec1 = best_prec1,
                    train_error5=train_prec5, val_error5=val_prec5,
                    overall_training_time=overall_training_time)
        results.save()


def forward(data_loader, model_new, criterion, save_path, epoch=0, training=True):
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()
    auc_score = 0

    j = 0

    if training:

        global inputs_1, target_1, per_epoch_time

        train_iter = enumerate(data_loader)
        data_time.update(time.time() - end)
        print("Data Load Time:", time.time() - end)
        done_looping = False
        if epoch == args.stages[0]:
            _, (inputs_1, target_1) = next(train_iter)
            if args.binary:
                target_1[target_1 < args.num_classes // 2] = 0
                target_1[target_1 >= args.num_classes // 2] = 1
            j += 1

        _, (inputs_2, target_2) = next(train_iter)
        if args.binary:
            target_2[target_2 < args.num_classes // 2] = 0
            target_2[target_2 >= args.num_classes // 2] = 1

        j += 1

        loop_start_time = time.time()
        while not done_looping:
            inputs = torch.cat([inputs_1, inputs_2])
            target = torch.cat([target_1, target_2])

            t1 = time.time()
            input_var = Variable(inputs.type(args.type), volatile=not training)
            if args.gpus is not None:
                target = target.cuda()
            target_var = Variable(target)
            output, _= model_new(input_var)  # 0, 1
            #torch.cuda.synchronize()
            #print("Forward Time", time.time() - t1)

            #torch.cuda.synchronize()
            #t1 = time.time()

            # print(output, target.view(-1,1))
            loss1 = criterion(output[0:args.batch_size], target_var[0:args.batch_size])
            loss2 = criterion(output[args.batch_size:], target_var[args.batch_size:])

            loss1_max = torch.Tensor.detach(torch.max(loss1))
            loss2_max = torch.Tensor.detach(torch.max(loss2))

            exp_loss_1 = torch.mean(torch.exp((loss1 - loss1_max) / args.curlamda))
            exp_loss_2 = torch.mean(torch.exp((loss2 - loss2_max) / args.curlamda))

            exp_loss = (exp_loss_1 + exp_loss_2)/2

            loss = loss1_max + args.curlamda * torch.log(
                torch.mean(torch.exp((loss1 - loss1_max) / args.curlamda)))  # + args.lamda * training_data_size


            prec1, prec5 = accuracy(output[0:args.batch_size], target_var[0:args.batch_size].view(-1,1), topk=(1, 5))
            losses.update(loss.item(), inputs_1.size(0))
            top1.update(prec1.item(), inputs_1.size(0))
            top5.update(prec5.item(), inputs_1.size(0))
            model_new.zero_grad()
            exp_loss.backward()

            args.y_t = exp_loss_1 * torch.exp(
                         loss1_max / args.curlamda) + (1 - args.a_t) * args.y_t
            for name, param in model_new.named_parameters():  # load the name and value of every layer.
                if not args.resume or 'linear' in name and 'layer3.4' in name:
                    if name not in state.keys() or epoch in args.stages:
                        state[name] = 0
                    state[name] =  param.grad * min(torch.exp(
                             loss1_max/ args.curlamda), args.curlamda)+ (1-args.a_t) * state[name]

                    param.data.add_(-args.curlr, args.curlamda * state[name] / args.y_t + args.wd * param.data)

            # Update State and Y_t
            args.y_t = args.y_t - exp_loss_2.item() * torch.exp(loss2_max / args.curlamda)
            i = 0
            for name, param in model_new.named_parameters():
                # state[name] = state[name] - LASTHALFGRADIENT[i].to(0)
                if args.resume:
                    if  'linear' in name and 'layer3.4' in name:
                        state[name] = state[name] - torch.exp(loss2_max / args.curlamda) * LASTHALFGRADIENT[i-8].to(0)
                        i += 1
                else:
                    state[name] = state[name] - torch.exp(loss2_max / args.curlamda) * LASTHALFGRADIENT[i].to(0)
                    i += 1

            if j <= len(data_loader) - 1:
                _, (inputs_1, target_1) = next(train_iter)
                if args.binary:
                    target_1[target_1 < args.num_classes // 2] = 0
                    target_1[target_1 >= args.num_classes // 2] = 1

                inputs_1, inputs_2 = inputs_2, inputs_1
                target_1, target_2 = target_2, target_1
                j += 1
            else:
                inputs_1 = inputs_2
                target_1 = target_2
                done_looping = True

            iter_time = time.time() - end
            if j % args.print_freq == 0:

                 logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                              'Iter Time {iter_time:.3f}\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     epoch, j, len(data_loader),
                     phase='TRAINING' if training else 'EVALUATING',
                     iter_time=iter_time,
                     data_time=data_time, loss=losses, top1=top1, top5=top5))
            stop = time.time()

        print("Assign_time:", stop - loop_start_time)
        batch_time.update(time.time() - end)
        end = time.time()

    else:
        if args.auc:
            target_list = np.array([])
            pred_target_list = np.array([])

        for i, (inputs, target) in enumerate(data_loader):
            if args.binary:
                target[target < args.num_classes // 2] = 0
                target[target >= args.num_classes // 2] = 1

            data_time.update(time.time() - end)
            if args.gpus is not None:
                target = target.cuda()
            input_var = Variable(inputs.type(args.type), volatile=not training)
            target_var = Variable(target)
            output1, _ = model_new(input_var)
            loss1 = criterion(output1, target_var)
            loss1_max = torch.Tensor.detach(torch.max(loss1))

            loss1 = loss1_max + args.lamda * torch.log(
                torch.mean(torch.exp((loss1_max - loss1_max) / args.lamda)))  # + args.lamda * training_data_size

            if args.auc:

                target_list = np.concatenate((target_list, target.cpu().numpy()))
                _, pred_target = torch.max(output1.data, 1)
                pred_target_list = np.concatenate((pred_target_list, pred_target.cpu().numpy()))


            if type(output1) is list:
                output1 = output1[0]
            prec1, prec5 = accuracy(output1.data, target, topk=(1, 2))
            losses.update(loss1.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            iter_time = time.time() - end

            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}/{1}][{2}/{3}]\t'
                             'Iter Time {iter_time:.3f}\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, args.epochs, i, len(data_loader),
                    phase='TRAINING' if training else 'EVALUATING',
                    iter_time=iter_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

            # logging.info("Cost of Time Per-Epoch %s secs", str(time.time() - end))
            batch_time.update(time.time() - end)
            end = time.time()


        if args.auc:
            fpr, tpr, threshold = metrics.roc_curve(target_list, pred_target_list, pos_label=1)
            auc_score = metrics.auc(fpr, tpr)
    return losses.avg, top1.avg, top5.avg, auc_score




def train(data_loader, model_new, criterion, save_path, epoch, optimizer):
    model_new.train()

    return forward(data_loader, model_new, criterion, save_path, epoch, training=True)

def validate(data_loader, model_new, criterion, save_path, epoch):
    model_new.eval()
    return forward(data_loader, model_new, criterion, save_path, epoch, training=False)




def network_frozen(args, model):
    last_block_number = 0
    if args.model == 'resnet20':
        last_block_number = 2
    last_block_pattern = 'layer3.' + str(last_block_number)
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
        if 'linear' not in param_name and last_block_pattern not in param_name:
                param.requires_grad = False



if __name__ =='__main__':
    main()