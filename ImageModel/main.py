import argparse

import os
import time
import torch.distributed as dist

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from model import GeneralVae, PictureDecoder, PictureEncoder
import pickle
from loss import customLoss
from DataLoader import MoleLoader
from invert import *
import numpy as np
import pandas as pd


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('--deterministic', action='store_true')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')

parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

cudnn.benchmark = True

args = parser.parse_args()

print("opt_level = {}".format(args.opt_level))
print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))



# binding_aff = pd.read_csv("/homes/aclyde11/moldata/moses/norm_binding_aff.csv")
# binding_aff_orig = binding_aff
# binding_aff['id'] = binding_aff['id'].astype('int64')
# binding_aff = binding_aff.set_index('id')
# print(binding_aff.head())

smiles_lookup = pd.read_table("/homes/aclyde11//moses/data/train.csv", names=['smiles', 'split'])
print(smiles_lookup.head())


def main():
    global best_prec1, args

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    model = None
    if args.pretrained:
        print("=> using pre-trained model")
        # model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model")
        # model = models.__dict__[args.arch]()
        checkpoint = torch.load('/home/aclyde11/imageVAE/im_im_small/model/epoch_67.pt', map_location=torch.device('cpu'))
        encoder = PictureEncoder()
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder = PictureDecoder()
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        model = GeneralVae(encoder, decoder)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()

    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    print(args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.85)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    criterion = customLoss()

    train_dataset = MoleLoader(smiles_lookup, num=1000000)
    val_dataset = MoleLoader(smiles_lookup,   num=5000)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)

    best_prec1 = 100000
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        if args.prof:
            break
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (_, data, _) in enumerate(train_loader):
        adjust_learning_rate(args, optimizer, epoch, i, len(train_loader))
        data = data.cuda()
        if args.prof:
            if i > 10:
                break

        # compute output
        if args.prof: torch.cuda.nvtx.range_push("forward")
        recon_batch, mu, logvar, _ = model(data)
        if args.prof: torch.cuda.nvtx.range_pop()
        loss = criterion(recon_batch, data, mu, logvar)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.prof: torch.cuda.nvtx.range_pop()

        if args.prof: torch.cuda.nvtx.range_push("step")
        optimizer.step()
        if args.prof: torch.cuda.nvtx.range_pop()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(args, loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), data.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses, ))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()


    for i, (_, data, _) in enumerate(val_loader):
        data = data.cuda()
        # compute output
        with torch.no_grad():
            recon_batch, mu, logvar, _ = model(data)
            loss = criterion(recon_batch, data, mu, logvar)

        # measure accuracy and record loss

        if args.distributed:
            reduced_loss = reduce_tensor(args, loss.data)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses))


    return losses.avg


if __name__ == '__main__':
    main()