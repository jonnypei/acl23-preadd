import os
import random
import time
import pickle
import sys
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from methods.fudge.data import Dataset
from methods.fudge.model import Model
from methods.fudge.util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from methods.fudge.constants import *


def train(model, dataset, optimizer, criterion, epoch, args, data_start_index):
    model.train()
    if data_start_index == 0:
        dataset.shuffle('train', seed=epoch + args.seed)
    if args.epoch_max_len is not None:
        data_end_index = min(data_start_index +
                             args.epoch_max_len, len(dataset.splits['train']))
        loader = dataset.loader('train', num_workers=args.num_workers, indices=list(
            range(data_start_index, data_end_index)))
        data_start_index = data_end_index if data_end_index < len(
            dataset.splits['train']) else 0
    else:
        loader = dataset.loader('train', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Training: ')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = [tensor.to(args.device) for tensor in batch]
        inputs, lengths, labels, masks = batch
        scores = model(inputs)
        loss = criterion(scores.flatten()[masks.flatten() == 1], labels.flatten()[
                         masks.flatten() == 1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.detach(), len(labels))
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)


@torch.no_grad()
def validate(model, dataset, criterion, epoch, args):
    model.eval()
    loader = dataset.loader('val', num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Validation: ')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = [tensor.to(args.device) for tensor in batch]
        inputs, lengths, labels, masks = batch  # TODO adjust as needed
        scores = model(inputs)
        loss = criterion(scores.flatten()[masks.flatten() == 1], labels.flatten()[
                         masks.flatten() == 1])
        loss_meter.update(loss.detach(), len(labels))
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)
    return loss_meter.avg


def main(args):
    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        model = Model(model_args, dataset.gpt_pad_id)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        data_start_index = checkpoint['data_start_index']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.ckpt, checkpoint['epoch']))
    else:
        model = Model(args, dataset.gpt_pad_id)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_metric = 1e8  # lower is better for cross entropy
        data_start_index = 0
    print('num params', num_params(model))
    criterion = nn.BCEWithLogitsLoss().to(args.device)

    if args.evaluate:
        epoch = 0
        validate(model, dataset, criterion, epoch, args)
        return
    for epoch in range(args.epochs):
        print("TRAINING: Epoch {} at {}".format(epoch, time.ctime()))
        data_start_index = train(
            model, dataset, optimizer, criterion, epoch, args, data_start_index)
        if epoch % args.validation_freq == 0:
            print("VALIDATION: Epoch {} at {}".format(epoch, time.ctime()))
            metric = validate(model, dataset, criterion, epoch, args)

            if metric < best_val_metric:
                print('new best val metric', metric)
                best_val_metric = metric
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': best_val_metric,
                    'optimizer': optimizer.state_dict(),
                    'data_start_index': data_start_index,
                    'args': args
                }, os.path.join(args.save_dir, f'{args.task}_model_best.pth'))


if __name__ == '__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--task', type=str, required=True,
                        choices=['toxicity', 'bias'])

    # SAVE/LOAD
    parser.add_argument('--save_dir', type=str,
                        default=SAVE_PATH, help='where to save ckpts')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='load ckpt from file if given')
    parser.add_argument('--dataset_info', type=str, default=None,
                        help='load dataset info from file if given')

    # MODEL ARCHITECTURE
    parser.add_argument('--model_string', 
                        type=str, 
                        required=True, 
                        choices=fudge_models,
                        help="choose base model")

    # TRAINING
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--epoch_max_len', type=int, default=None)
    parser.add_argument('--validation_freq', type=int,
                        default=1, help='validate every X epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str,
                        default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_workers', type=int, default=20,
                        help='num workers for data loader')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    # PRINTING
    parser.add_argument('--train_print_freq', type=int, default=100,
                        help='how often to print metrics (every X batches)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        assert args.ckpt is not None

    main(args)
