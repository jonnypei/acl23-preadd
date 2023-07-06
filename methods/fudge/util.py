import os
import time
import sys
from contextlib import contextmanager

import torch

from methods.fudge.constants import *
from methods.fudge.model import Model

loaded_fudge_models = {}


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clamp(x, limit):
    return max(-limit, min(x, limit))


def pad_to_length(tensor, length, dim, value=0):
    """
    Pad tensor to given length in given dim using given value (value should be numeric)
    """
    assert tensor.size(dim) <= length
    if tensor.size(dim) < length:
        zeros_shape = list(tensor.shape)
        zeros_shape[dim] = length - tensor.size(dim)
        zeros_shape = tuple(zeros_shape)
        return torch.cat([tensor, torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device).fill_(value)], dim=dim)
    else:
        return tensor


def pad_mask(lengths: torch.LongTensor) -> torch.ByteTensor:
    """
    Create a mask of seq x batch where seq = max(lengths), with 0 in padding locations and 1 otherwise. 
    """
    # lengths: bs. Ex: [2, 3, 1]
    max_seqlen = torch.max(lengths)
    expanded_lengths = lengths.unsqueeze(0).repeat(
        (max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
    indices = torch.arange(max_seqlen).unsqueeze(1).repeat(
        (1, lengths.size(0))).to(lengths.device)  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs
    return expanded_lengths > indices


class ProgressMeter(object):
    """
    Display meter
    """

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries.append(time.ctime(time.time()))
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """
    Display meter
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def load_fudge_model(path, task, model_string, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if path in loaded_fudge_models:
        return loaded_fudge_models[path]
    else:
        class FakeArgs:
            def __init__(self, task, model_string):
                self.task = task
                self.model_string = model_string
        model = Model(FakeArgs(task, model_string), None)
        model.load_state_dict(torch.load(
            path, map_location='cpu')['state_dict'])
        model.eval()
        model = model.to(device)
        loaded_fudge_models[path] = model
        return model
