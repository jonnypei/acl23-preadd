import random
import os
import sys
import pickle
import json
import math
from collections import defaultdict, namedtuple

from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model
import numpy as np
from tqdm import tqdm, trange
import torch

from util import suppress_stdout
from constants import *

DatasetInfo = namedtuple('DatasetInfo', [])  # TODO fields in list


def collate(batch):
    # [example in batch] = (input, length, label)

    inputs = [b[0] for b in batch]
    lengths = torch.LongTensor([b[1] for b in batch])
    max_length = lengths.max()
    masks = []
    labels = [b[2] for b in batch]
    for i in range(len(inputs)):
        assert len(labels[i]) == lengths[i]
        # Mask
        masks.append(torch.cat([torch.ones(len(inputs[i])).long(), torch.zeros(
            max_length - len(inputs[i])).long()], dim=0))

        # Pad input and corresonding label sequence
        if len(inputs[i]) < max_length:
            inputs[i] = torch.cat([inputs[i], torch.zeros(
                max_length - len(inputs[i])).long()], dim=0)
            labels[i] = torch.cat([torch.FloatTensor(labels[i]), torch.zeros(
                max_length - len(labels[i]), 1)], dim=0)
        else:
            labels[i] = torch.FloatTensor(labels[i])

    inputs = torch.stack(inputs, dim=0)         # batch x seq
    labels = torch.stack(labels, dim=0)       # batch x seq
    masks = torch.stack(masks, dim=0)

    # print("Inputs shape:", inputs.shape)
    # print("Labels shape:", labels.shape)
    # print("Masks shape:", masks.shape)

    return (inputs, lengths, labels, masks)


class Dataset:
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.debug = args.debug
        self.batch_size = args.batch_size
        # self.data_dir = args.data_dir
        self.toxicity = (args.task == 'toxicity')
        self.bias = (args.task == 'bias')

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_string)
        self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        self.gpt_pad_id = self.tokenizer.encode(PAD_TOKEN)[0]

        if self.toxicity:  # toxicity
            sentences = []
            with open(TOXICITY_DATA_PATH, "r") as f:
                for line in f:
                    line = json.loads(line)
                    text = line['prompt']['text'] + \
                        line['continuation']['text']
                    if 'toxicity' not in line['prompt'] or line['prompt']['toxicity'] is None or 'toxicity' not in line['continuation'] or line['continuation']['toxicity'] is None:
                        continue
                    toxicity = max(line['prompt']['toxicity'],
                                   line['continuation']['toxicity'])
                    # if line_data['toxicity']:
                    #     pos_label = line_data['toxicity']
                    #     # pos_label = 1 if line_data['toxicity'] >= 0.5 else 0
                    #     # neg_label = 1 - pos_label
                    #     sentences.append((line_data['text'], [pos_label]))
                    sentences.append((text, [toxicity]))
            train = sentences[:len(sentences)*4//5]
            val = sentences[len(sentences)*4//5:]
        else:  # bias
            sentences = []
            for fname in os.listdir(BIAS_DATA_PATH):
                with open(os.path.join(BIAS_DATA_PATH, fname), "r") as f:
                    # 1 if anti, 0 if pro
                    pos_label = 1 if fname[0] == "a" else 0
                    # neg_label = 1 - pos_label
                    if fname[-3:] == "dev" and 'type2' in fname:  # train
                        sentences.extend(
                            [(line.strip(), [pos_label]) for line in f])
                    # else: # val
                    #     val.extend([(line.strip(), [pos_label]) for line in f]) # don't use test even as validation lmao
            random.shuffle(sentences)
            train = sentences[:len(sentences)*4//5]
            val = sentences[len(sentences)*4//5:]
        self.splits = {}
        if self.debug:
            self.splits['train'] = train[:self.batch_size]
            # for debugging let's make sure we can memorize at least
            self.splits['val'] = train[:self.batch_size]
            # self.splits['train'] = random.sample(train, 10*self.batch_size)
            # self.splits['val'] = random.sample(val, 10*self.batch_size)
        else:
            self.splits['train'] = train
            self.splits['val'] = val

        print('done loading data')
        print('split sizes:')
        for key in ['train', 'val']:
            print(key, len(self.splits[key]))

        if args.dataset_info is not None:
            with open(args.dataset_info, 'rb') as rf:
                self.dataset_info = pickle.load(rf)
        else:
            self.dataset_info = DatasetInfo()  # TODO fields

    def shuffle(self, split, seed=None):
        assert split in ['train', 'val']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])

    def loader(self, split, num_workers=20, indices=None):
        assert split in ['train', 'val']
        data = self.splits[split] if indices is None else [
            self.splits[split][i] for i in indices]
        return torch.utils.data.DataLoader(SplitLoader(data, self), batch_size=self.batch_size, pin_memory=True, collate_fn=collate, num_workers=num_workers)


class SplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, data, parent):
        super(SplitLoader).__init__()
        self.data = data
        self.pos = 0
        self.parent = parent

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self.data):
                raise StopIteration
            # SPLITLOADER IMPLEMENTATION START
            raw_input, label = self.data[self.pos]
            input = self.parent.tokenizer.encode(
                raw_input, return_tensors='pt')[0]
            length = len(input)
            labels = [label] * length
            example = (input, length, labels)  # TODO (might need to change?)
            # SPLITLOADER IMPLEMENTATION END
            valid = True
            self.pos += increment
        return example
