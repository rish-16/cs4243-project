from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import random

import numpy as np


class BertDataset(Dataset):
    DATASET_DIR = 'dataset/dataset.npy'

    def __init__(self, doodle_size, real_size):
        super(BertDataset, self).__init__()

        dataset = np.load(self.DATASET_DIR, allow_pickle=True)[()]
        dataset_names = dataset.keys()

        self.
        self.tokenizer = tokenizer
        self.length = length
        self.x = x
        self.return_idx = return_idx
        self.y = torch.tensor(y)
        self.tokens_cache = {}

    def tokenize(self, x):
        dic = self.tokenizer.batch_encode_plus(
            [x],  # input must be a list
            max_length=self.length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        return [x[0] for x in dic.values()]  # get rid of the first dim

    def __getitem__(self, idx):
        int_idx = int(idx)
        assert idx == int_idx
        idx = int_idx
        if idx not in self.tokens_cache:
            self.tokens_cache[idx] = self.tokenize(self.x[idx])
        input_ids, token_type_ids, attention_mask = self.tokens_cache[idx]
        if self.return_idx:
            return input_ids, token_type_ids, attention_mask, self.y[idx], idx, self.x[idx]
        return input_ids, token_type_ids, attention_mask, self.y[idx]

    def __len__(self):
        return len(self.y)


class TrainSamplerMultilCassUnit(Sampler):
    def __init__(self, dataset, sample_unit_size):
        super().__init__(None)
        self.x = dataset.x
        self.y = dataset.y
        self.sample_unit_size = sample_unit_size
        print(f'train sampler with sample unit size {sample_unit_size}')
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        indices = list(range(len(self.y)))
        label_cluster = {}
        for i in indices:
            label = self.y[i].item()
            if label not in label_cluster:
                label_cluster[label] = []
            label_cluster[label].append(i)

        dataset_matrix = []
        for key, value in label_cluster.items():
            random.shuffle(value)
            # value = [key] * len(value)    # debugging use
            num_valid_samples = len(value) // self.sample_unit_size * self.sample_unit_size
            dataset_matrix.append(torch.tensor(value[:num_valid_samples]).view(self.sample_unit_size, -1))

        tuples = torch.cat(dataset_matrix, dim=1).transpose(1, 0).split(1, dim=0)
        tuples = [x.flatten().tolist() for x in tuples]
        random.shuffle(tuples)
        all = sum(tuples, [])

        print(f'from dataset sampler: original dataset size {len(self.y)}, resampled dataset size {len(all)}. '
              f'sample unit size {self.sample_unit_size}')

        return iter(all)

    def __len__(self):
        return self.length


class TransformerEnsembleDataset(Dataset):
    def __init__(self, x, y, tokenizers, lengths):
        super(TransformerEnsembleDataset, self).__init__()
        self.x = x
        self.tokenizers = tokenizers
        self.lengths = lengths
        self.caches = [{} for i in range(len(tokenizers))]
        self.y = torch.tensor(y)

    def tokenize(self, x, i):
        dic = self.tokenizers[i].batch_encode_plus(
            batch_text_or_text_pairs=[x],  # input must be a list
            max_length=self.lengths[i],
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        return [x[0] for x in dic.values()]  # get rid of the first dim

    def __getitem__(self, idx):
        if idx not in self.caches[0]:
            for i in range(len(self.tokenizers)):
                self.caches[i][idx] = self.tokenize(self.x[idx], i)

        return [self.caches[i][idx] for i in range(len(self.tokenizers))], self.y[idx]

    def __len__(self):
        return len(self.y)
