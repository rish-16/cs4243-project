from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import random
import numpy as np


class ImageDataset(Dataset):
    DATASET_DIR = {True: 'dataset/dataset_train.npy', False: 'dataset/dataset_test.npy'}
    DOODLE_KEY = 'sketchy_doodle'  # 'google_doodles'
    REAL_KEY = 'sketchy_real'   # 'google_real'

    def __init__(self, doodle_size, real_size, train: bool):
        super(ImageDataset, self).__init__()
        dataset = np.load(self.DATASET_DIR[train], allow_pickle=True)[()]
        doodle, real = dataset[self.DOODLE_KEY], dataset[self.REAL_KEY]
        self.doodle_dict = doodle
        self.real_dict = real

        # sanity check
        assert doodle.keys() == real.keys(), f'doodle and real images label classes do not match'

        # process classes
        label_idx = {}
        for key in doodle.keys():
            if key not in label_idx:
                label_idx[key] = len(label_idx)
        self.label_idx = label_idx

        # parse data and labels
        self.doodle_data, self.doodle_label = self._return_x_y_pairs(doodle, label_idx)
        self.real_data, self.real_label = self._return_x_y_pairs(real, label_idx)

        # data preprocessing
        self.doodle_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(doodle_size),
            transforms.ToTensor(),
            transforms.Normalize(self.doodle_data.mean(), self.doodle_data.std())
        ])
        self.real_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(real_size),
            transforms.ToTensor(),
            transforms.Normalize(self.real_data.mean(axis=(0,1,2)), self.real_data.std(axis=(0,1,2)))
        ])

        print(f'Dataset {self.DOODLE_KEY} and {self.REAL_KEY}. '
              f'Doodle data size {len(self.doodle_data)}, real data size {len(self.real_data)}, '
              f'ratio {len(self.doodle_data)/len(self.real_data)}')

    def _return_x_y_pairs(self, data_dict, category_mapping):
        xs, ys = [], []
        for key in data_dict.keys():
            data = data_dict[key]
            labels = [category_mapping[key]] * len(data)
            xs.append(data)
            ys.extend(labels)
        return np.concatenate(xs, axis=0), np.array(ys)

    def __getitem__(self, idx):
        # naive sampling scheme - sample with replacement
        # sample label first so that doodle and real data belong to the same category
        label = random.choice(list(self.label_idx.keys()))
        doodle_data = self.doodle_preprocess(random.choice(self.doodle_dict[label]))
        real_data = self.real_preprocess(random.choice(self.real_dict[label]))
        numer_label = self.label_idx[label]
        return doodle_data, numer_label, real_data, numer_label

    def __len__(self):
        return max(len(self.doodle_data), len(self.real_data))     # could be arbitrary number


if __name__ == '__main__':
    dataset = ImageDataset(64, 32, train=True)
    print('done!')
