from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import random
import numpy as np


class ImageDataset(Dataset):
    DATASET_DIR = 'dataset/dataset.npy'
    DOODLE_KEY = 'google_doodle'
    REAL_KEY = 'google_real'

    def __init__(self, doodle_size, real_size):
        super(ImageDataset, self).__init__()
        dataset = np.load(self.DATASET_DIR, allow_pickle=True)[()]
        doodle, real = dataset[self.DOODLE_KEY], dataset[self.REAL_KEY]

        # sanity check
        assert set(doodle.keys()) == set(real.keys()), f'doodle and real images label classes do not match'

        # process classes
        label_idx = {}
        for key in doodle.keys():
            if key not in label_idx:
                label_idx[key] = len(label_idx)

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

        print(f'Dataset. Doodle data size {len(self.doodle_data)}, real data size {len(self.real_data)}, '
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
        doodle_idx, real_idx = random.randint(0, len(self.doodle_data) - 1), random.randint(0, len(self.real_data) - 1)
        doodle_data = self.doodle_preprocess(self.doodle_data[doodle_idx])
        real_data = self.real_preprocess(self.real_data[real_idx])
        return doodle_data, self.doodle_label[doodle_idx], real_data, self.real_label[real_idx]

    def __len__(self):
        return max(len(self.doodle_data), len(self.real_data))     # could be arbitrary number


if __name__ == '__main__':
    dataset = ImageDataset(64, 32)
    print('finished!')
