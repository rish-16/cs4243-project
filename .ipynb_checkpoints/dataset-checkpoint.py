from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import random
import numpy as np
import cv2


def combined_dataset(datasets, size):
    combined_dataset = {}
    for name, dataset in datasets.items():
        for class_name, class_data in dataset.items():
            if class_name not in combined_dataset:
                combined_dataset[class_name] = []
            # resize data so they can be stacked
            resized = []
            for data in class_data:
                resized.append(cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA))
            resized = np.stack(resized, axis=0)
            combined_dataset[class_name].append(resized)
    for class_name, lst_datasets in combined_dataset.items():
        combined_dataset[class_name] = np.concatenate(lst_datasets, axis=0)
    return combined_dataset


class ImageDataset(Dataset):
    DATASET_DIR = {True: 'dataset/dataset_train.npy', False: 'dataset/dataset_test.npy'}

    def __init__(self, doodles_list, real_list, doodle_size, real_size, train: bool):
        super(ImageDataset, self).__init__()

        dataset = np.load(self.DATASET_DIR[train], allow_pickle=True)[()]

        doodle_datasets = {name: data for name, data in dataset.items() if name in doodles_list}
        real_datasets = {name: data for name, data in dataset.items() if name in real_list}
        self.doodle_dict = combined_dataset(doodle_datasets, doodle_size)
        self.real_dict = combined_dataset(real_datasets, real_size)

        # sanity check
        assert set(self.doodle_dict.keys()) == set(self.real_dict.keys()), \
            f'doodle and real images label classes do not match'

        # process classes
        label_idx = {}
        for key in self.doodle_dict.keys():
            if key not in label_idx:
                label_idx[key] = len(label_idx)
        self.label_idx = label_idx

        # parse data and labels
        self.doodle_data, self.doodle_label = self._return_x_y_pairs(self.doodle_dict, label_idx)
        self.real_data, self.real_label = self._return_x_y_pairs(self.real_dict, label_idx)

        # data preprocessing
        self.doodle_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(doodle_size),
            transforms.ToTensor(),
            transforms.Normalize((self.doodle_data/255).mean(), (self.doodle_data/255).std())   # IMPORTANT / 255
        ])

        self.real_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(real_size),
            transforms.ToTensor(),
            transforms.Normalize((self.real_data/255).mean(axis=(0, 1, 2)), (self.real_data/255).std(axis=(0, 1, 2)))
        ])

        print(f'Train = {train}. Doodle list: {doodles_list}, \n real list: {real_list}. \n classes: {label_idx.keys()} \n'
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
    doodles = ['sketchy_doodle', 'tuberlin', 'google_doodles']
    reals = ['sketchy_real', 'google_real', 'cifar']
    doodle_size = 64
    real_size = 64
    train_set = ImageDataset(doodles, reals, doodle_size, real_size, train=True)
    val_set = ImageDataset(doodles, reals, doodle_size, real_size, train=False)
    print(len(train_set[0]))
