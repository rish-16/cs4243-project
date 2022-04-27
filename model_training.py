import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import random
import numpy as np
import cv2
from dataset_collection import *

classes = ['airplane', 'bird', 'car', 'cat', 'dog', 'frog', 'horse', 'ship', 'truck']
idx2class = {i: c for i, c in enumerate(classes)}
class2idx = {c: i for i, c in enumerate(classes)}

def labelled_dataset(d):
    xs = []
    ys = []
    for c, data in d.items():
        for dat in data:
            xs.append(dat)
            ys.append(class2idx[c])
    return np.asarray(xs), np.asarray(ys)

class DoodleDataset(Dataset):
    def __init__(self, size=64, train=True, split=0.8):
        super(DoodleDataset, self).__init__()
        self.X, self.Y = self.load_datasets(train, size, split)
        self.T = self.get_transforms(size)
    def load_datasets(self, train, size, split):
        traind, testd = collapse_datasets(get_doodle_datasets(), size, split)
        d = traind if train else testd
        X, Y = labelled_dataset(d)
        return X, Y
    def get_transforms(self, size):
        T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((self.X/255).mean(), (self.X/255).std())])
        return T
    def __getitem__(self, idx):
        return self.T(self.X[idx]), self.Y[idx]
    def __len__(self):
        return len(self.X)
    
class RealDataset(Dataset):
    def __init__(self, size=64, train=True, split=0.8):
        super(RealDataset, self).__init__()
        self.X, self.Y = self.load_datasets(train, size, split)
        self.T = self.get_transforms(size)
    def load_datasets(self, train, size, split):
        traind, testd = collapse_datasets(get_real_datasets(), size, split)
        d = traind if train else testd
        X, Y = labelled_dataset(d)
        return X, Y
    def get_transforms(self, size):
        T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((self.X/255).mean(axis=(0, 1, 2)),
                                 (self.X/255).std(axis=(0, 1, 2)))])
        return T
    def __getitem__(self, idx):
        return self.T(self.X[idx]), self.Y[idx]
    def __len__(self):
        return len(self.X)
    
class ContrastiveDataset(Dataset):
    def __init__(self, size=64, train=True, split=0.8):
        super(ContrastiveDataset, self).__init__()
        self.X1, self.Y1, self.X2, self.Y2 = self.load_datasets(train, size, split)
        self.T1, self.T2 = self.get_transforms(size)
    def load_datasets(self, train, size, split):
        traindd, testdd = collapse_datasets(get_doodle_datasets(), size, split)
        trainrd, testrd = collapse_datasets(get_real_datasets(), size, split)
        dd = traindd if train else testdd
        rd = trainrd if train else testrd
        X1, Y1 = labelled_dataset(dd)
        X2, Y2 = labelled_dataset(rd)
        return X1, Y1, X2, Y2
    def get_transforms(self, size):
        T1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((self.X1/255).mean(), (self.X1/255).std())])
        T2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((self.X2/255).mean(axis=(0, 1, 2)),
                                 (self.X2/255).std(axis=(0, 1, 2)))])
        return T1, T2
    def __getitem__(self, idx):
        label = random.choice(list(idx2class.keys()))
        idx1 = random.choice(np.where(self.Y1 == label)[0])
        idx2 = random.choice(np.where(self.Y2 == label)[0])
        return self.T1(self.X1[idx1]), label, self.T2(self.X2[idx2]), label
    def __len__(self):
        return max(len(self.X1), len(self.X2))
    
    
def save_model(checkpoint_dir, checkpoint_name, model, verbose=False):
    """
    Create directory /Checkpoint under exp_data_path and save encoder as cp_name
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, checkpoint_name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # convert to non-parallel form
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f'Saved model at: {model_path}')
    
def load_model(model, checkpoint_path, verbose=False, strict=True):
    """
    Load weights to model and take care of weight parallelism
    """
    assert os.path.exists(checkpoint_path)
    assert f"trained model {checkpoint_path} does not exist"
    try:
        model.load_state_dict(torch.load(checkpoint_path), strict=strict)
    except:
        state_dict = torch.load(checkpoint_path)
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=strict)
    if verbose:
        print(f'Loaded model from: {checkpoint_path}')
    return model

def compute_accuracy(pred, label):
    pred, label = pred.cpu(), label.cpu()
    return (pred.argmax(1) == label).sum().item() / len(label)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
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
        return self.avg
    
class SimilarityDataset(Dataset):
    def __init__(self, size=64, split=0.8):
        super(SimilarityDataset, self).__init__()
        self.idxs, self.X1, self.X2, self.Y = self.load_datasets(split)
        self.T1, self.T2 = self.get_transforms(size)
    def load_datasets(self, split):
        sketchy_pairs = load_dataset("dataset/sketchy_pairs.npy")
        idxs = sketchy_pairs['idxs']
        X1 = sketchy_pairs['doodles']
        X2 = sketchy_pairs['reals']
        Y = sketchy_pairs['classes']
        n = int(split*len(X1))
        return idxs[n:], X1[n:], X2[n:], Y[n:]
    def get_transforms(self, size):
        T1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((self.X1/255).mean(), (self.X1/255).std())])
        T2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((self.X2/255).mean(axis=(0, 1, 2)),
                                 (self.X2/255).std(axis=(0, 1, 2)))])
        return T1, T2
    def __getitem__(self, idx):
        return self.idxs[idx], self.T1(self.X1[idx]), self.T2(self.X2[idx]), self.Y[idx]
    def __len__(self):
        return len(self.X1)

    
def linear_input_units(layers, n_channels=3, size=64):
    x = torch.empty(1, n_channels, size, size)
    for layer in layers:
        x = layer(x)
    return x.size(-1)

def layer2units(n_linear, layer_i):
    return 2**(n_linear-layer_i-1) * 64

class MLP(nn.Module):
    def __init__(self,
                 n_input=64*64,
                 n_classes=9,
                 n_linear=2,
                 dropout=0.1):
        super(MLP, self).__init__()
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = []
        if n_linear == 1:
            self.layers.append(nn.Linear(n_input, n_classes, bias=False))
        else:
            self.layers.append(nn.Linear(n_input, layer2units(n_linear, 1), bias=False))
            self.layers.append(self.act)
            self.layers.append(self.dropout)
            for i in range(2, n_linear):
                self.layers.append(nn.Linear(layer2units(n_linear, i-1), layer2units(n_linear, i), bias=False))
                self.layers.append(self.act)
                self.layers.append(self.dropout)
            self.layers.append(nn.Linear(64, n_classes, bias=False))
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=9,
                 n_filters=32,
                 k_size=3,
                 p_size=2,
                 n_conv=2,
                 n_linear=2,
                 dropout=0.1):
        super(CNN, self).__init__()
        self.p = nn.MaxPool2d((p_size, p_size))
        self.flatten = nn.Flatten()
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [nn.Conv2d(n_channels, n_filters, (k_size, k_size), padding=1), self.p, self.act]
        for _ in range(n_conv-1):
            self.layers.append(nn.Conv2d(n_filters, n_filters, (k_size, k_size), padding=1))
            self.layers.append(self.p)
            self.layers.append(self.act)
        self.layers.append(self.flatten)
        linear_units = linear_input_units(self.layers, n_channels)
        if n_linear == 1:
            self.layers.append(nn.Linear(linear_units, n_classes, bias=False))
        else:
            self.layers.append(nn.Linear(linear_units, layer2units(n_linear, 1), bias=False))
            self.layers.append(self.act)
            self.layers.append(self.dropout)
            for i in range(2, n_linear):
                self.layers.append(nn.Linear(layer2units(n_linear, i-1), layer2units(n_linear, i), bias=False))
                self.layers.append(self.act)
                self.layers.append(self.dropout)
            self.layers.append(nn.Linear(64, n_classes, bias=False))
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.layers(x)
    
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
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
        return self.avg
    
def running_mean(val):
    