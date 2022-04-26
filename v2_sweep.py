import torch, sys, argparse
import argparse
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint

from torch.utils.data import Dataset, DataLoader, Sampler
from training_config import doodles, reals, doodle_size, real_size, NUM_CLASSES
from utils import *  # bad practice, nvm

from torchvision import transforms
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
        return max(len(self.doodle_data), len(self.real_data)) # could be arbitrary number

class V2ConvNet(nn.Module):
    def __init__(self, in_c, 
                 num_classes, 
                 channel_list=[64, 128, 192, 256, 512], 
                 pool_option=(1,1), 
                 hidden=256, 
                 dropout=0.2, 
                 add_layers=False):
        super().__init__()
        
        layer1 = nn.Conv2d(in_c, channel_list[0], kernel_size=3)
        layer2 = nn.Conv2d(channel_list[0], channel_list[0], kernel_size=3)
        layers = [layer1, layer2]
        
        for i in range(1, len(channel_list)):
            layers.append(
                nn.Conv2d(channel_list[i-1], channel_list[i], kernel_size=3, stride=2, padding=1, bias=True)
            )
            layers.append(
                nn.Conv2d(channel_list[i], channel_list[i], kernel_size=3, stride=2, padding=1, bias=True)
            )
            layers.append(
                nn.BatchNorm2d(channel_list[i])
            )
            layers.append(
                nn.Dropout(dropout)
            )
            layers.append(nn.ReLU())
            
        self.conv = nn.Sequential(*layers)
        
        self.flatten = nn.AdaptiveAvgPool2d(pool_option)
            
        self.fc = nn.Sequential(*[
            nn.Linear(pool_option[0] * pool_option[1] * channel_list[-1], hidden),
            nn.Linear(hidden, num_classes)
        ])

    def forward(self, x, return_feats=False):
        feats = self.conv(x)
        x = x.view(x.size(0), 512, -1).mean(2)
        x = self.fc(x)

        if return_feats:
            return x, feats

        return x

fix_seed(0)  # zero seed by default
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def build_model(doodle_channels, real_channels, hidden, dropout):
    doodle_model = V2ConvNet(1, 9, channel_list=doodle_channels, hidden=hidden)
    real_model = V2ConvNet(3, 9, channel_list=real_channels, hidden=hidden)
    
    return doodle_model, real_model

def build_dataset(batch_size):
    train_set = ImageDataset(doodles, reals, doodle_size, real_size, train=True)
    val_set = ImageDataset(doodles, reals, doodle_size, real_size, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    return train_loader, val_loader

def train_epoch(epoch, model1, model2, train_loader, val_loader, criterion, optimizer, scheduler):
    loss1_model1 = AverageMeter()
    loss1_model2 = AverageMeter()
    loss2_model1 = AverageMeter()
    loss2_model2 = AverageMeter()
    loss3_combined = AverageMeter()
    acc_model1 = AverageMeter()
    acc_model2 = AverageMeter()

    model1.train()
    model2.train()
    total_loss = 0
    
    c1, c2, t = 0, 0, 0.1
    
    metadata = {}

    for i, (x1, y1, x2, y2) in enumerate(train_loader):
        # doodle, label, real, label
        x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()

        # train model1 (doodle)
        pred1, feats1 = model1(x1, return_feats=True)
        loss_1 = criterion(pred1, y1)  # classification loss
        loss1_model1.update(loss_1.item())
        loss_model1 = loss_1

        # train model2 (real)
        pred2, feats2 = model2(x2, return_feats=True)
        loss_1 = criterion(pred2, y2)  # classification loss
        loss1_model2.update(loss_1.item())
        loss_model2 = loss_1

        loss = loss_model1 + loss_model2
        total_loss += loss.item()

        # statistics
        acc_model1.update(compute_accuracy(pred1, y1))
        acc_model2.update(compute_accuracy(pred2, y2))

        # optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    metadata.update({
        'train acc 1': float(acc_model1.avg),
        'train acc 2': float(acc_model2.avg),
        'train epoch': epoch,
        'l1m1': float(loss1_model1.avg),
        'l1m2': float(loss1_model2.avg),
        'total_loss': float(total_loss / base_bs)
    })

    # validation
    model1.eval(), model1.eval()
    acc_model1.reset(), acc_model2.reset()
    with torch.no_grad():
        for i, (x1, y1, x2, y2) in enumerate(val_loader):
            pred1, feats1 = model1(x1, return_feats=True)
            pred2, feats2 = model2(x2, return_feats=True)
            acc_model1.update(compute_accuracy(pred1, y1))
            acc_model2.update(compute_accuracy(pred2, y2))

    metadata.update({
        'val epoch': epoch,
        'val acc 1': float(acc_model1.avg),
        'val acc 2': float(acc_model2.avg),
    })

    scheduler.step()
  
    return metadata

def train_model(config=None):
    num_epochs, base_bs, base_lr = 20, 512, 2e-2
    
    model1, model2 = build_model(
                        config["channels"],
                        config["channels"],
                        config["dim"],
                        config["dropout"]
                    )
    
    model1 = nn.DataParallel(model1).cuda()
    model2 = nn.DataParallel(model2).cuda()

    optimizer = torch.optim.AdamW(
        params=list(model1.parameters()) + list(model2.parameters()), 
        lr=config["lr"],
        weight_decay=3e-4
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        epoch_data = train_epoch(
            epoch,
            model1, 
            model2, 
            train_loader, 
            val_loader,
            criterion,
            optimizer,
            scheduler
        )

        pprint (epoch_data)

parser = argparse.ArgumentParser()

parser.add_argument('--channels', nargs='+', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--dim', type=int)
parser.add_argument('--lr', type=float)

config = {}

for var, value in parser.parse_args()._get_kwargs():
    if value is not None:
        config[var] = value

print (config)

base_bs = 512
train_loader, val_loader = build_dataset(base_bs)
train_model(config)