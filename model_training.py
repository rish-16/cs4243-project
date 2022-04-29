import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import random
import numpy as np
import cv2
import copy
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
    return np.asarray(xs), np.asarray(ys).astype(np.int64)

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
        return self.T1(self.X1[idx1]), self.T2(self.X2[idx2]), label
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
                 n_input,
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
    def forward(self, x, return_feat=False):
        feat = self.layers[:-1](x.flatten(1))
        x = self.layers[-1](feat)
        if return_feat:
            return x, feat
        return x


# class CNN(nn.Module):
#     def __init__(self,
#                  n_channels,
#                  n_classes=9,
#                  n_filters=32,
#                  k_size=3,
#                  p_size=2,
#                  n_conv=2,
#                  n_linear=2,
#                  dropout=0.1):
#         super(CNN, self).__init__()
#         self.p = nn.MaxPool2d((p_size, p_size))
#         self.flatten = nn.Flatten()
#         self.act = nn.LeakyReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.layers = [nn.Conv2d(n_channels, n_filters, (k_size, k_size), padding=1), self.p, self.act]
#         for _ in range(n_conv-1):
#             self.layers.append(nn.Conv2d(n_filters, n_filters, (k_size, k_size), padding=1))
#             self.layers.append(self.p)
#             self.layers.append(self.act)
#         self.layers.append(self.flatten)
#         linear_units = linear_input_units(self.layers, n_channels)
#         if n_linear == 1:
#             self.layers.append(nn.Linear(linear_units, n_classes, bias=False))
#         else:
#             self.layers.append(nn.Linear(linear_units, layer2units(n_linear, 1), bias=False))
#             self.layers.append(self.act)
#             self.layers.append(self.dropout)
#             for i in range(2, n_linear):
#                 self.layers.append(nn.Linear(layer2units(n_linear, i-1), layer2units(n_linear, i), bias=False))
#                 self.layers.append(self.act)
#                 self.layers.append(self.dropout)
#             self.layers.append(nn.Linear(64, n_classes, bias=False))
#         self.layers = nn.Sequential(*self.layers)
#     def forward(self, x, return_feat=False):
#         feat = self.layers[:-1](x)
#         x = self.layers[-1](feat)
#         if return_feat:
#             return x, feat
#         return x


def convbn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class CNN(nn.Module):
    CHANNELS = [64, 128, 192, 256, 512]
    POOL = (1, 1)
    def __init__(self, in_c, num_classes, dropout=0.2, add_layers=False):
        super(CNN).__init__()
        layer1 = convbn(in_c, self.CHANNELS[1], kernel_size=3, stride=2, padding=1, bias=True)
        layer2 = convbn(self.CHANNELS[1], self.CHANNELS[2], kernel_size=3, stride=2, padding=1, bias=True)
        layer3 = convbn(self.CHANNELS[2], self.CHANNELS[3], kernel_size=3, stride=2, padding=1, bias=True)
        layer4 = convbn(self.CHANNELS[3], self.CHANNELS[4], kernel_size=3, stride=2, padding=1, bias=True)
        pool = nn.AdaptiveAvgPool2d(self.POOL)
        layer1_2 = convbn(self.CHANNELS[1], self.CHANNELS[1], kernel_size=3, stride=1, padding=0, bias=True)
        layer2_2 = convbn(self.CHANNELS[2], self.CHANNELS[2], kernel_size=3, stride=1, padding=0, bias=True)
        layer3_2 = convbn(self.CHANNELS[3], self.CHANNELS[3], kernel_size=3, stride=1, padding=0, bias=True)
        layer4_2 = convbn(self.CHANNELS[4], self.CHANNELS[4], kernel_size=3, stride=1, padding=0, bias=True)
        self.layers = nn.Sequential(layer1, layer1_2, layer2, layer2_2, layer3, layer3_2, layer4, layer4_2, pool)
        self.nn = nn.Linear(self.POOL[0] * self.POOL[1] * self.CHANNELS[4], num_classes)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, return_feats=False):
        feats = self.layers(x).flatten(1)
        x = self.nn(self.dropout(feats))
        if return_feats:
            return x, feats

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
    
def compute_sim_matrix(feats):
    """
    Takes in a batch of features of size (bs, feat_len).
    """
    sim_matrix = F.cosine_similarity(feats.unsqueeze(2).expand(-1, -1, feats.size(0)),
                                     feats.unsqueeze(2).expand(-1, -1, feats.size(0)).transpose(0, 2),
                                     dim=1)
    return sim_matrix


def compute_target_matrix(labels):
    """
    Takes in a label vector of size (bs)
    """
    label_matrix = labels.unsqueeze(-1).expand((labels.shape[0], labels.shape[0]))
    trans_label_matrix = torch.transpose(label_matrix, 0, 1)
    target_matrix = (label_matrix == trans_label_matrix).type(torch.float)
    return target_matrix


def contrastive_loss(pred_sim_matrix, target_matrix, temperature):
    return F.kl_div(F.softmax(pred_sim_matrix / temperature).log(), F.softmax(target_matrix / temperature),
                    reduction="batchmean", log_target=False)


def compute_contrastive_loss_from_feats(feats, labels, temperature):
    sim_matrix = compute_sim_matrix(feats)
    target_matrix = compute_target_matrix(labels)
    loss = contrastive_loss(sim_matrix, target_matrix, temperature)
    return loss

class CNNCL(nn.Module):
    def __init__(self,
                 n_classes=9,
                 n_filters=32,
                 k_size=3,
                 p_size=2,
                 n_conv=2,
                 n_linear=2,
                 dropout=0.1,
                 c1=1,
                 c2=1,
                 t=0.1):
        super(CNNCL, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.t = t
        self.dmodel = CNN(1, n_classes, n_filters, k_size, p_size, n_conv, n_linear, dropout)
        self.rmodel = CNN(3, n_classes, n_filters, k_size, p_size, n_conv, n_linear, dropout)
    def forward(self, x1, x2):
        pred1, feat1 = self.dmodel(x1, return_feat=True)
        pred2, feat2 = self.rmodel(x2, return_feat=True)
        return pred1, feat1, pred2, feat2
    def loss(self, pred1, feat1, pred2, feat2, y):
        xent = nn.CrossEntropyLoss()
        xent_loss1 = xent(pred1, y)
        xent_loss2 = xent(pred2, y)
        cont_loss1 = compute_contrastive_loss_from_feats(feat1, y, self.t)
        cont_loss2 = compute_contrastive_loss_from_feats(feat2, y, self.t)
        cont_loss3 = compute_contrastive_loss_from_feats(feat1*feat2, y, self.t)
        loss = (xent_loss1 + xent_loss2 
                + self.c1 * (cont_loss1 + cont_loss2)
                + self.c2 * cont_loss3)
        return loss

    
class Trainer:
    def __init__(self, model, trainset, valset, epochs, bs):
        self.model = model
        self.contrastive = isinstance(trainset, ContrastiveDataset)
        self.train_loader = DataLoader(trainset, batch_size=bs)
        self.val_loader = DataLoader(valset, batch_size=len(valset))
        self.epochs = epochs
        self.history = None
        self.best_model = None
        self.best_perf = None
        self.optimizer = torch.optim.AdamW(params=list(self.model.parameters()), lr=1e-3, weight_decay=3e-4)
        self.xent = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
    def track(self, metric, value):
        if self.history is None:
            self.history = {
                "epoch": [],
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []}
        assert metric in self.history
        self.history[metric].append(value)
    def train_epoch(self):
        self.model.train()
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        if not self.contrastive:
            for i, (x, y) in enumerate(self.train_loader):
                pred = self.model(x)
                loss = self.xent(pred, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                acc = compute_accuracy(pred, y)
                avg_loss.update(loss.item())
                avg_acc.update(acc)
        else:
            for i, (x1, x2, y) in enumerate(self.train_loader):
                pred1, feat1, pred2, feat2 = self.model(x1, x2)
                loss = self.model.loss(pred1, feat1, pred2, feat2, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                acc = (compute_accuracy(pred1, y) + compute_accuracy(pred2, y)) / 2
                avg_loss.update(loss.item())
                avg_acc.update(acc)
        return avg_acc.avg, avg_loss.avg
    def evaluate_epoch(self):
        self.model.eval()
        if not self.contrastive:
            x, y = next(iter(self.val_loader))
            with torch.no_grad():
                pred, feat = self.model(x, return_feat=True)
                loss = self.xent(pred, y)
            _, yhat = torch.max(pred, 1)
            acc = compute_accuracy(pred, y)
        else:
            x1, x2, y = next(iter(self.val_loader))
            with torch.no_grad():
                pred1, feat1, pred2, feat2 = self.model(x1, x2)
                loss = self.model.loss(pred1, feat1, pred2, feat2, y)
            acc = (compute_accuracy(pred1, y) +  compute_accuracy(pred2, y)) / 2
        return acc, loss.item()
    def train(self, verbose=False):
        for epoch in range(self.epochs):
            train_acc, train_loss = self.train_epoch()
            val_acc, val_loss = self.evaluate_epoch()
            self.track("epoch", epoch)
            self.track("train_loss", train_loss)
            self.track("train_acc", train_acc)
            self.track("val_loss", val_loss)
            self.track("val_acc", val_acc)
            if verbose:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")
            self.update_best_model(self.model, val_acc)
        return self.history
    def update_best_model(self, model, perf):
        if self.best_model is None or perf > self.best_perf:
            self.best_model = copy.deepcopy(model)
            self.best_perf = perf
    def save(self, idx, params=None, verbose=False):
        save_model(f"exp_data/{idx}", f"{idx}_model.pt", self.best_model, verbose=verbose)
        save_history(f"exp_data/{idx}", f"{idx}_history.pkl", self.history, verbose=verbose)
        if params:
            save_params(f"exp_data/{idx}", f"{idx}_params.pkl", params, verbose=verbose)
    def plot(self, metric='val_acc'):
        plt.plot(self.history['epoch'], self.history[metric], label=f"{metric}")
        plt.legend(loc='best')
        plt.show()
        
import pickle
def save_pickle(data, f, verbose=False):
    with open(f, 'wb') as file:
        pickle.dump(data, file)
    if verbose:
        print(f"Saved file at: {f}")
def load_pickle(f, verbose=False):
    with open(f, 'rb') as file:
        data = pickle.load(file)
    if verbose:
        print(f"Loaded file at: {f}")
    return data

def save_history(checkpoint_dir, checkpoint_name, history, verbose=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    hist_path = os.path.join(checkpoint_dir, checkpoint_name)
    save_pickle(history, hist_path)
    if verbose:
        print(f'Saved history at: {hist_path}')
def load_history(checkpoint_path, verbose=False):
    assert os.path.exists(checkpoint_path)
    history = load_pickle(checkpoint_path)
    if verbose:
        print(f"Loaded history from: {checkpoint_path}")
    return history

def save_params(checkpoint_dir, checkpoint_name, params, verbose=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    params_path = os.path.join(checkpoint_dir, checkpoint_name)
    save_pickle(params, params_path)
    if verbose:
        print(f'Saved hyperparameters at: {hist_path}')
def load_params(checkpoint_path, verbose=False):
    assert os.path.exists(checkpoint_path)
    params = load_pickle(checkpoint_path)
    if verbose:
        print(f"Loaded hyperparameters from: {checkpoint_path}")
    return params