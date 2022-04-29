import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import torch.nn.functional as F

import torch.nn as nn
import torch.utils.data.dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from model_training import *
from dataset_collection import *

def print_accuracy(y, yhat, classes):
    cm = confusion_matrix(y, yhat)
    accuracy = cm.diagonal()/cm.sum(1)
    print("{:>12} {:>10}".format("", "accuracy"), end="\n\n")
    for c, a in zip(classes, accuracy):
        print("{:>12} {:>10.2f}".format(c, a))
    return accuracy

def evaluate_model(model, model_dir, dataset, classes, report=False):
    dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x, y = next(iter(dl))
    model = load_model(model, model_dir).eval()
    with torch.no_grad():
        preds, feats = model(x, return_feats=True)
        _, yhat = torch.max(preds, 1)
    accuracy = print_accuracy(y, yhat, classes)
    if report:
        print(classification_report(y, yhat, target_names=classes))
    return accuracy
    
    
class SimilarityDataset(Dataset):
    def __init__(self, size=64, split=0.8):
        super(SimilarityDataset, self).__init__()
        self.idxs, self.X1, self.X2, self.Y = self.load_datasets()
        self.T1, self.T2 = self.get_transforms(size)
    def load_datasets(self):
        sketchy_pairs = get_sketchy_pairs()
        idxs = sketchy_pairs['idxs']
        X1 = sketchy_pairs['doodles']
        X2 = sketchy_pairs['reals']
        Y = sketchy_pairs['labels']
        return idxs, X1, X2, Y
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
    
def print_similarity(feats1, feats2, y, classes):
    def scale_feat(x):
        return (x - x.min())/(x.max() - x.min())
    assert feats1.shape[1] == feats2.shape[1]
    sims = F.cosine_similarity(scale_feat(feats1), scale_feat(feats2))
    print("{:>12} {:>10}".format("", "similarity"), end="\n\n")
    for i, c in enumerate(classes):
        print("{:>12} {:>10.2f}".format(c, sims[y == i].mean().item()))
    print("\n{:>12} {:>10.2f}".format("aggregate", sims.mean().item()))
    return sims

class Similarity:
    def __init__(self):
        d = SimilarityDataset()
        dl = DataLoader(d, batch_size=len(d), shuffle=False)
        self.idx, self.x1, self.x2, self.y = next(iter(dl))
        del d, dl
    def evaluate(self, dmodel, rmodel):
        dmodel.eval()
        rmodel.eval()
        with torch.no_grad():
            preds1, feats1 = dmodel(self.x1, return_feat=True)
            preds2, feats2 = rmodel(self.x2, return_feat=True)
        sims = print_similarity(feats1, feats2, self.y, classes)
        return sims
