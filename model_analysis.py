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

from model_training import *

class Tsne:
    def __init__(self, d):
        dl = DataLoader(d, batch_size=len(d))
        self.x, self.y = next(iter(dl))
    def visualise(self, model_arc, model_dir):
        model = load_model(model_arc, model_dir)
        with torch.no_grad():
            preds, feats = model(self.x, return_feats=True)
        self.plot_tsne(feats)
    def plot_tsne(self, feats):
        c = TSNE(n_components=2).fit_transform(feats)
        for i, cl in enumerate(classes):
            plt.scatter(c[self.y==i][:,0], c[self.y==i][:,1], label=cl, s=0.7)
        plt.legend(loc='best')
        plt.show()