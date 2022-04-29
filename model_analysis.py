import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch.nn as nn
import torch.utils.data.dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_training import *

class Tsne:
    def __init__(self, d):
        dl = DataLoader(d, batch_size=len(d))
        self.x, self.y = next(iter(dl))
    def visualise(self, model):
        model = model.eval()
        with torch.no_grad():
            preds, feats = model(self.x, return_feat=True)
        self.plot_tsne(feats)
    def plot_tsne(self, feats):
        c = TSNE(n_components=2).fit_transform(feats)
        for i, cl in enumerate(classes):
            plt.scatter(c[self.y==i][:,0], c[self.y==i][:,1], label=cl, s=0.7)
        plt.legend(loc='best')
        plt.show()
        
class GradCAMUtil:
    def __init__(self, d, bs):
        self.d = d
        self.bs = bs
        self.loader = DataLoader(d, batch_size=bs, shuffle=False)
    
    def visualise_single(self, img, tensor, model, layer, target):
        cam = GradCAM(model, [layer])
        overlay = cam(torch.unsqueeze(tensor, axis=0), [ClassifierOutputTarget(target)])[0,:]
        vis = show_cam_on_image(img/255, overlay, use_rgb=True)
        return vis    
       
    def visualise_random_batch(self, model, n=10):
        tensors, targets = next(iter(self.loader))
        imgs = self.d.X
        
        random_idx = torch.randint(0, self.bs, (n,1))
        
        tensors = tensors[random_idx]
        targets = targets[random_idx]
        imgs = imgs[random_idx]
        
        convs = []
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                convs.append(layer)
            elif isinstance(layer, nn.Sequential):
                for l in layer.children():
                    if isinstance(l, nn.Conv2d):
                        convs.append(l)
                    elif isinstance(l, nn.Sequential):
                        for la in l.children():
                            if isinstance(la, nn.Conv2d):
                                convs.append(la)
                            elif isinstance(la, nn.Sequential):
                                raise Exception()
        layers = convs
        
        for i in range(n):
            fig = plt.figure(figsize=(20, 16))
            
            for j, layer in enumerate([None] + layers):
                if layer == None:
                    plt.subplot(1, len(layers)+1, j+1)
                    plt.imshow(imgs[i].squeeze(0))
                    plt.title("Original")
                else:
                    vis = self.visualise_single(imgs[i].squeeze(0), tensors[i].squeeze(0), model, layer, targets[i].item())
                    plt.subplot(1, len(layers)+1, j+1)
                    plt.imshow(vis)
                    plt.title("Layer {}".format(j+1))
                    
            plt.show()