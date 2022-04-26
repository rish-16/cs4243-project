import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import Counter
import os
import imageio
import glob
from torch.utils.data import Dataset, TensorDataset, DataLoader
import cv2

def plot_dataset_dist(d, title="untitled"):
    plt.title(title)
    plt.bar([c for c in d.keys()], [x.shape[0] for x in d.values()])
    plt.show()
    
    
def plot_dataset_mean(d):
    cats = list(d.keys())
    col = 5 if len(cats) > 8 else 4
    fig, ax = plt.subplots(2, col, figsize=(15,6))
    plt.axis('off')
    for r in range(2):
        for c in range(col):
            try:
                mean_img = d[cats[r * col + c]].mean(axis=0).astype(int)
                ax[r,c].axis('off')
                ax[r,c].set_title(cats[r * col + c])
                ax[r,c].set_xticks([])
                ax[r,c].set_yticks([])
                ax[r,c].imshow(mean_img, cmap='gray')
            except:
                break
    plt.show()