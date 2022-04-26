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
    def __init__(self, size=64):
        super(DoodleDataset, self).__init__()
        self.X, self.Y = self.load_datasets()
        self.T = self.get_transforms(size)
    def load_datasets(self):
        d = collapse_datasets(get_doodle_datasets())
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
    def __init__(self, size=64):
        super(RealDataset, self).__init__()
        self.X, self.Y = self.load_datasets()
        self.T = self.get_transforms(size)
    def load_datasets(self):
        d = collapse_datasets(get_real_datasets())
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
    def __init__(self, size=64):
        super(ContrastiveDataset, self).__init__()
        self.X1, self.Y1, self.X2, self.Y2 = self.load_datasets()
        self.T1, self.T2 = self.get_transforms(size)
    def load_datasets(self):
        dd = collapse_datasets(get_doodle_datasets())
        rd = collapse_datasets(get_real_datasets())
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
        idx1 = random.choice(np.where(dataset.Y1 == label)[0])
        idx2 = random.choice(np.where(dataset.Y2 == label)[0])
        return self.X1[idx1], label, self.X2[idx2], label
    def __len__(self):
        return max(len(self.X1), len(self.X2))
    
    
def save_model(checkpoint_dir, checkpoint_name, model):
    """
    Create directory /Checkpoint under exp_data_path and save encoder as cp_name
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, checkpoint_name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # convert to non-parallel form
    torch.save(model.state_dict(), model_path)
    print(f'Saved model at: {model_path}')
    
def load_model(model, checkpoint_path, verbose=True, strict=True):
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