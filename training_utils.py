import torch, copy, random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pprint import pprint

class V1MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super(DoodleMLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)

    def forward(self, x, return_feats=False):
        x = x.flatten(1)    # flatten a pic into a vector
        x = self.l1(x)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.dropout(x)
        x = self.l2(x)
        feat = x
        x = self.relu(x)
        # x = self.bn2(x)
        x = self.dropout(x)
        x = self.l3(x)

        if return_feats:
            return x, feat

        return x

    def train_epoch(self, train_loader):
        pass

class V2ConvNet(nn.Module):
    def __init__(self, in_c, num_classes, config):
        super().__init__()

        channel_list = config['channel_list']
        pool_option = config['pool_option']
        hidden = config['hidden_dim']
        dropout = config['dropout']
        
        layer1 = nn.Conv2d(in_c, channel_list[0], kernel_size=3)
        layers = [layer1]
        
        for i in range(1, len(channel_list)):
            layers.append(
                nn.Conv2d(channel_list[i-1], channel_list[i], kernel_size=3, stride=2, padding=1, bias=True)
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

    def train_epoch(self, train_loader, optimizer, loss_fn, ):
        for idx, (x, y) in enumerate(train_loader):
            
            pred = self(x, y)

    def evaluate(self, val_loader):
        pass

class Trainer:
    def __init__(self, model, nepochs, trainset, valset, batchsize):
        self.model = model
        self.train_loader = DataLoader(trainset, batch_size=batchsize)
        self.val_loader = DataLoader(valset, batch_size=valset.shape[0])
        self.nepochs = nepochs

    def fit(self, verbose=False, return_history=False):
        total_loss = 0
        total_acc = 0
        history = {
            "epochs": [],
            "train_loss": [],
            "val_acc": []
        }
        for epoch in range(self.nepochs):
            epoch_train_acc, epoch_train_loss = self.model.train_epoch(self.train_loader)
            epoch_val_acc, epoch_val_loss = self.model.evaluate(self.val_loader)

            total_loss += epoch_val_loss
            total_acc += epoch_val_acc

            history['epochs'].append(epoch)
            history['train_loss'].append(epoch_train_loss)
            history["val_acc"].append(epoch_val_acc)

            if verbose:
                print ("Epoch: {} | Train Loss: {} | Val Acc: {}".format(epoch, epoch_train_loss, epoch_val_acc))

        avg_loss = total_loss / self.nepochs
        avg_acc = total_acc / self.nepochs

        if return_history:
            return avg_acc, avg_loss, history

        return avg_acc, avg_loss
