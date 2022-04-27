import torch, copy, random
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from dataset import *
import cv2

from torch.utils.data import Dataset, DataLoader, Sampler
from training_config import doodles, reals, doodle_size, real_size, NUM_CLASSES
from utils import *  # bad practice, nvm
from losses import compute_contrastive_loss_from_feats

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

class GreedySearch:
    def __init__(self, config, model1):
        '''
        config is a dictionary of (hparam:values) pairs
        that need to be tuned.

        model1/model2 is the model class that 
        hasn't been instantiated yet.
        '''

        self.config = config
        self.model1 = model1

    def _train_epoch(self, epoch, model1, train_loader, val_loader, criterion, optimizer, scheduler):
        loss1_model1 = AverageMeter()
        acc_model1 = AverageMeter()

        model1.train()
        total_loss = 0
        
        c1, c2, t = 0, 0, 0.1
        
        metadata = {}

        for i, (x1, y1, x2, y2) in enumerate(train_loader):
            # doodle, label, real, label
            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()

            # train model
            pred1, feats1 = model1(x1, return_feats=True)
            loss_1 = criterion(pred1, y1)  # classification loss
            loss1_model1.update(loss_1.item())

            # statistics
            acc_model1.update(compute_accuracy(pred1, y1))
            acc_model2.update(compute_accuracy(pred2, y2))

            # optimization
            loss_1.backward()
            optimizer.step()
            optimizer.zero_grad()

        metadata.update({
            'train_acc': float(acc_model1.avg),
            'train_epoch': epoch,
            'train_loss': float(loss1_model1.avg),
        })

        # validation
        model1.eval(), model1.eval()
        acc_model1.reset()
        with torch.no_grad():
            for i, (x1, y1, x2, y2) in enumerate(val_loader):
                pred1, feats1 = model1(x1, return_feats=True)
                acc_model1.update(compute_accuracy(pred1, y1))

        metadata.update({
            'val_epoch': epoch,
            'val_acc': float(acc_model1.avg)
        })

        scheduler.step()
    
        return metadata

    def _sample_rest(self, temp_config):
        '''
        Takes in the config and randomly choses 
        the other hparams to be picked.
        '''

        others = {}
        for hparam, values in temp_config.items():
            others[hparam] = values[0]

        return others

    def _build_dataset(self, bs):
        train_set = ImageDataset(doodles, reals, doodle_size, real_size, train=True)
        val_set = ImageDataset(doodles, reals, doodle_size, real_size, train=False)

        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

        return train_loader, val_loader

    def _train_model(self, config):
        '''
        Takes in the config of the model to be trained.
        '''
        
        num_epochs, base_bs, base_lr = 20, 512, 2e-2
        model1 = self.model1(3, 9, config)
        optimizer = torch.optim.AdamW(params=model1.parameters(), lr=config["learning_rate"], weight_decay=3e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        model1 = nn.DataParallel(model1).cuda()

        train_loader, val_loader = self._build_dataset(base_bs)
        
        epochs = []
        for epoch in range(num_epochs):
            # get training metadata
            epoch_data = self._train_epoch(
                epoch,
                model1, 
                train_loader, 
                val_loader,
                criterion,
                optimizer,
                scheduler
            )

            val_acc = epoch_data['val_acc']

            epochs.append(val_acc)
            
        return epochs[-1] # get the validation accuracy from the last epoch

    def tune(self):
        config_cp = copy.deepcopy(self.config)

        final_set = {}
        for hparam, _ in config_cp.items():
            final_set[hparam] = None

        for hparam, values in self.config.items():
            print ("Currently tuning : ", hparam)
            config_cp.pop(hparam)

            best_acc = float('-inf')
            best_choice = None

            if best_choice == None:
                other_hparams = self._sample_rest(config_cp)
            else:
                other_hparams = self._sample_rest(config_cp)
                for done_hparam, val in final_set.items():
                    other_hparams[done_hparam] = val

            print ("Complete list: ", other_hparams)

            for choice in values:
                other_hparams[hparam] = choice
                val_acc = self._train_model(other_hparams)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_choice = choice

            final_set[hparam] = best_choice

        return final_set

def greedy_test():
    config = {
        "channel_list": [
            [64, 192, 256]
        ],
        "dropout": [0.3],
        "hidden_dim": [256],
        "pool_option": [(1,1)],
        "learning_rate": [0.02]
    }

    engine = GreedySearch(config, V2ConvNet)
    optimal_set = engine.tune()

    return optimal_set

print (greedy_test())
